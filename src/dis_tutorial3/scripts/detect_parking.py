#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data


from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import math

from ultralytics import YOLO
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from gtts import gTTS
import pygame
from tempfile import TemporaryFile
import time

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1
        self.elipses = []
        self.parkings = []

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        # New subscription for the top camera image
        self.top_camera_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.top_camera_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.top_pointcloud_callback, qos_profile_sensor_data)


        # Publiser for the visualization markers
        self.elipse_marker_pub = self.create_publisher(Marker,"/ringMarker", QoSReliabilityPolicy.BEST_EFFORT)

        self.parking_marker_pub = self.create_publisher(Marker,"/parkMarker", QoSReliabilityPolicy.BEST_EFFORT)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)


    def image_callback(self, data):

        self.elipses = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        blue = cv_image[:,:,0]
        green = cv_image[:,:,1]
        red = cv_image[:,:,2]

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # gray = red

        # Apply Gaussian Blur
        gray = cv2.GaussianBlur(gray,(3,3),0)

        # Do histogram equalization
        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)


        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Detected contours", gray)
        cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)


        # Find two elipses with same centers
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
                if dist >= 5:
                    continue

                # The rotation of the elipses should be whitin 4 degrees of eachother
                if angle_diff>4:
                    continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                
                # # The widths of the ring along the major and minor axis should be roughly the same
                # border_major = (le[1][1]-se[1][1])/2
                # border_minor = (le[1][0]-se[1][0])/2
                # border_diff = np.abs(border_major - border_minor)

                if e1[0][1] < cv_image.shape[1]/2:
                    self.get_logger().info(f"UPPER HALF OF IMAGE AAAAAAAAAAAAAAAAAAAAAAA")
                    continue
                    
                candidates.append((e1,e2))

        print("Processing is done! found", len(candidates), "candidates for rings")

        # Plot the rings on the image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            self.elipses.append(center)
            print(center, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

            cv2.imshow("image", cv_image)
            key = cv2.waitKey(1)
            if key==27:
                print("exiting")
                exit()

    def pointcloud_callback(self, data):

        # get point cloud attributes
        height = data.height
        width = data.width
        point_step = data.point_step
        row_step = data.row_step

        # iterate over face coordinates
        for x,y in self.elipses:
            try:
                # Get transform from "/base_link" to "/map" frame
                trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))

                # get 3-channel representation of the poitn cloud in numpy format
                a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
                a = a.reshape((height,width,3))
                # read center coordinates
                x = int(x)
                y = int(y)
                d = a[x,y,:]

                # Convert face point from robot frame to map frame
                face_point_robot_frame = PointStamped()
                face_point_robot_frame.header = data.header
                face_point_robot_frame.point.x = float(d[0])
                face_point_robot_frame.point.y = float(d[1])
                face_point_robot_frame.point.z = float(d[2])

                #print("b", face_point_robot_frame.point.z)

                face_point_map_stamped = tfg.do_transform_point(face_point_robot_frame, trans)
                #print("c", face_point_map_stamped)

                # Extract transformed face point
                face_point_map = face_point_map_stamped.point
                #print("d", face_point_map)


                # create marker
                marker = Marker()
                marker.header.frame_id = "/map"
                marker.header.stamp = data.header.stamp

                marker.type = Marker.SPHERE
                marker.id = 0

                # Set the scale of the marker
                scale = 0.1
                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = scale

                # Set the color
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                # Set the pose of the marker
                marker.pose.position.x = face_point_map.x
                marker.pose.position.y = face_point_map.y
                marker.pose.position.z = face_point_map.z

                self.elipse_marker_pub.publish(marker)


            except TransformException as e:
                self.get_logger().error(f"Transform exception: {e}")

    def depth_callback(self,data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        depth_image[depth_image==np.inf] = 0

        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)

    def top_camera_callback(self, data):
        
        self.parkings = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        cv2.circle(gray, (159, 263), 69, (160,160,150), -1) # hide robot body black circle

        # Apply Gaussian blur to reduce noise and improve circle detection
        gray_blurred = 255 - cv2.GaussianBlur(gray, (9, 9), 2)

        moo, thresh = cv2.threshold(gray_blurred, 200, 255, cv2.THRESH_BINARY) 
        cv2.imshow("thresh", thresh)
        
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=60, param2=10, minRadius=80, maxRadius=85) # from qr perspective
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, radius) in circles:
                # Draw the circle on the image for visualization
                cv2.circle(cv_image, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(cv_image, (x, y), 2, (0, 0, 255), 3)  # Draw the center of the circle

                # Add the center to the parkings list
                self.parkings.append((x, y))
                print("CENTER:", (x, y))

        # Show the processed image
        cv2.imshow("Top Camera View", cv_image)
        cv2.waitKey(1)

    def top_pointcloud_callback(self, data):

        # get point cloud attributes
        height = data.height
        width = data.width
        point_step = data.point_step
        row_step = data.row_step

        # iterate over face coordinates
        for x,y in self.parkings:
            try:
                if x >= width or x < 0 or y >= height or y < 0:
                    continue

                # get 3-channel representation of the poitn cloud in numpy format
                a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
                a = a.reshape((height,width,3))
                # read center coordinates
                x = int(x)
                y = int(y)
                d = a[y,x,:]

                # create marker
                marker = Marker()
                marker.header.frame_id = "/top_camera_link"
                marker.header.stamp = data.header.stamp

                marker.type = Marker.SPHERE
                marker.id = 0

                # Set the scale of the marker
                scale = 0.2
                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = scale

                # Set the color
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0

                # Set the pose of the marker
                marker.pose.position.x = float(d[0])
                marker.pose.position.y = float(d[1])
                marker.pose.position.z = float(d[2])

                #self.get_logger().info(f"Publishing marker: {marker} on topic: {self.marker_pub.topic_name}")
                self.parking_marker_pub.publish(marker)


            except TransformException as e:
                self.get_logger().error(f"Transform exception: {e}")


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

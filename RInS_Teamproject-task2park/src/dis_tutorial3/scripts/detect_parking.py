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
        self.rings = []
        self.center_array = []
        self.previous_centers = []
        self.ring_color = "unknown"
        self.ring_markerN = 0

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        # Publiser for the visualization markers
        self.marker_pub = self.create_publisher(Marker,"/parkMarker", QoSReliabilityPolicy.BEST_EFFORT)

        self.ring_marker_pub = self.create_publisher(Marker,"/ringMarker", QoSReliabilityPolicy.BEST_EFFORT)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize Pygame mixer for playing audio
        pygame.mixer.init()

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        temp_file = TemporaryFile()
        tts.write_to_fp(temp_file)
        temp_file.seek(0)
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

    def is_new_ring(self, new_center):
        for center in self.previous_centers:
            dist = np.sqrt((new_center.x - center[0])**2 + (new_center.y - center[1])**2 + (new_center.z - center[2])**2)

            #print("dist:")
            #print(dist)
            if dist < 0.35:
                return False
        return True

    def is_circle(self, cnt):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) > 6:
            print("approx", len(approx))
            return True
        return False

    def image_callback(self, data):

        self.elipses = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        # Transform image to HSV(Robust to detect the color when the light is changing)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for red, black, green, and blue in HSV
        color_ranges = {
            "red": [(0, 50, 50), (10, 255, 255)],
            "green": [(50, 50, 50), (70, 255, 255)],
            "blue": [(110, 50, 50), (130, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 30)]
        }

        blue = cv_image[:,:,0]
        green = cv_image[:,:,1]
        red = cv_image[:,:,2]

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        #gray = cv2.GaussianBlur(gray,(3,3),0)

        # Do histogram equalization
        #gray = cv2.equalizeHist(gray)

        edges = cv2.Canny(gray, 50, 150)

        thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        cv2.imshow("Binary Image", thresh)
        cv2.waitKey(1)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Detected contours", gray)
        cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 15 and len(cnt) >= 9 and self.is_circle(cnt):
                print("len(cnt)", len(cnt))
                print("cnt.shape[0]", cnt.shape[0])
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
                if dist >= 3:
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

                # The widths of the ring along the major and minor axis should be roughly the same
                border_major = (le[1][1]-se[1][1])/2
                border_minor = (le[1][0]-se[1][0])/2
                border_diff = np.abs(border_major - border_minor)

                if border_diff>4:
                    continue

                #if e1[0][1] < cv_image.shape[1]/2:
                    #self.get_logger().info(f"UPPER HALF OF IMAGE")
                    #continue

                # Only consider candidates with small size
                l = (le[1][0] + le[1][1]) / 2
                ##print(l)
                if l < 45:
                    candidates.append((e1, e2))
                else:
                    continue
                #candidates.append((e1, e2))

        #print("Processing is done! found", len(candidates), "candidates for rings")

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
            center = (e1[0][1], e1[0][0], size)
            #print("center1", e1[0][1])
            #print("center2", e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]


            ring_inner_region = hsv_image[x_min:x_max, y_min:y_max]

            # Determine the color of the ring
            max_color_pixels = 0

            for color, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(ring_inner_region, np.array(lower), np.array(upper))
                color_pixels = cv2.countNonZero(mask)

                if color_pixels > max_color_pixels:
                    max_color_pixels = color_pixels
                    self.ring_color = color

            #self.get_logger().info(f"Detected ring color: {ring_color}")

            #self.elipses.append(center)
            #print(center, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

            self.center_array.append(center)
            #if e1[0][1] > cv_image.shape[1]/2:
                #self.get_logger().info(f"eclipses")
                #self.elipses.append(center)
                #self.detect_type = 2
            #else:
                #self.get_logger().info(f"rings")
                #self.rings.append(center)
                #self.detect_type = 1


            if len(candidates)>0:
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

        # iterate over point coordinates
        for x,y,size in self.center_array:
            try:
                # Get transform from "/base_link" to "/map" frame
                trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))

                # get 3-channel representation of the poitn cloud in numpy format
                a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
                a = a.reshape((height,width,3))
                # read center coordinates
                x = int(x)
                y = int(y)
                #print("x",x)
                #print("y",y)
                
                half_length = int(size/2)
                x_min = max(0, x - half_length)
                x_max = min(width - 1, x + half_length)
                y_min = max(0, y - half_length)
                y_max = min(height - 1, y + half_length)

                valid_points = []
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        point = a[x, y, :]
                        if not np.any(np.isnan(point)) and not np.any(np.isinf(point)):
                            valid_points.append(point)

                # Calculate the average of valid points
                if valid_points:
                    average_point = np.mean(valid_points, axis=0)
                    #print("Average point within the square:", average_point)

                    # Convert the average point from the sensor frame to the map frame
                    average_point_sensor_frame = PointStamped()
                    average_point_sensor_frame.header = data.header
                    average_point_sensor_frame.point.x = float(average_point[0])
                    average_point_sensor_frame.point.y = float(average_point[1])
                    average_point_sensor_frame.point.z = float(average_point[2])

                    #print("as", average_point_sensor_frame.point.x)
                    #print("as", average_point_sensor_frame.point.y)
                    #print("as", average_point_sensor_frame.point.z)

                    ring_point_map_stamped = tfg.do_transform_point(average_point_sensor_frame, trans)
                    #print("c", face_point_map_stamped)

                    # Extract transformed face point
                    ring_point_map = ring_point_map_stamped.point
                    #print("qq", ring_point_map.x)
                    #print("dq", ring_point_map.y)
                    #print("dqqq", ring_point_map.z)

                    #dont touch
                    point_robot_frame = PointStamped()
                    point_robot_frame.header.frame_id = "oakd_link"
                    point_robot_frame.header.stamp = self.get_clock().now().to_msg()
                    point_robot_frame.header = data.header
                    point_robot_frame.point.x = 0.0
                    point_robot_frame.point.y = 0.0
                    point_robot_frame.point.z = 0.0

                    robot_on_map = tfg.do_transform_point(point_robot_frame, trans)

                    #print("ROBOT ON MAP")
                    #print(robot_on_map)


                    dx = robot_on_map.point.x - ring_point_map.x
                    dy = robot_on_map.point.y - ring_point_map.y
                    dis = math.sqrt(dx**2 + dy**2)
                    direct_x = dx / dis
                    direct_y = dy / dis
                    scadirx = direct_x * 0.2
                    scadiry = direct_y * 0.2

                    #print()
                    #print(scadirx)
                    #print(scadiry)
                    #print()

                    #print(ring_point_map.z)
                    if ring_point_map.z > -0.03 and self.is_new_ring(ring_point_map):
                        marker_ring = Marker()
                        marker_ring.header.frame_id = "/map"
                        marker_ring.header.stamp = data.header.stamp

                        marker_ring.type = Marker.SPHERE
                        #marker_ring.id = len(self.center_array) - 1
                        marker_ring.id = self.ring_markerN

                        self.ring_markerN += 1

                        # Set the scale of the marker
                        scale = 0.2
                        marker_ring.scale.x = scale
                        marker_ring.scale.y = scale
                        marker_ring.scale.z = scale

                        # Set the color
                        if ring_point_map.z >= 0.5:

                            if self.ring_color == "green":
                                marker_ring.color.r = 0.0
                                marker_ring.color.g = 1.0
                                marker_ring.color.b = 0.0
                                marker_ring.color.a = 1.0
                            else:
                                marker_ring.color.r = 1.0
                                marker_ring.color.g = 1.0
                                marker_ring.color.b = 1.0
                                marker_ring.color.a = 1.0
                        else:
                            marker_ring.color.r = 0.0
                            marker_ring.color.g = 0.0
                            marker_ring.color.b = 1.0
                            marker_ring.color.a = 1.0

                        # Set the pose of the marker
                        marker_ring.pose.position.x = ring_point_map.x
                        marker_ring.pose.position.y = ring_point_map.y
                        marker_ring.pose.position.z = ring_point_map.z

                        self.ring_marker_pub.publish(marker_ring)
                        self.previous_centers.append((ring_point_map.x, ring_point_map.y, ring_point_map.z))

                        self.center_array = []

                        if self.ring_color != "unknown":
                            self.speak(f"{self.ring_color}")
                            self.get_logger().info(f"Detected ring color: {self.ring_color}")
                    else:
                        # create marker
                        marker = Marker()
                        marker.header.frame_id = "/map"
                        marker.header.stamp = data.header.stamp

                        marker.type = Marker.SPHERE
                        marker.id = 0

                        # Set the scale of the marker
                        scale = 0.2
                        marker.scale.x = scale
                        marker.scale.y = scale
                        marker.scale.z = scale

                        # Set the color 
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0

                        # Set the pose of the marker
                        marker.pose.position.x = ring_point_map.x - scadirx
                        marker.pose.position.y = ring_point_map.y - scadiry
                        marker.pose.position.z = ring_point_map.z

                        #self.get_logger().info(f"Publishing marker: {marker} on topic: {self.marker_pub.topic_name}")
                        self.marker_pub.publish(marker)

                        self.center_array = []
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


def createPS(self, frame_id, point):
    point_robot_frame = PointStamped()
    point_robot_frame.header.frame_id = frame_id
    point_robot_frame.header.stamp = self.get_clock().now().to_msg()
    point_robot_frame.header = data.header
    point_robot_frame.point.x = point[0]
    point_robot_frame.point.y = point[1]
    point_robot_frame.point.z = point[2]
    return point_robot_frame

def is_valid_value(value):
    return not (np.isnan(value) or np.isinf(value))



def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

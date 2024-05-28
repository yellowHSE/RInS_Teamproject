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
        self.parkings = []

        # New subscription for the top camera image
        self.top_camera_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.top_camera_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.top_pointcloud_callback, qos_profile_sensor_data)

        self.parking_marker_pub = self.create_publisher(Marker,"/parkMarker", QoSReliabilityPolicy.BEST_EFFORT)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        #cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)


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
                marker.color.g = 1.0
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

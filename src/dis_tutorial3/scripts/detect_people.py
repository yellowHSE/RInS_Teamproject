#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math

from ultralytics import YOLO
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import PointStamped

from gtts import gTTS
import pygame
from tempfile import TemporaryFile
import time

#from robot_commander import robot_commander

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult


#rc = robot_commander()

#def distance(p1, p2):
	#return math.sqrt((float(p1[0])-float(p2[0]))**2 + (float(p1[1]) - float(p2[1]))**2)

#def point_in_area(point, array, threshold):
	#for p in array:
		#if distance(point, p) <= threshold:
			#return True

	#return False

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		marker_topic = "/people_marker"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.saved_markers = []

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		# TF2 Buffer and Listener for transformations
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# Previously detected face points
		self.prev_face_points = []

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			# self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue


				bbox = bbox[0]

				'''
				# Check for brown boundary in the bounding box
				if self.has_brown_boundary(cv_image, bbox):
					self.get_logger().info("Not People")
					continue
				'''

				self.get_logger().info(f"Person has been detected!")

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

				self.faces.append((cx,cy))
				#print(self.faces)

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()

		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step

		# iterate over face coordinates
		for x,y in self.faces:
			try:
				# Get transform from "/base_link" to "/map" frame
				trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))

				# get 3-channel representation of the poitn cloud in numpy format
				a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
				a = a.reshape((height,width,3))
				# read center coordinates
				d = a[y,x,:]

				# Convert face point from robot frame to map frame
				face_point_robot_frame = PointStamped()
				face_point_robot_frame.header = data.header
				face_point_robot_frame.point.x = float(d[0])
				face_point_robot_frame.point.y = float(d[1])
				face_point_robot_frame.point.z = float(d[2])

				face_point_map_stamped = tfg.do_transform_point(face_point_robot_frame, trans)

				# Extract transformed face point
				face_point_map = face_point_map_stamped.point


				# If current face point is same as any of the previous face points, skip publishing marker
				#if any(np.allclose([face_point_map.x, face_point_map.y, face_point_map.z], [prev_face_point.x, prev_face_point.y, prev_face_point.z]) for prev_face_point in self.prev_face_points):
 				#	continue

				if len(self.prev_face_points) > 0 and self.is_close(face_point_map, threshold=0.5):
					self.get_logger().info("I WAS HERE")
					continue


 				# Add current face point to the list of previous face points
				self.prev_face_points.append(face_point_map)
				#greeting()

				for point in self.prev_face_points:
						self.get_logger().info(f"({self.prev_face_points})")

				#if len(self.prev_face_points) == 3:
				#	self.get_logger().info("ROBOT STOP")
				#	self.get_logger().info("Prev face points:")
				#	for point in self.prev_face_points:
				#		self.get_logger().info(f"({self.prev_face_points})")
				#		rclpy.shutdown()
					#	return

				# create marker
				marker = Marker()
				marker.header.frame_id = "/map"
				marker.header.stamp = data.header.stamp

				marker.type = Marker.SPHERE
				marker.id = len(self.prev_face_points) - 1  # Assign unique ID for each marker

				# Set the scale of the marker
				scale = 0.2
				marker.scale.x = scale
				marker.scale.y = scale
				marker.scale.z = scale

				# Set the color
				marker.color.r = 1.0
				marker.color.g = 1.0
				marker.color.b = 0.0
				marker.color.a = 1.0

				# Set the pose of the marker
				marker.pose.position.x = face_point_map.x
				marker.pose.position.y = face_point_map.y
				marker.pose.position.z = face_point_map.z

				self.get_logger().info(f"Publishing marker: {marker} on topic: {self.marker_pub.topic_name}")
				self.marker_pub.publish(marker)


			except TransformException as e:
				self.get_logger().error(f"Transform exception: {e}")

	def is_close(self, current_point, threshold):

		for prev_point in self.prev_face_points:
			distance = math.sqrt((current_point.x - prev_point.x) ** 2 + (current_point.y - prev_point.y) ** 2)
			if distance < threshold:
				return True

		return False

	def has_brown_boundary(self, image, bbox):
		# Define brown color range in HSV
		lower_brown = np.array([10, 100, 20])
		upper_brown = np.array([20, 255, 200])

		# Extract bounding box region
		x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
		bbox_region = image[y1:y2, x1:x2]

		brown_threshold = 6

		x1_expanded = max(0, x1 - brown_threshold)
		y1_expanded = max(0, y1 - brown_threshold)
		x2_expanded = min(image.shape[1], x2 + brown_threshold)
		y2_expanded = min(image.shape[0], y2 + brown_threshold)

		# Create a mask for the expanded region
		mask_outer = np.zeros(image.shape[:2], dtype=np.uint8)
		mask_outer[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 255

		# Create a mask for the inner bounding box region
		mask_inner = np.zeros(image.shape[:2], dtype=np.uint8)
		mask_inner[y1:y2, x1:x2] = 255

		# Subtract the inner mask from the outer mask to get the border region
		mask_border = cv2.subtract(mask_outer, mask_inner)

		# Apply the mask to the original image to get the border region
		border_region = cv2.bitwise_and(image, image, mask=mask_border)

		# Display the border region
		cv2.imshow("Border Region", border_region)
		cv2.waitKey(1)

		# Convert the border region to HSV color space
		hsv_border = cv2.cvtColor(border_region, cv2.COLOR_BGR2HSV)

		# Create a mask for brown color in the border region
		mask_brown = cv2.inRange(hsv_border, lower_brown, upper_brown)

		# Check if there is any brown color in the mask
		if cv2.countNonZero(mask_brown) > 0:
			return True
		return False

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()

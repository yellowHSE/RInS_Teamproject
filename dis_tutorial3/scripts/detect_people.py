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

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue



				self.get_logger().info(f"Person has been detected!")

				bbox = bbox[0]

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
				if any(np.allclose([face_point_map.x, face_point_map.y, face_point_map.z], [prev_face_point.x, prev_face_point.y, prev_face_point.z]) for prev_face_point in self.prev_face_points):
 					continue

 				# Add current face point to the list of previous face points
				self.prev_face_points.append(face_point_map)

				if len(self.prev_face_points) == 3:
					self.get_logger().info("ROBOT STOP")
					self.get_logger().info("Prev face points:")
					for point in self.prev_face_points:
						self.get_logger().info(f"({self.prev_face_points})")
						rclpy.shutdown()
						return

				# create marker
				marker = Marker()
				marker.header.frame_id = "/map"
				marker.header.stamp = data.header.stamp

				marker.type = Marker.SPHERE
				marker.id = len(self.prev_face_points) - 1  # Assign unique ID for each marker

				# Set the scale of the marker
				scale = 0.1
				marker.scale.x = scale
				marker.scale.y = scale
				marker.scale.z = scale

				# Set the color
				marker.color.r = 1.0
				marker.color.g = 1.0
				marker.color.b = 1.0
				marker.color.a = 1.0

				# Set the pose of the marker
				marker.pose.position.x = face_point_map.x
				marker.pose.position.y = face_point_map.y
				marker.pose.position.z = face_point_map.z

				self.marker_pub.publish(marker)
			except TransformException as e:
				self.get_logger().error(f"Transform exception: {e}")

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()

#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PointStamped

from gtts import gTTS
import pygame
from tempfile import TemporaryFile


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)

        self.pose_frame_id = 'map'
        self.face_coordinate_msg = None
        self.detected_face_num = 0

        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.face_coordinate_received = False

        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)

        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)

        self.face_coord_sub = self.create_subscription(Marker, '/people_marker', self.face_coord_callback, 10)

        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)

        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")

    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    def face_coord_callback(self, msg):

        self.face_coordinate_msg = msg
        # Callback function to handel face coordinate messages
        # self.get_logger().info(f"Received face marker: {msg}")

        face_pose = msg.pose
        face_position = face_pose.position

        x = face_position.x
        y = face_position.y
        z = face_position.z

        self.get_logger().info(f"Face coordinate: ({x}, {y}, {z})")
        self.face_coordinate_received = True

    def approachToFace(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to face: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Face to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        while not self.isTaskComplete():
            time.sleep(1)


        time.sleep(2)
        self.greeting()
        return True

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        #self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
        #          str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def greeting(self):
        text_to_speak = "Hello."
        tts = gTTS(text=text_to_speak, lang='en')

        temp_file = TemporaryFile()
        tts.write_to_fp(temp_file)
        temp_file.seek(0)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        temp_file.close()

    def move_robot_to_face(self, marker):
        # Create a PoseStamped message for the face coordinate
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = marker.pose.position.x
        pose.pose.position.y = marker.pose.position.y
        pose.pose.position.z = marker.pose.position.z
        pose.pose.orientation.w = 1.0  # Assuming no rotation is needed

        self.info(f'Approach to Face - move robot{pose}')

        if self.approachToFace(pose):
            self.info("Robot reached the face coordinate")
            #self.greeting()
        else:
            self.info("Fail to reach the face coordinate")

        while not self.isTaskComplete():
            self.info("Waiting to complete approach method")
            time.sleep(1)

        # Clear face coordinate message
        self.face_coordinate_msg = None
        self.face_coordinate_received = False

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return

    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return

    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return

def main(args=None):

    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()

    # Finally send it a goal to reach

    goal_positions = [
        #{'x': -1.0, 'y': -0.5, 'yaw': 0.57},
        {'x': 0.0, 'y': -1.8, 'yaw': 0.57},
        {'x': 1.2, 'y': -1.8, 'yaw': 0.57},
        {'x': 2.6, 'y': -1.3, 'yaw': 0.57},
        {'x': 2.3, 'y': -1.3, 'yaw': 0.57},
        {'x': 1.0, 'y': -0.5, 'yaw': 1.0},
        {'x': 2.5, 'y': -0.5, 'yaw': 0.57},
        {'x': 2.5, 'y': 2.2, 'yaw': 0.57},
        {'x': 0.0, 'y': 3.2, 'yaw': 0.57},
        {'x': -1.0, 'y': 3.2, 'yaw': 0.57},
        {'x': -1.5, 'y': 4.1, 'yaw': 0.57},
        {'x': -1.0, 'y': 1.0, 'yaw': 0.57}
    ]


    for goal in goal_positions:
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = rc.get_clock().now().to_msg()

        goal_pose.pose.position.x = goal['x']
        goal_pose.pose.position.y = goal['y']
        goal_pose.pose.orientation = rc.YawToQuaternion(goal['yaw'])

        rc.goToPose(goal_pose)

        while not rc.isTaskComplete():
            # Check if face coordinate message is received
            if rc.face_coordinate_received:
                rc.info("face detected")

                rc.detected_face_num = rc.detected_face_num + 1

                # rc.cancelTask()
                # Move robot to face coordinate
                rc.move_robot_to_face(rc.face_coordinate_msg)

                # Wait for 5 seconds
                time.sleep(5)

                #rc.greeting()
                # Clear face coordinate message
                rc.face_coordinate_msg = None
                rc.face_coordinate_received = False

                if(rc.detected_face_num > 2):
                    break

            else:
                #rc.info("Waiting")
                time.sleep(1)
        if(rc.detected_face_num > 2):
            break
        #rc.info("Goal reached")


    rc.destroyNode()

    # And a simple example
if __name__=="__main__":
    main()

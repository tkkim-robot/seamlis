#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np
import math
from shapely.geometry import LineString
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleLocalPosition
from nav_msgs.msg import Odometry

from seamlis.exploration import ExplorationManager
from seamlis.utils import env, plotting

from nvblox_msgs.msg import DistanceMapSlice

import cv2
import os

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class ExplorationROSNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        # Parameters
        self.dt = 0.05
        self.controller_type = {'pos': 'mpc_cbf', 'att': 'gatekeeper'}

        # Robot specs and init pose
        robot_spec = {
            'model': 'DoubleIntegrator2D',
            'w_max': 0.3,
            'a_max': 0.5,
            'v_max': 0.5,
            'fov_angle': 70.0,
            'cam_range': 3.0,
            'sensor': 'rgbd',
            'robot_id': 0,
        }
        self.robot_specs = [robot_spec]

        x_init = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # match your waypoint[3]
        self.exploration_manager = ExplorationManager(
            [x_init], self.robot_specs, self.controller_type,
            exploration_algorithm='Frontier',
            dt=self.dt, show_animation=True, save_animation=False
        )
        
        # setup obstacle position #FIXME:
        self.exploration_manager.controller_list[0].obs = np.array([[-1.75, -0.5, 0.75]])

        # ROS interfaces
        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            '/px4_3/fmu/out/vehicle_local_position',
            self.odom_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.map_subscription = self.create_subscription(
            DistanceMapSlice,
            '/nvblox_node/static_map_slice',
            self.map_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.publisher = self.create_publisher(Float32MultiArray, 'ctrl_vel', 10)

        # Exploration state
        self.exploration_initialized = False
        self.robot_goal = None
        self.infeasible_flag = False

        # Frontiers
        self.frontier = None
        self.received_first_map = False

        # Plotting
        self.odom_x_list = []
        self.odom_y_list = []
        self.vicon_pose = None

        self.env_handler = env.Env()

        # Initial frontier and goal assignment
        # self.initialize_exploration()

        # TODO: timer
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz

    def _extract_frontier(self, msg):
        # Convert flat ESDF data into 2D grid
        map_slice = np.array(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        width, height = msg.width, msg.height
        resolution = msg.resolution
        unknown_val = msg.unknown_value
        epsilon = 0.2  # Threshold for free vs obstacle

        # Map bounds in metric coordinates
        min_x, max_x = self.env_handler.x_range
        min_y, max_y = self.env_handler.y_range

        #print(f"min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}")

        # Masks
        obstacle_mask = (map_slice < epsilon) & (map_slice != unknown_val)
        known_free_mask = (map_slice != unknown_val) & (map_slice >= epsilon)
        unknown_mask = (map_slice == unknown_val)

        # Padding for edge safety
        padded_unknown = np.pad(unknown_mask, 1, mode='constant')
        padded_obstacle = np.pad(obstacle_mask, 1, mode='constant')

        # Detect frontier cells
        frontier_mask = np.zeros_like(known_free_mask, dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_unknown = padded_unknown[1+dy:1+dy+height, 1+dx:1+dx+width]
            neighbor_obstacle = padded_obstacle[1+dy:1+dy+height, 1+dx:1+dx+width]
            frontier_mask |= known_free_mask & neighbor_unknown & (~neighbor_obstacle)

        # Remove isolated frontier cells (erosion-like operation)
        kernel = np.ones((3,3), np.uint8)
        frontier_count = cv2.filter2D((frontier_mask).astype(np.uint8), -1, kernel)
        frontier_mask &= (frontier_count >= 3)  # Keep only cells with at least 3 frontier neighbors

        # Get frontier pixel coordinates
        frontier_indices = np.argwhere(frontier_mask)
        origin_x, origin_y = msg.origin.x, msg.origin.y

        #print(f"origin_x: {origin_x}, origin_y: {origin_y}")

        # Convert pixel (row, col) to metric (x, y)
        frontier_points = np.array([
            [origin_x + col * resolution, -(origin_y + row * resolution)]
            for row, col in frontier_indices
        ], dtype=np.float32)


        # Handle empty or invalid frontier points
        if frontier_points is None or len(frontier_points) == 0:
            frontier_points = np.empty((0, 2), dtype=np.float32)
            frontier_indices = np.empty((0, 2), dtype=np.int32)
        elif len(frontier_points.shape) == 1:
            # If single point, reshape to 2D array
            frontier_points = frontier_points.reshape(1, -1)
            frontier_indices = frontier_indices.reshape(1, -1)

        # Filter frontiers outside map bounds
        bounds_mask = (frontier_points[:, 0] >= min_x) & (frontier_points[:, 0] <= max_x) & \
                     (frontier_points[:, 1] >= min_y) & (frontier_points[:, 1] <= max_y)
        frontier_points = frontier_points[bounds_mask]
        frontier_indices = frontier_indices[bounds_mask]

        # Filter out frontiers that are too close to the robot
        if self.vicon_pose is not None:
            robot_pos = np.array([self.vicon_pose.x, self.vicon_pose.y])
            #print(f"robot_pos: {robot_pos}")
            distances = np.linalg.norm(frontier_points - robot_pos, axis=1)
            mask = distances >= self.exploration_manager.controller_list[0].reached_threshold
            frontier_points = frontier_points[mask]
            frontier_indices = frontier_indices[mask]
            # Update frontier mask based on filtered points
            frontier_mask[:] = False  # Clear existing mask
            frontier_mask[frontier_indices[:, 0], frontier_indices[:, 1]] = True

        # Wrap into shapely LineString so you can use .coords
        self.frontier = LineString(frontier_points)

        # Save visualization image
        frontier_vis = np.zeros((height, width, 3), dtype=np.uint8)
        frontier_vis[known_free_mask] = [0, 255, 0]
        frontier_vis[obstacle_mask] = [0, 0, 255]
        frontier_vis[frontier_mask] = [255, 255, 255]

        # draw 5*5 square white at the origin
        frontier_vis[0:5, 0:5] = [255, 255, 255]

        cv2.imwrite("/workspaces/colcon_ws/frontiers.png", frontier_vis)


    def map_callback(self, msg):
        self.received_first_map = True

        if not msg.data:
            self.get_logger().warn("Received empty DistanceMapSlice.")
            return

        min_val = min(msg.data)
        #self.get_logger().info(f"Minimum value in DistanceMapSlice: {min_val}")

        self._extract_frontier(msg)

        if not self.exploration_initialized:
            self.get_logger().info("Initializing exploration.")
            self.initialize_exploration()

    def initialize_exploration(self):
        # TODO: check whether at least one callback has been called for map_callback
        # TODO: and, in map_callback, you receive nvblox map, and
        # TODO: whenever the callback being called, you do post-processing on that map
        # TODO: to extract frontier. I believe it will be just one line
        # TODO: self.nvblox_frontiers = np.where(map_value is smaller than something)
        # self.exploration_manager.frontiers = self.exploration_manager.get_frontiers()
        if not self.received_first_map:
            self.get_logger().warn("Waiting for first map slice...")
            return
        self.exploration_manager.frontiers = self.frontier
        self.exploration_manager.update_all_goals()
        # Main loop that updates goals
        self.exploration_initialized = True
        # while not self.exploration_complete():
        #     robots_reached_goals = self.move_robots()
            
        #     if any(robots_reached_goals):
        #         self.frontiers = self.get_frontiers()  # Update frontiers
        #         self.update_goals_for_completed(robots_reached_goals)


    # TODO:
    def timer_callback(self):

        if not self.exploration_initialized:
            self.get_logger().warn("Exploration not initialized.")
            return
        
        robots_reached_goals = self.exploration_manager.move_robots()
        
        if any(robots_reached_goals):
            #print(f"robots_reached_goals: {robots_reached_goals}")
            self.exploration_manager.frontiers = self.frontier  # Update frontiers
            self.exploration_manager.update_goals_for_completed(robots_reached_goals)

    def odom_callback(self, msg):

        controller = self.exploration_manager.controller_list[0]
        goal = controller.goal

        # PX4 local position update
        pose = Odometry().pose.pose.position
        pose.x = msg.x
        pose.y = msg.y
        pose.z = msg.z

        orientation = [0, 0, angle_normalize(msg.heading + np.pi / 2)]

        velocity = Odometry().twist.twist.linear
        velocity.x = msg.vx
        velocity.y = msg.vy

        controller.set_robot_state(pose, orientation, velocity)
        self.vicon_pose = pose

        if not self.exploration_initialized:
            return
        #print(f"goal: {goal}")

        if goal is None and controller.state_machine != 'stop':
            self.get_logger().info("Exploration complete....")

            msg = Float32MultiArray()
            msg.data = [self.vicon_pose.x, self.vicon_pose.y, 0,0, 0.0, 0,0, 0.0, 0,0, 0.0, pose.z]
            self.publisher.publish(msg)
            return
        
        u = controller.get_control_input()
        yaw_rate = controller.get_att_input()
        x_next, yaw_input = controller.get_full_state()

        # print("u: ", u)
        # print("yaw_rate: ", yaw_rate)
        # print("x_next: ", x_next)
        # print("yaw_input: ", yaw_input)


        full_state_u = np.concatenate((x_next, u, [[yaw_input]], yaw_rate, [[pose.z]]), axis=0)
        msg = Float32MultiArray()
        msg.data = [float(val) for val in full_state_u.flatten()]
        self.publisher.publish(msg)

        #self.get_logger().info(f'Publishing: {msg.data}')

        # if self.infeasible_flag:
        #     self.infeasible_flag = True
        #     self.get_logger().warn("Infeasible state detected!")
        #     u = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #     msg.data = u.tolist()
        #     self.publisher.publish(msg)

        # if ret == -1:  # Waypoint reached
        #     self.get_logger().info("Waypoint reached. Recomputing frontiers.")
        #     u = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #     msg.data = u.tolist()
        #     self.publisher.publish(msg)
        #     self.exploration_manager.frontiers = self.exploration_manager.get_frontiers()
        #     self.exploration_manager.update_goals_for_completed([True])
        #     if self.exploration_manager.show_animation:
        #         self.exploration_manager.update_visualization()

        # Logging / Visualization
        self.odom_x_list.append(pose.x)
        self.odom_y_list.append(pose.y)
        # Plot later
        
    def shutdown(self):
        self.get_logger().info("Completing exploration.")

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationROSNode()
    rclpy.spin(node)
    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from tracking import LocalTrackingController
from algorithms.co_scan import CoScanPlanner
from utils import plotting
from utils import env

"""
Created on June 22nd, 2024
@author: Taekyung Kim

@description: 

@functions to be implemented: 
    An Exploration class
        1. correctly assign frontier as goal
        2. assign global goal at every iterations? or just assign it after it reaches the goal?

@required-scripts: tracking.py, utils/plotting.py, utils/env.py, algorithms/co_scan.py
"""

class ExplorationManager:
    def __init__(self, X0s, type='DynamicUnicycle2D', exploration_algorithm='CoScan',
                  num_robot=1, dt=0.05,
                  show_animation=False, save_animation=False):
        self.type = type
        self.num_robot = num_robot
        self.dt = dt

        self.plot_handler = plotting.Plotting()
        self.ax, self.fig = self.plot_handler.plot_grid("Local Tracking Controller")
        self.env_handler = env.Env()

        self.show_animation = show_animation
        self.save_animation = save_animation

        self.controller_list = []
        for i in range(num_robot):
            X0 = X0s[i]
            tracking_controller = LocalTrackingController(X0, type=type, 
                                         robot_id=i,
                                         dt=dt,
                                         show_animation=show_animation,
                                         save_animation=save_animation,
                                         ax=self.ax, fig=self.fig,
                                         env=self.env_handler)
            self.controller_list.append(tracking_controller)
        self.merged_global_map = Polygon() 
        self.frontiers = LineString()

        self.frontiers_scatter = self.ax.scatter([],[],s=10,facecolors='orange',edgecolors='orange')
        self.global_goals_scatter = self.ax.scatter([],[],s=10,facecolors='blue',edgecolors='blue')

        # Set up the environment
        self.set_env_obstacles(self.env_handler)
        self.set_env_workspace(self.env_handler)

        if exploration_algorithm == 'CoScan':
            self.exploration_algorithm = CoScanPlanner()
        else:
            raise ValueError(f"Exploration algorithm {exploration_algorithm} is not implemented")


    def set_env_workspace(self, env_handler):
        # Add workspace boundary
        self.env_workspace = Polygon([(0, 0), (0, env_handler.height), 
                             (env_handler.width, env_handler.height), 
                             (env_handler.width, 0)])
        return self.env_workspace
        
    def set_env_obstacles(self, env_handler):
        # Get obstacle information
        obs_rectangle = env_handler.obs_rectangle
        obs_circle = env_handler.obs_circle

        # Create obstacle geometries
        obstacle_geometries = []
        
        for rect in obs_rectangle:
            x, y, w, h = rect
            obstacle_geometries.append(Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]))
        for circle in obs_circle:
            x, y, r = circle
            obstacle_geometries.append(Point(x, y).buffer(r))

        # Combine all obstacles
        self.env_obstacles = unary_union(obstacle_geometries)
        return self.env_obstacles

    def set_waypoints(self, robot_ids, waypoints):
        # Convert robot_ids and waypoints to lists if a single instance is provided
        if not isinstance(robot_ids, list):
            robot_ids = [robot_ids]
            waypoints = [waypoints]
        
        # Ensure the number of robot_ids matches the number of waypoints
        if len(robot_ids) != len(waypoints):
            raise ValueError("The number of robot_ids must match the number of waypoints")

        # Set waypoints for each specified robot
        for robot_id, waypoint in zip(robot_ids, waypoints):
            if 0 <= robot_id < self.num_robot:
                self.controller_list[robot_id].set_waypoints(waypoint)
            else:
                raise ValueError(f"Invalid robot_id: {robot_id}. Must be between 0 and {self.num_robot - 1}")
            
    def merge_sensing_footprints(self):
        '''
        Merge all sensing footprints from all robots into a global, centralized map
        '''
        all_footprints = []

        # Collect sensing footprints from all robots
        for controller in self.controller_list:
            robot_footprints = controller.robot.sensing_footprints
            all_footprints.append(robot_footprints)

        # Merge all footprints using shapely's unary_union
        return unary_union(all_footprints)

    def get_frontiers(self):
        '''
        Merge global map and extract frontiers
        '''
        self.merged_global_map = self.merge_sensing_footprints()
        if isinstance(self.merged_global_map, Polygon):
            boundaries = [self.merged_global_map.exterior]
        elif isinstance(self.merged_global_map, MultiPolygon):
            boundaries = [poly.exterior for poly in self.merged_global_map.geoms]
        else:
            raise ValueError(f"Unexpected type for merged_global_map: {type(self.merged_global_map)}")
        
        frontiers = []
        for boundary in boundaries:
            coords = np.array(boundary.coords)
            frontiers.extend(self.interpolate_and_filter_frontier(coords))

        return LineString(frontiers)  # Return frontiers as a LineString

    def interpolate_and_filter_frontier(self, coords, step=0.3):
        '''
        @description: Vectorized interpolation between points and checking if interpolated points are valid frontiers
        @note: shapely object's exterior only contains the boundary points, so we need to interpolate between them
        '''
        # Calculate distances between consecutive points
        diff = np.diff(coords, axis=0)
        distances = np.sqrt((diff**2).sum(axis=1))
        
        # Calculate number of steps for each segment
        steps = np.maximum(np.ceil(distances / step).astype(int), 1)
        
        # Generate interpolation parameters
        cum_steps = np.cumsum(steps)
        total_steps = cum_steps[-1]
        
        indices = np.arange(total_steps)
        segment_indices = np.searchsorted(cum_steps, indices, side='right')
        
        # Correct calculation of segment_steps
        segment_starts = np.concatenate(([0], cum_steps[:-1]))
        segment_steps = indices - segment_starts[segment_indices]
        
        # Ensure alphas are calculated correctly
        steps_broadcast = steps[segment_indices]
        alphas = segment_steps / steps_broadcast

        # Interpolate points
        start_points = coords[segment_indices]
        end_points = coords[segment_indices + 1]
        
        interpolated_points = start_points + alphas[:, np.newaxis] * (end_points - start_points)
        
        # Check validity of interpolated points
        valid_mask = np.array([self.env_workspace.contains(Point(p)) and not self.env_obstacles.contains(Point(p)) 
                            for p in interpolated_points])
        
        return interpolated_points[valid_mask]

    def control_step(self):
        '''
        Simulate one control step and compute frontiers for all robots
        '''
        for controller in self.controller_list:
            controller.control_step()
        self.frontiers = self.get_frontiers()
        if self.show_animation:
            self.frontiers_scatter.set_offsets(self.frontiers.coords)
            self.controller_list[0].draw_plot()

    def explore(self):
        '''
        Main exploration loop
        '''
        self.frontiers = self.get_frontiers() # initially get frontiers

        while not self.exploration_complete():
            self.global_goals = self.update_global_goals()
            print(self.global_goals)
            if self.show_animation:
                self.global_goals_scatter.set_offsets(self.global_goals)
            for i, controller in enumerate(self.controller_list):
                controller.set_waypoints([self.global_goals[i]]) # assign a single waypoint as goal
            self.control_step()
            #self.replan_if_necessary()

    def update_global_goals(self):
        np_obstacle_map = self.get_obstacle_map()
        np_frontier_map = self.get_frontier_map()
        agent_positions = self.get_robot_positions()
        global_goals =self.exploration_algorithm.get_long_term_goals(np_obstacle_map, np_frontier_map, agent_positions)
        return self.env_handler.grid_to_f(global_goals)

    def get_obstacle_map(self):
        obstacle_map = np.zeros(self.env_handler.get_map_shape(), dtype=np.int8)
        # FIXME:
        # for geom in self.env_obstacles.geoms:
        #     x, y = geom.exterior.coords.xy
        #     rr, cc = polygon(y, x)
        #     obstacle_map[rr, cc] = 1
        return obstacle_map

    def get_frontier_map(self):
        frontier_map = np.zeros(self.env_handler.get_map_shape(), dtype=np.int8)
        # TODO: this for loop can be vectorized
        for x, y in self.frontiers.coords:
            x, y = self.env_handler.f_to_grid([x, y])
            frontier_map[int(y), int(x)] = 1
        return frontier_map

    def get_robot_positions(self):
        positions = np.array([controller.robot.get_position() for controller in self.controller_list])
        return self.env_handler.f_to_grid(positions)

    def replan_if_necessary(self):
        for i, controller in enumerate(self.controller_list):
            current_goal = controller.get_current_goal()
            if self.goal_unreachable(current_goal):
                new_goal = self.exploration_algorithm.replan(
                    self.get_obstacle_map(),
                    self.get_frontier_map(),
                    current_goal,
                    self.get_planning_window(controller)
                )
                controller.set_goal(new_goal)

    def goal_unreachable(self, goal):
        obstacle_map = self.get_obstacle_map()
        return obstacle_map[int(goal[1]), int(goal[0])] == 1

    def get_planning_window(self, controller):
        pos = controller.robot.get_position()
        window_size = 20  # Adjust as needed
        x1 = max(0, int(pos[0]) - window_size)
        x2 = min(self.env_handler.width, int(pos[0]) + window_size)
        y1 = max(0, int(pos[1]) - window_size)
        y2 = min(self.env_handler.height, int(pos[1]) + window_size)
        return (x1, x2, y1, y2)

    def exploration_complete(self):
        return len(self.frontiers.coords) == 0
        

def main():
    dt = 0.5

    # temporal
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [15, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = waypoints[0]
    x_init2 = waypoints[-1]
    type = 'DynamicUnicycle2D'
    exploration = ExplorationManager([x_init,x_init2], type=type, num_robot=2, dt=dt,
                                    show_animation=True,
                                    save_animation=False)
    exploration.explore()
    # to check the set_waypoints function
        # exploration.set_waypoints(0, waypoints)
        # exploration.set_waypoints(1, waypoints[::-1])
        # for i in range(1000):
        #     exploration.control_step()
    

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math
    
    main()

    
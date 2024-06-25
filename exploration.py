import numpy as np
import matplotlib.pyplot as plt

from tracking import LocalTrackingController
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from utils import plotting
from utils import env

"""
Created on June 22nd, 2024
@author: Taekyung Kim

@description: 

@functions to be implemented: 
    An Exploration class
        1. can handle multiple agents by setting up multiple controllers
        2. manage the global map
        3. extract the global frontier

@required-scripts: tracking.py
"""

class ExplorationManager:
    def __init__(self, X0s, type='DynamicUnicycle2D', num_robot=1, dt=0.05,
                  show_animation=False, save_animation=False):
        self.type = type
        self.num_robot = num_robot
        self.dt = dt

        plot_handler = plotting.Plotting()
        self.ax, self.fig = plot_handler.plot_grid("Local Tracking Controller")
        env_handler = env.Env()

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
                                         env=env_handler)
            self.controller_list.append(tracking_controller)
        self.merged_global_map = Polygon() 
        self.frontiers = LineString()

        self.frontiers_scatter = self.ax.scatter([],[],s=10,facecolors='orange',edgecolors='orange')

        # Set up the environment
        self.set_env_obstacles(env_handler)
        self.set_env_workspace(env_handler)


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
        self.merged_global_map = self.merge_sensing_footprints()
        if isinstance(self.merged_global_map, Polygon):
            boundaries = [self.merged_global_map.exterior]
        elif isinstance(self.merged_global_map, MultiPolygon):
            boundaries = [poly.exterior for poly in self.merged_global_map.geoms]
        else:
            raise ValueError(f"Unexpected type for merged_global_map: {type(self.merged_global_map)}")

        # Extract frontiers (boundary points not inside obstacles and within workspace)
        frontiers = []
        for boundary in boundaries:
            for point in boundary.coords:
                point_geom = Point(point)
                if self.env_workspace.contains(point_geom) and not self.env_obstacles.contains(point_geom):
                    frontiers.append(point)

        return LineString(frontiers)  # Return frontiers as a LineString
    
    def control_step(self):
        for controller in self.controller_list:
            controller.control_step()
        self.frontiers = self.get_frontiers()
        if self.show_animation:
            self.frontiers_scatter.set_offsets(self.frontiers.coords)

        self.controller_list[0].draw_plot()
        

def main():
    dt = 0.05

    # temporal
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [10, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = waypoints[0]
    x_init2 = waypoints[-1]
    type = 'DynamicUnicycle2D'
    exploration = ExplorationManager([x_init,x_init2], type=type, num_robot=2, dt=dt,
                                    show_animation=True,
                                    save_animation=False)
    exploration.set_waypoints(0, waypoints)
    exploration.set_waypoints(1, waypoints[::-1])
    for i in range(1000):
        exploration.control_step()
    

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math
    
    main()

    
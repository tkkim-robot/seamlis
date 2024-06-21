import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvxpy as cp
import os
import glob
import subprocess

"""
Created on June 20th, 2024
@author: Taekyung Kim

@description: 

@required-scripts: robots/robot.py
"""

class QPError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''
    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)


class LocalTrackingController:
    def __init__(self, X0, type='DynamicUnicycle2D', dt=0.05,
                  show_animation=False, save_animation=False, plotting=None, env=None):
        self.type = type
        self.dt = dt

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 1.0

        if self.type == 'Unicycle2D':
            self.alpha = 1.0
            self.v_max = 1.0
            self.w_max = 0.5
        elif self.type == 'DynamicUnicycle2D':
            self.alpha1 = 1.5
            self.alpha2 = 1.5
            # v_max is set to 1.0 inside the robot class
            self.a_max = 0.5
            self.w_max = 0.5
            X0 = np.array([X0[0], X0[1], X0[2], 0.0]).reshape(-1, 1)

        self.show_animation = show_animation
        self.save_animation = save_animation
        if self.save_animation:
            self.current_directory_path = os.getcwd() 
            if not os.path.exists(self.current_directory_path + "/output/animations"):
                os.makedirs(self.current_directory_path + "/output/animations")
            self.save_per_frame = 2
            self.ani_idx = 0

        self.plotting = plotting
        self.obs = np.array(env.obs_circle)
        self.unknown_obs = None

        if show_animation:
            # Initialize plotting
            if self.plotting is None:
                self.fig = plt.figure()
                self.ax = plt.axes()
            else:
                # plot the obstacles
                self.ax, self.fig = self.plotting.plot_grid("Path Following")
            plt.ion()
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_aspect(1)

            # Visualize goal and obstacles
            self.ax.scatter(waypoints[:, 0], waypoints[:, 1], c='g', s=10)
        else:
            self.ax = plt.axes() # dummy placeholder

        # Setup control problem
        self.setup_robot(X0)
        self.setup_control_problem()

    def setup_robot(self, X0):
        from robots.robot import BaseRobot
        self.robot = BaseRobot(X0.reshape(-1, 1), self.dt, self.ax, self.type)

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((1, 2), value=np.zeros((1, 2)))
        self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

        if self.type == 'Unicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.v_max,
                           cp.abs(self.u[1]) <= self.w_max]
        elif self.type == 'DynamicUnicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <= self.a_max,
                            cp.abs(self.u[1]) <= self.w_max]
        self.cbf_controller = cp.Problem(objective, constraints)

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.current_goal_index = 0

    def goal_reached(self, current_position, goal_position):
        return np.linalg.norm(current_position[:2] - goal_position[:2]) < self.reached_threshold

    def set_unknown_obs(self, unknown_obs):
        # set initially
        self.unknown_obs = unknown_obs
        for (ox, oy, r) in self.unknown_obs :
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='orange',
                    fill=True,
                    alpha=0.4
                )
            )
        self.robot.test_type = 'cbf_qp'

    def get_nearest_obs(self, detected_obs):
        # if there was new obstacle detected, update the obs
        if len(detected_obs) != 0:
            all_obs = np.vstack((self.obs, detected_obs))
            return np.array(detected_obs).reshape(-1, 1)
        else:
            all_obs = self.obs

        if len(all_obs) == 0:
            return None

        radius = all_obs[:, 2]
        distances = np.linalg.norm(all_obs[:, :2] - self.robot.X[:2].T, axis=1)
        min_distance_index = np.argmin(distances-radius)
        nearest_obstacle = all_obs[min_distance_index]
        return nearest_obstacle.reshape(-1, 1)
    
    def is_collide_unknown(self):
        if self.unknown_obs is None:
            return False
        for obs in self.unknown_obs:
            # check if the robot collides with the obstacle
            robot_radius = self.robot.robot_radius
            distance = np.linalg.norm(self.robot.X[:2] - obs[:2])
            return distance < obs[2] + robot_radius

    def update_goal(self):
        '''
        Update the goal from waypoints
        '''
        if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
            self.current_goal_index += 1
            # Check if all waypoints are reached;
            if self.current_goal_index >= len(self.waypoints):
                print("All waypoints reached.")
                return None
        goal = np.array(self.waypoints[self.current_goal_index][0:2]) # set goal to next waypoint's (x,y)
        return goal

    def control_step(self):
        '''
        Simulate one step of tracking control with CBF-QP with the given waypoints.
        Output: 
            - -1: all waypoints reached
            - 0: normal
            - 1: visibility violation
            - raise QPError: if the QP is infeasible or the robot collides with the obstacle
        '''

        goal = self.update_goal()
        if goal is None:
            return -1

        # 1. Update the detected obstacles
        detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        nearest_obs = self.get_nearest_obs(detected_obs)

        # 2. Update the CBF constraints
        if nearest_obs is None:
            # deactivate the CBF constraints
            self.A1.value = np.zeros_like(self.A1.value)
            self.b1.value = np.zeros_like(self.b1.value)
        elif self.type == 'Unicycle2D':
            h, dh_dx = self.robot.agent_barrier(nearest_obs)
            self.A1.value[0,:] = dh_dx @ self.robot.g()
            self.b1.value[0,:] = dh_dx @ self.robot.f() + self.alpha * h
        elif self.type == 'DynamicUnicycle2D':
            h, h_dot, dh_dot_dx = self.robot.agent_barrier(nearest_obs)
            self.A1.value[0,:] = dh_dot_dx @ self.robot.g()
            self.b1.value[0,:] = dh_dot_dx @ self.robot.f() + (self.alpha1+self.alpha2) * h_dot + self.alpha1*self.alpha2*h

        # 3. Compuite nominal control input, pre-defined in the robot class
        self.u_ref.value = self.robot.nominal_input(goal)

         # 4. Solve this yields a new `self.u``
        self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)

        # 5. Raise an error if the QP is infeasible, or the robot collides with the obstacle
        collide = self.is_collide_unknown()
        if self.cbf_controller.status != 'optimal' or collide:
            if self.show_animation:
                self.robot.render_plot()
                current_position = self.robot.X[:2].flatten()
                self.ax.text(current_position[0]+0.5, current_position[1]+0.5, '!', color='red', weight='bold', fontsize=22)
                self.fig.canvas.draw()
                plt.pause(5)

            if self.save_animation:
                plt.savefig(self.current_directory_path +
                            "/output/animations/" + "t_step_" + str(self.ani_idx) + ".png")
            raise QPError

        # 6. Step the robot
        self.robot.step(self.u.value)
        if self.show_animation:
            self.robot.render_plot()

        # 7. Update sensing information
        self.robot.update_sensing_footprints()
        self.robot.update_safety_area()
        #if self.current_goal_index > 5: #FIXME:
        beyond_flag = self.robot.is_beyond_sensing_footprints()
        if beyond_flag and self.show_animation:
            print("Visibility Violation")

        if self.show_animation:
            self.fig.canvas.draw()
            plt.pause(0.01)
            if self.save_animation and self.ani_idx % self.save_per_frame == 0:
                plt.savefig(self.current_directory_path +
                            "/output/animations/" + "t_step_" + str(self.ani_idx) + ".png")
                self.ani_idx += 1

        return beyond_flag
    
    def run_all_steps(self, tf=30):
        print("===================================")
        print("============ Tracking =============")
        print("Start following the generated path.")
        unexpected_beh = 0

        for _ in range(int(tf / self.dt)):
            ret = self.control_step()
            unexpected_beh += ret
            if ret == -1: # all waypoints reached
                break
            
            
        # convert the image sequence to a video
        if self.show_animation and self.save_animation:
            subprocess.call(['ffmpeg',
                            '-i', self.current_directory_path+"/output/animations/" + "/t_step_%01d.png",
                            '-r', '60',  # Changes the output FPS to 30
                            '-pix_fmt', 'yuv420p',
                            self.current_directory_path+"/output/animations/tracking.mp4"])

            for file_name in glob.glob(self.current_directory_path +
                            "/output/animations/*.png"):
                os.remove(file_name)

        print("=====   Tracking finished    =====")
        print("===================================\n")
        if self.show_animation:
            plt.ioff()
            plt.close()

        return unexpected_beh

if __name__ == "__main__":
    dt = 0.05
    # import sys
    # import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from utils import plotting
    from utils import env
    import math

    env_type = env.type

    # temporal
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [10, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)
    env_handler = env.Env()

    #type = 'Unicycle2D'
    type = 'DynamicUnicycle2D'
    tracking_controller = LocalTrackingController(x_init, type=type, dt=dt,
                                         show_animation=True,
                                         save_animation=False,
                                         plotting=plot_handler,
                                         env=env_handler)
    # randomly generate 5 unknown obstacles
    x_range = env_handler.x_range
    y_range = env_handler.y_range
    # unknown_obs = np.random.uniform(low=[x_range[0], y_range[0], 0], high=[x_range[1], y_range[1], 0], size=(20, 3))
    # unknown_obs[:, 2] = 0.5
    # unknown_obs = np.vstack((unknown_obs, [
    #                     [10, 8, 0.5],
    #                     [10.5, 8, 0.5],
    #                     [11, 8, 0.5]])
    # )

    # if env_type == 1:
    #     unknown_obs = np.array([[13.0, 10.0, 0.5],
    #                             [12.0, 13.0, 0.5],
    #                             [15.0, 20.0, 0.5],
    #                             [20.5, 20.5, 0.5],
    #                             [24.0, 15.0, 0.5]])
    # elif env_type == 2: 
    #     unknown_obs = np.array([[9.0, 8.8, 0.3]]) 

    #tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=30)

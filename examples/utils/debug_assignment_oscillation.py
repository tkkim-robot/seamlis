import os
import sys

import numpy as np
from shapely.geometry import LineString

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exploration import ExplorationManager
from safe_control.utils import env
from examples.test_exploration import build_initial_states, get_robot_specs


def build_manager():
    num_agent = 1
    x0s = build_initial_states(num_agent)
    robot_specs = get_robot_specs(num_agent, use_astar=False)
    robot_specs[0]["unknown_obs_persistent_fov"] = True
    robot_specs[0]["visibility_violation_mode"] = "point_mass"
    robot_specs[0]["visibility_violation_tolerance"] = 0.02

    env_handler = env.Env(width=16.0, height=12.0, known_obs=np.empty((0, 7)), resolution=0.2)
    manager = ExplorationManager(
        x0s,
        robot_specs,
        {"pos": "mpc_cbf", "att": "velocity_tracking_yaw"},
        exploration_algorithm="CoScan",
        dt=0.1,
        show_animation=False,
        save_animation=False,
        env_handler=env_handler,
        known_obs=np.empty((0, 7)),
        unknown_obs=np.empty((0, 3)),
        use_astar_waypoints=False,
        coverage_target=0.98,
    )
    manager.latest_obstacle_map = manager.get_obstacle_map()
    return manager


def set_robot_pose(manager, xy):
    robot = manager.controller_list[0].robot
    robot.X[0, 0] = float(xy[0])
    robot.X[1, 0] = float(xy[1])
    if robot.X.shape[0] >= 4:
        robot.X[2, 0] = 0.0
        robot.X[3, 0] = 0.0


def frontier_cells_to_line(manager, frontier_cells):
    frontier_cells = np.array(frontier_cells, dtype=np.int32)
    frontier_points = manager.env_handler.grid_to_f(frontier_cells)[:, :2]
    return LineString(frontier_points.tolist())


def run_progress_case(manager):
    manager._reset_goal_assignment_history(0)
    manager._deadlock_cooldown[0] = 0

    progress_positions = [
        [4.0, 4.0],
        [4.9, 4.1],
        [5.8, 4.2],
        [6.7, 4.3],
        [7.6, 4.4],
    ]
    progress_goals = [
        [5.8, 4.6],
        [6.7, 4.7],
        [7.6, 4.8],
        [8.5, 4.9],
        [9.4, 5.0],
    ]

    triggered = False
    for pos, goal in zip(progress_positions, progress_goals):
        set_robot_pose(manager, pos)
        triggered = triggered or manager._record_goal_assignment(0, np.array(goal, dtype=float))

    return triggered


def run_oscillation_case(manager):
    manager._reset_goal_assignment_history(0)
    manager._deadlock_cooldown[0] = 0
    manager._deadlock_exclusions[0] = []

    local_trap_frontiers = [
        [34, 20],
        [35, 21],
        [34, 22],
        [45, 35],
        [46, 36],
        [45, 37],
        [60, 28],
        [61, 29],
        [62, 30],
    ]
    manager.frontiers = frontier_cells_to_line(manager, local_trap_frontiers)

    oscillation_positions = [
        [8.0, 6.0],
        [8.2, 6.4],
        [7.9, 6.1],
        [8.1, 6.3],
        [8.0, 6.0],
    ]
    oscillation_goals = [
        [6.8, 4.2],
        [9.2, 7.6],
        [6.9, 4.4],
        [9.1, 7.8],
        [6.8, 4.3],
    ]

    triggered = False
    for pos, goal in zip(oscillation_positions, oscillation_goals):
        set_robot_pose(manager, pos)
        triggered = manager._record_goal_assignment(0, np.array(goal, dtype=float)) or triggered

    recovery_goal = manager._select_recovery_goal(0, avoid_goal=np.array(oscillation_goals[-1], dtype=float))
    return triggered, recovery_goal


def main():
    manager = build_manager()

    progress_triggered = run_progress_case(manager)
    print(f"progress_case_triggered={progress_triggered}")
    if progress_triggered:
        raise RuntimeError("Progress case should not trigger assignment-churn recovery.")

    oscillation_triggered, recovery_goal = run_oscillation_case(manager)
    print(f"oscillation_case_triggered={oscillation_triggered}")
    print(f"oscillation_recovery_goal={None if recovery_goal is None else np.round(recovery_goal, 3).tolist()}")

    if not oscillation_triggered:
        raise RuntimeError("Oscillation case should trigger assignment-churn recovery.")
    if recovery_goal is None or float(recovery_goal[0]) <= 10.5:
        raise RuntimeError("Recovery goal should jump to the distant frontier band.")

    print("debug_assignment_oscillation: PASS")


if __name__ == "__main__":
    main()

import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.co_scan import CoScanPlanner
from algorithms.frontier_vanilla import FrontierPlanner


def build_maps():
    obstacle = np.zeros((8, 10), dtype=np.int32)
    frontier = np.zeros_like(obstacle)
    frontier[2, 7] = 1
    frontier[5, 3] = 1
    frontier[6, 8] = 1
    return obstacle, frontier


def assert_goals_in_bounds(goals, shape):
    h, w = shape
    for goal in goals:
        if goal is None:
            continue
        gx, gy = int(goal[0]), int(goal[1])
        assert 0 <= gx < w, f"x goal out of bounds: {gx}"
        assert 0 <= gy < h, f"y goal out of bounds: {gy}"


def run_boundary_regression():
    obstacle, frontier = build_maps()
    # Deliberately exceed both x and y map limits.
    agent_pos = np.array(
        [
            [10, 8],
            [9, 7],
        ],
        dtype=np.int32,
    )
    agent_orientations = np.array([0.0, -np.pi / 2.0], dtype=float)

    coscan = CoScanPlanner()
    frontier_planner = FrontierPlanner()

    coscan_goals = coscan.get_long_term_goals(obstacle, frontier, agent_pos, agent_orientations)
    frontier_goals = frontier_planner.get_long_term_goals(obstacle, frontier, agent_pos, agent_orientations)

    assert_goals_in_bounds(coscan_goals, obstacle.shape)
    assert_goals_in_bounds(frontier_goals, obstacle.shape)

    # Also verify the helper paths that index into the maps do not crash.
    coscan.global_goals = np.array([[99, 99], [3, 5]], dtype=np.int32)
    assert coscan.check_finish(frontier, [False, False]) is True
    replanned = coscan.replan(obstacle, frontier, goal=np.array([99, 99], dtype=np.int32), planning_window=(0, obstacle.shape[1], 0, obstacle.shape[0]))
    assert isinstance(replanned, tuple) and len(replanned) == 2


if __name__ == "__main__":
    run_boundary_regression()
    print("Planner boundary clipping regression passed.")

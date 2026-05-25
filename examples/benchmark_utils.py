"""Shared benchmark helpers used by the public run_benchmark.py entrypoint."""

import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_key, "1")

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exploration import ExplorationManager
from safe_control.utils import env
from examples.test_exploration import build_initial_states, get_robot_specs


ATTITUDES = ["velocity_tracking_yaw", "simple", "visibility_area", "gatekeeper"]
BENCHMARK_START_POSITIONS = np.array(
    [
        [2.0, 2.0],
        [2.0, 16.0],
        [22.0, 4.0],
    ],
    dtype=float,
)
BENCHMARK_CHALLENGING_YAWS = np.array(
    [
        -np.pi / 2.0,
        -np.pi / 2.0,
        0.0,
    ],
    dtype=float,
)


DEFAULT_GATEKEEPER_PARAMS = {
    "gatekeeper_nominal": "visibility_area",
    "gatekeeper_backup": "velocity_tracking_yaw",
    # Benchmark defaults: allow longer nominal commitments while keeping
    # a moderate backup horizon so gatekeeper stays faster than pure VT
    # without giving up the early blindside veto.
    "gatekeeper_nominal_horizon": 1.3,
    "gatekeeper_backup_horizon": 1.2,
    "gatekeeper_event_offset": 0.0,
    "gatekeeper_horizon_discount": 0.05,
    "gatekeeper_validation_slack": 0.15,
    "gatekeeper_validation_tube_margin": 0.20,
    "gatekeeper_braking_distance_margin": 1.00,
}


@dataclass
class MapConfig:
    map_id: str
    known_obs: np.ndarray
    unknown_obs: np.ndarray
    source_seed: int
    initial_states: Optional[np.ndarray] = None


def _angle_normalize(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def sample_benchmark_initial_states(
    rng: np.random.Generator,
    yaw_jitter_deg: float = 0.0,
) -> np.ndarray:
    states = np.hstack((BENCHMARK_START_POSITIONS.copy(), BENCHMARK_CHALLENGING_YAWS.reshape(-1, 1)))
    yaw_jitter = rng.uniform(
        -np.deg2rad(yaw_jitter_deg),
        np.deg2rad(yaw_jitter_deg),
        size=states.shape[0],
    )
    states[:, 2] = _angle_normalize(states[:, 2] + yaw_jitter)
    return states


def _superellipse_outside(x: float, y: float, radius: float, walls: np.ndarray, margin: float = 0.04) -> bool:
    if walls.size == 0:
        return True
    for wall in walls:
        ox, oy, a, b, e, theta = wall[:6]
        ct = math.cos(theta)
        st = math.sin(theta)
        px = ct * (x - ox) + st * (y - oy)
        py = -st * (x - ox) + ct * (y - oy)
        den_a = max(abs(a) + radius + margin, 1e-3)
        den_b = max(abs(b) + radius + margin, 1e-3)
        expo = max(float(e), 2.0)
        h = (abs(px) / den_a) ** expo + (abs(py) / den_b) ** expo - 1.0
        if h <= 0.0:
            return False
    return True


def _circles_nonoverlap(x: float, y: float, radius: float, circles: np.ndarray, margin: float = 0.08) -> bool:
    if circles.size == 0:
        return True
    dxy = circles[:, :2] - np.array([x, y], dtype=float).reshape(1, 2)
    d = np.linalg.norm(dxy, axis=1)
    return bool(np.all(d > (circles[:, 2] + radius + margin)))


def _sample_circle(
    rng: np.random.Generator,
    width: float,
    height: float,
    walls: np.ndarray,
    existing_circles: np.ndarray,
    starts_xy: np.ndarray,
    r_min: float,
    r_max: float,
    start_clearance: float,
    clearance_margin: float,
    wall_margin: float,
    boundary_margin: float,
    mode: str,
    centers: Optional[np.ndarray] = None,
    cluster_std: float = 0.85,
    max_tries: int = 1500,
) -> Optional[np.ndarray]:
    for _ in range(max_tries):
        r = float(rng.uniform(r_min, r_max))
        if mode == "cluster" and centers is not None and len(centers) > 0:
            center = centers[int(rng.integers(0, len(centers)))]
            x = float(center[0] + rng.normal(0.0, cluster_std))
            y = float(center[1] + rng.normal(0.0, cluster_std))
        else:
            x = float(rng.uniform(r + 0.4, width - r - 0.4))
            y = float(rng.uniform(r + 0.4, height - r - 0.4))

        if x <= r + boundary_margin or x >= width - r - boundary_margin or y <= r + boundary_margin or y >= height - r - boundary_margin:
            continue
        if not _superellipse_outside(x, y, r, walls, margin=wall_margin):
            continue
        if not _circles_nonoverlap(x, y, r, existing_circles, margin=clearance_margin):
            continue
        if starts_xy.size > 0:
            d_start = np.linalg.norm(starts_xy - np.array([x, y], dtype=float).reshape(1, 2), axis=1)
            if np.any(d_start <= (r + start_clearance)):
                continue
        return np.array([x, y, r], dtype=float)
    return None


def _fixed_circle_valid(
    x: float,
    y: float,
    radius: float,
    walls: np.ndarray,
    existing_circles: np.ndarray,
    starts_xy: np.ndarray,
    start_clearance: float,
    clearance_margin: float,
    wall_margin: float,
    boundary_margin: float,
    width: float,
    height: float,
) -> bool:
    if x <= radius + boundary_margin or x >= width - radius - boundary_margin:
        return False
    if y <= radius + boundary_margin or y >= height - radius - boundary_margin:
        return False
    if not _superellipse_outside(x, y, radius, walls, margin=wall_margin):
        return False
    if not _circles_nonoverlap(x, y, radius, existing_circles, margin=clearance_margin):
        return False
    if starts_xy.size > 0:
        d_start = np.linalg.norm(starts_xy - np.array([x, y], dtype=float).reshape(1, 2), axis=1)
        if np.any(d_start <= (radius + start_clearance)):
            return False
    return True


def generate_random_indoor_map(
    rng: np.random.Generator,
    map_index: int,
    base_known_obs: np.ndarray,
    env_width: float,
    env_height: float,
    starts_xy: np.ndarray,
) -> MapConfig:
    map_seed = int(rng.integers(0, 2**31 - 1))
    seed_seq = np.random.SeedSequence(map_seed)
    (
        state_seq,
        known_seq,
        ab_anchor_seq,
        background_seq,
        source_seq,
    ) = seed_seq.spawn(5)
    state_rng = np.random.default_rng(state_seq)
    known_rng = np.random.default_rng(known_seq)
    ab_anchor_rng = np.random.default_rng(ab_anchor_seq)
    background_rng = np.random.default_rng(background_seq)
    source_rng = np.random.default_rng(source_seq)

    initial_states = sample_benchmark_initial_states(state_rng)
    max_robot_radius = 0.18
    # Keep random maps dense, but reject obstacle placements that create
    # near-impossible single-robot passages.
    min_obstacle_boundary_gap = 2.0 * max_robot_radius + 0.20

    flags = base_known_obs[:, 6] if base_known_obs.size > 0 else np.zeros(0, dtype=float)
    walls = base_known_obs[np.isclose(flags, 1.0)].copy() if base_known_obs.size > 0 else np.empty((0, 7), dtype=float)
    base_circles = base_known_obs[~np.isclose(flags, 1.0), :3].copy() if base_known_obs.size > 0 else np.empty((0, 3), dtype=float)
    if base_circles.size == 0:
        base_circles = np.array(
            [
                [3.0, 3.0, 0.35],
                [3.0, 15.0, 0.35],
                [9.0, 4.0, 0.35],
                [11.0, 14.0, 0.35],
                [13.0, 6.0, 0.35],
                [18.0, 7.0, 0.35],
                [20.0, 15.0, 0.35],
                [21.0, 10.0, 0.35],
            ],
            dtype=float,
        )

    known_circles: List[np.ndarray] = []
    for base in base_circles:
        accepted = None
        existing = np.array(known_circles, dtype=float) if len(known_circles) > 0 else np.empty((0, 3), dtype=float)
        for _ in range(400):
            r = float(np.clip(base[2] + known_rng.normal(0.0, 0.05), 0.28, 0.44))
            x = float(np.clip(base[0] + known_rng.normal(0.0, 0.35), r + 0.5, env_width - r - 0.5))
            y = float(np.clip(base[1] + known_rng.normal(0.0, 0.35), r + 0.5, env_height - r - 0.5))
            if not _superellipse_outside(x, y, r, walls, margin=0.04):
                continue
            if not _circles_nonoverlap(x, y, r, existing, margin=min_obstacle_boundary_gap):
                continue
            d_start = np.linalg.norm(starts_xy - np.array([x, y], dtype=float).reshape(1, 2), axis=1)
            if np.any(d_start <= (r + 0.8)):
                continue
            accepted = np.array([x, y, r], dtype=float)
            break

        if accepted is None:
            base_ok = (
                _superellipse_outside(float(base[0]), float(base[1]), float(base[2]), walls, margin=0.04)
                and _circles_nonoverlap(
                    float(base[0]),
                    float(base[1]),
                    float(base[2]),
                    existing,
                    margin=min_obstacle_boundary_gap,
                )
            )
            if base_ok and starts_xy.size > 0:
                d_start = np.linalg.norm(starts_xy - base[:2].reshape(1, 2), axis=1)
                base_ok = bool(np.all(d_start > (float(base[2]) + 0.8)))
            if base_ok:
                accepted = base.copy()

        if accepted is not None:
            known_circles.append(accepted)

    known_circles_arr = np.array(known_circles, dtype=float) if len(known_circles) > 0 else np.empty((0, 3), dtype=float)
    known_circles_arr = np.hstack((known_circles_arr, np.zeros((known_circles_arr.shape[0], 4), dtype=float)))
    known_obs = np.vstack((known_circles_arr, walls))

    # Unknown-obstacle sampling: denser corridor-adjacent clusters to stress
    # side-looking attitudes while remaining solvable for forward-looking ones.
    blindside_primary_ab = np.array(
        [
            [3.92, 2.06, 0.265],
            [4.46, 2.28, 0.295],
            [5.55, 2.05, 0.255],
            [3.92, 15.94, 0.265],
            [4.46, 15.72, 0.295],
            [5.55, 14.95, 0.255],
        ],
        dtype=float,
    )
    blindside_anchors_ab = np.array(
        [
            [5.00, 2.60],
            [5.00, 15.40],
        ],
        dtype=float,
    )
    blindside_anchors_ab += ab_anchor_rng.normal(0.0, 0.04, size=blindside_anchors_ab.shape)

    cluster_centers_ab = np.array(
        [
            [5.60, 2.90],
            [5.60, 15.10],
        ],
        dtype=float,
    )
    cluster_centers_bg = np.array(
        [
            [5.0, 5.5],
            [9.8, 11.6],
            [11.6, 3.3],
            [20.2, 10.8],
            [8.5, 16.0],
            [10.6, 8.9],
            [6.7, 10.3],
            [14.6, 8.0],
            [17.9, 12.9],
        ],
        dtype=float,
    )
    cluster_centers_ab += ab_anchor_rng.normal(0.0, 0.20, size=cluster_centers_ab.shape)
    cluster_centers_bg += background_rng.normal(0.0, 0.20, size=cluster_centers_bg.shape)
    cluster_centers = np.vstack((cluster_centers_ab, cluster_centers_bg))

    unknown_circles: List[np.ndarray] = []
    known_for_clearance = known_obs[:, :3].copy() if known_obs.size > 0 else np.empty((0, 3), dtype=float)
    total_unknown = 14
    clustered_unknown = 8

    for x, y, r in blindside_primary_ab:
        existing_unknown = np.array(unknown_circles, dtype=float) if len(unknown_circles) > 0 else np.empty((0, 3), dtype=float)
        combined_existing = existing_unknown
        if known_for_clearance.size > 0:
            combined_existing = (
                np.vstack((known_for_clearance, existing_unknown))
                if existing_unknown.size > 0
                else known_for_clearance
            )
        if _fixed_circle_valid(
            x=float(x),
            y=float(y),
            radius=float(r),
            walls=walls,
            existing_circles=combined_existing,
            starts_xy=starts_xy,
            start_clearance=0.95,
            clearance_margin=0.18,
            wall_margin=0.06,
            boundary_margin=0.24,
            width=env_width,
            height=env_height,
        ):
            unknown_circles.append(np.array([x, y, r], dtype=float))

    for anchor in blindside_anchors_ab:
        existing_unknown = np.array(unknown_circles, dtype=float) if len(unknown_circles) > 0 else np.empty((0, 3), dtype=float)
        combined_existing = existing_unknown
        if known_for_clearance.size > 0:
            combined_existing = (
                np.vstack((known_for_clearance, existing_unknown))
                if existing_unknown.size > 0
                else known_for_clearance
            )
        sampled = _sample_circle(
            rng=ab_anchor_rng,
            width=env_width,
            height=env_height,
            walls=walls,
            existing_circles=combined_existing,
            starts_xy=starts_xy,
            r_min=0.24,
            r_max=0.28,
            start_clearance=0.95,
            clearance_margin=0.18,
            wall_margin=0.06,
            boundary_margin=0.24,
            mode="cluster",
            centers=anchor.reshape(1, 2),
            cluster_std=0.10,
            max_tries=800,
        )
        if sampled is not None:
            unknown_circles.append(sampled)

    for idx in range(max(total_unknown - len(unknown_circles), 0)):
        mode = "cluster" if idx < clustered_unknown else "uniform"
        existing_unknown = np.array(unknown_circles, dtype=float) if len(unknown_circles) > 0 else np.empty((0, 3), dtype=float)
        combined_existing = existing_unknown
        if known_for_clearance.size > 0:
            combined_existing = (
                np.vstack((known_for_clearance, existing_unknown))
                if existing_unknown.size > 0
                else known_for_clearance
            )
        sampled = _sample_circle(
            rng=background_rng,
            width=env_width,
            height=env_height,
            walls=walls,
            existing_circles=combined_existing,
            starts_xy=starts_xy,
            r_min=0.24,
            r_max=0.30,
            start_clearance=0.95,
            clearance_margin=0.20,
            wall_margin=0.06,
            boundary_margin=0.24,
            mode=mode,
            centers=cluster_centers if mode == "cluster" else None,
            cluster_std=0.45,
            max_tries=2000,
        )
        if sampled is not None:
            unknown_circles.append(sampled)

    if len(unknown_circles) == 0:
        unknown_circles.append(np.array([9.8, 11.8, 0.30], dtype=float))
    unknown_obs = np.array(unknown_circles, dtype=float)

    return MapConfig(
        map_id=f"map_{map_index:03d}",
        known_obs=known_obs,
        unknown_obs=unknown_obs,
        source_seed=int(source_rng.integers(0, 2**31 - 1)),
        initial_states=initial_states,
    )


def run_exploration_case(
    map_cfg: MapConfig,
    num_agent: int,
    algo: str,
    attitude: str,
    dt: float,
    tf: float,
    coverage_target: float,
    use_astar: bool,
    gatekeeper_params: Dict[str, float],
    robot_spec_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    # Make each benchmark case deterministic regardless of process scheduling.
    geom_hash = hashlib.sha256()
    geom_hash.update(np.ascontiguousarray(map_cfg.known_obs, dtype=np.float64).tobytes())
    geom_hash.update(np.ascontiguousarray(map_cfg.unknown_obs, dtype=np.float64).tobytes())
    if map_cfg.initial_states is not None:
        geom_hash.update(np.ascontiguousarray(map_cfg.initial_states, dtype=np.float64).tobytes())
    geom_digest = geom_hash.hexdigest()
    seed_key = f"{map_cfg.source_seed}|{geom_digest}|{num_agent}|{algo}|{attitude}"
    seed_bytes = hashlib.sha256(seed_key.encode("utf-8")).digest()
    case_seed = int.from_bytes(seed_bytes[:4], byteorder="little", signed=False)
    np.random.seed(case_seed)
    random.seed(case_seed)

    if map_cfg.initial_states is not None:
        x0s = [np.array(s, dtype=float).copy() for s in np.asarray(map_cfg.initial_states, dtype=float)[:num_agent]]
    else:
        x0s = build_initial_states(num_agent)
    robot_specs = get_robot_specs(num_agent, use_astar=use_astar)
    for spec in robot_specs:
        spec["unknown_obs_persistent_fov"] = True
        # Fair visibility-violation accounting across all attitudes.
        spec["visibility_violation_mode"] = "point_mass"
        spec["visibility_violation_tolerance"] = 0.02
        if attitude == "gatekeeper":
            spec["w_max"] = float(spec.get("w_max", 1.2))
            spec.update(gatekeeper_params)
        if robot_spec_overrides:
            spec.update(robot_spec_overrides)

    env_handler = env.Env(
        width=24.0,
        height=18.0,
        known_obs=map_cfg.known_obs,
        resolution=0.16,
    )
    controller_type = {"pos": "mpc_cbf", "att": attitude}
    exploration_algorithm = "CoScan" if algo == "coscan" else "Frontier"

    manager = ExplorationManager(
        x0s,
        robot_specs,
        controller_type,
        exploration_algorithm=exploration_algorithm,
        dt=dt,
        show_animation=False,
        save_animation=False,
        env_handler=env_handler,
        known_obs=map_cfg.known_obs,
        unknown_obs=map_cfg.unknown_obs,
        use_astar_waypoints=use_astar,
        coverage_target=coverage_target,
    )

    max_steps = int(tf / dt)
    t0 = time.perf_counter()
    success = bool(manager.explore(max_steps=max_steps))
    wallclock = float(time.perf_counter() - t0)

    violations = [len(c.robot.unsafe_points) for c in manager.controller_list]
    visibility_total = int(sum(violations))

    collision_info = manager.last_collision_info if manager.last_collision_info is not None else {}
    collision_type = str(collision_info.get("type", "")) if isinstance(collision_info, dict) else ""
    collision_stage = str(collision_info.get("stage", "")) if isinstance(collision_info, dict) else ""
    unknown_collision = collision_type == "unknown"
    collision_or_infeasible = manager.last_termination_reason == "collision_or_infeasible"

    result = {
        "map_id": map_cfg.map_id,
        "num_agent": int(num_agent),
        "algo": str(algo),
        "attitude": str(attitude),
        "success": bool(success),
        "coverage": float(manager.last_coverage_ratio),
        "termination_reason": str(manager.last_termination_reason),
        "step_count": int(manager.last_step_count),
        "sim_time": float(manager.last_sim_time),
        "wallclock_time": wallclock,
        "visibility_total": visibility_total,
        "visibility_per_robot": json.dumps(violations),
        "collision_or_infeasible": bool(collision_or_infeasible),
        "collision_type": collision_type,
        "collision_stage": collision_stage,
        "unknown_collision": bool(unknown_collision),
        "exploration_time_success": float(manager.last_sim_time) if success else None,
    }

    if isinstance(collision_info, dict) and len(collision_info) > 0:
        result["collision_info_json"] = json.dumps(collision_info)
    else:
        result["collision_info_json"] = ""

    if attitude == "gatekeeper":
        nominal_usage = []
        for controller in manager.controller_list:
            att_ctrl = getattr(controller, "att_controller", None)
            if att_ctrl is not None and hasattr(att_ctrl, "get_stats"):
                stats = att_ctrl.get_stats()
                nominal_usage.append(float(stats.get("nominal_seconds_avg_per_commit", 0.0)))
        result["gatekeeper_nominal_avg"] = float(np.mean(nominal_usage)) if len(nominal_usage) > 0 else 0.0
    else:
        result["gatekeeper_nominal_avg"] = None

    # Benchmark runs are headless; close figures promptly to avoid process-level
    # matplotlib figure accumulation during large sweeps.
    try:
        plt.close(manager.fig)
    except Exception:
        pass

    return result


def append_csv(path: str, row: Dict[str, object], header: List[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_results(
    rows: List[Dict[str, object]],
    paper_timeout_by_agents: Optional[Dict[int, float]] = None,
    explore_timeout_by_agents: Optional[Dict[int, float]] = None,
) -> List[Dict[str, object]]:
    def _to_bool(v: object) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ["1", "true", "yes", "y", "t"]

    grouped: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["algo"], int(row["num_agent"]), row["attitude"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        algo, num_agent, attitude = key
        g = grouped[key]
        n = len(g)
        raw_success_mask = np.array([_to_bool(r.get("success")) for r in g], dtype=bool)
        collision_mask = np.array([_to_bool(r.get("collision_or_infeasible")) for r in g], dtype=bool)
        unknown_collision_mask = np.array([_to_bool(r.get("unknown_collision")) for r in g], dtype=bool)
        vis_totals = np.array([int(r["visibility_total"]) for r in g], dtype=float)
        sim_times = np.array([float(r["sim_time"]) for r in g], dtype=float)
        paper_timeout = None
        if paper_timeout_by_agents is not None:
            paper_timeout = paper_timeout_by_agents.get(int(num_agent))
        explore_timeout = None
        if explore_timeout_by_agents is not None:
            explore_timeout = explore_timeout_by_agents.get(int(num_agent))
        if paper_timeout is None:
            success_mask = raw_success_mask
        else:
            success_mask = np.logical_and(raw_success_mask, sim_times <= float(paper_timeout))
        success_times_no_timeout = np.array(
            [
                float(r["exploration_time_success"])
                for r in g
                if r.get("exploration_time_success") not in [None, ""]
            ],
            dtype=float,
        )
        success_times_with_timeout = np.array(
            [
                float(r["exploration_time_success"])
                for r in g
                if r.get("exploration_time_success") not in [None, ""]
                and (
                    explore_timeout is None
                    or float(r["exploration_time_success"]) <= float(explore_timeout)
                )
            ],
            dtype=float,
        )

        summary_rows.append(
            {
                "algo": algo,
                "num_agent": num_agent,
                "attitude": attitude,
                "trials": n,
                "success_rate": float(np.mean(success_mask)) if n > 0 else 0.0,
                "collision_or_infeasible_rate": float(np.mean(collision_mask)) if n > 0 else 0.0,
                "unknown_collision_rate": float(np.mean(unknown_collision_mask)) if n > 0 else 0.0,
                "mean_visibility_total": float(np.mean(vis_totals)) if n > 0 else 0.0,
                "mean_visibility_per_robot": float(np.mean(vis_totals / max(num_agent, 1))) if n > 0 else 0.0,
                "mean_sim_time_all": float(np.mean(sim_times)) if n > 0 else 0.0,
                "mean_exploration_time_success_only_no_timeout": (
                    float(np.mean(success_times_no_timeout)) if success_times_no_timeout.size > 0 else None
                ),
                "mean_exploration_time_success_only_with_timeout": (
                    float(np.mean(success_times_with_timeout)) if success_times_with_timeout.size > 0 else None
                ),
            }
        )
    return summary_rows


def render_summary_markdown(
    summary_rows: List[Dict[str, object]],
    output_path: str,
    representative_map_id: str,
    gatekeeper_params: Dict[str, float],
    metadata_lines: Optional[List[str]] = None,
) -> None:
    lines = []
    lines.append("# Exploration Benchmark Summary")
    lines.append("")
    lines.append(f"- Representative map id: `{representative_map_id}`")
    lines.append(f"- Gatekeeper params: `{json.dumps(gatekeeper_params, sort_keys=True)}`")
    if metadata_lines:
        lines.extend(metadata_lines)
    lines.append("")
    lines.append("| Algo | Agents | Attitude | Trials | Success Rate | Collision/Infeasible Rate | Unknown Collision Rate | Mean Vis. Viol. (Total) | Mean Vis. Viol. / Robot | Mean Sim Time (All) | Mean Explore Time (Success Only, No Timeout) | Mean Explore Time (Success Only, With Timeout) |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        success_only_no_timeout = (
            f"{r['mean_exploration_time_success_only_no_timeout']:.2f}"
            if r["mean_exploration_time_success_only_no_timeout"] is not None
            else "N/A"
        )
        success_only_with_timeout = (
            f"{r['mean_exploration_time_success_only_with_timeout']:.2f}"
            if r["mean_exploration_time_success_only_with_timeout"] is not None
            else "N/A"
        )
        lines.append(
            "| {algo} | {num_agent} | {attitude} | {trials} | {sr:.2f} | {cr:.2f} | {ucr:.2f} | {mv:.2f} | {mvr:.2f} | {mt:.2f} | {mst0} | {mst1} |".format(
                algo=r["algo"],
                num_agent=r["num_agent"],
                attitude=r["attitude"],
                trials=r["trials"],
                sr=r["success_rate"],
                cr=r["collision_or_infeasible_rate"],
                ucr=r["unknown_collision_rate"],
                mv=r["mean_visibility_total"],
                mvr=r["mean_visibility_per_robot"],
                mt=r["mean_sim_time_all"],
                mst0=success_only_no_timeout,
                mst1=success_only_with_timeout,
            )
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def map_to_serializable(map_cfg: MapConfig) -> Dict[str, object]:
    return {
        "map_id": map_cfg.map_id,
        "source_seed": int(map_cfg.source_seed),
        "known_obs": map_cfg.known_obs.tolist(),
        "unknown_obs": map_cfg.unknown_obs.tolist(),
        "initial_states": None if map_cfg.initial_states is None else np.asarray(map_cfg.initial_states, dtype=float).tolist(),
    }

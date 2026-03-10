import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_key, "1")

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.benchmark_random_exploration import (  # noqa: E402
    AGENT_COUNTS,
    ALGOS,
    ATTITUDES,
    DEFAULT_GATEKEEPER_PARAMS,
    MapConfig,
    append_csv,
    find_hero_map_id,
    generate_random_indoor_map,
    map_to_serializable,
    render_summary_markdown,
    run_exploration_case,
    summarize_results,
)
from examples.test_exploration import build_indoor_exploration_env, build_initial_states  # noqa: E402


FIXED_UNKNOWN = np.array(
    [
        [3.92, 2.06, 0.265],
        [3.92, 15.94, 0.265],
        [5.55, 14.95, 0.255],
        [6.87, 10.24, 0.244],
        [8.32, 16.57, 0.243],
        [17.87, 12.28, 0.275],
        [14.93, 8.13, 0.248],
        [21.24, 16.08, 0.290],
    ],
    dtype=float,
)

VARIANT_B_EXTRA = np.array([[4.90, 15.40, 0.26]], dtype=float)

# Final curated recipe:
# - 1x layout 1: raises visibility-area violations without harming the safe pair
# - 3x layout 2: preserves the clean visibility-area failure case
# - 6x layout 2 + B-side variant: keeps the same collision case while making
#   velocity-tracking yaw meaningfully slower than gatekeeper
CURATED_LAYOUT_RECIPE: List[Tuple[int, str]] = [
    (1, "base"),
    (2, "base"),
    (2, "base"),
    (2, "base"),
    (2, "variant_b"),
    (2, "variant_b"),
    (2, "variant_b"),
    (2, "variant_b"),
    (2, "variant_b"),
    (2, "variant_b"),
]


def build_map(seed: int, layout_index: int, variant: str, map_id: str) -> MapConfig:
    env_width, env_height, base_known_obs, _ = build_indoor_exploration_env()
    starts = np.array([s[:2] for s in build_initial_states(3)], dtype=float)
    rng = np.random.default_rng(seed)
    map_cfg = None
    for idx in range(layout_index + 1):
        map_cfg = generate_random_indoor_map(
            rng=rng,
            map_index=idx,
            base_known_obs=base_known_obs,
            env_width=env_width,
            env_height=env_height,
            starts_xy=starts,
        )
    if map_cfg is None:
        raise RuntimeError(f"Failed to generate layout index {layout_index}")
    if variant == "base":
        unknown_obs = FIXED_UNKNOWN.copy()
    elif variant == "variant_b":
        unknown_obs = np.vstack((FIXED_UNKNOWN, VARIANT_B_EXTRA))
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    map_cfg.map_id = map_id
    map_cfg.unknown_obs = unknown_obs
    return map_cfg


def build_trial_maps(seed: int) -> List[MapConfig]:
    return [
        build_map(seed=seed, layout_index=layout_index, variant=variant, map_id=f"a00_m{i:02d}")
        for i, (layout_index, variant) in enumerate(CURATED_LAYOUT_RECIPE)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final curated exploration benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tf", type=float, default=600.0)
    parser.add_argument("--coverage_target", type=float, default=0.98)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/benchmark_seed42_curated_final_candidate",
    )
    parser.add_argument("--hero_max_gatekeeper_visibility", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "raw_results.csv")
    maps_path = os.path.join(args.output_dir, "trial_maps.json")
    hero_path = os.path.join(args.output_dir, "hero_case.json")
    summary_path = os.path.join(args.output_dir, "summary.md")

    if os.path.exists(raw_path):
        os.remove(raw_path)

    trial_maps = build_trial_maps(args.seed)
    jobs = [
        (m, n, algo, att)
        for m in trial_maps
        for n in AGENT_COUNTS
        for algo in ALGOS
        for att in ATTITUDES
    ]
    fieldnames = [
        "map_id",
        "num_agent",
        "algo",
        "attitude",
        "success",
        "coverage",
        "termination_reason",
        "step_count",
        "sim_time",
        "wallclock_time",
        "visibility_total",
        "visibility_per_robot",
        "collision_or_infeasible",
        "collision_type",
        "collision_stage",
        "unknown_collision",
        "exploration_time_success",
        "collision_info_json",
        "gatekeeper_nominal_avg",
        "attempt",
    ]
    rows: List[Dict[str, object]] = []

    if args.workers <= 1:
        for done, (m, n, algo, att) in enumerate(jobs, start=1):
            row = run_exploration_case(
                m,
                n,
                algo,
                att,
                args.dt,
                args.tf,
                args.coverage_target,
                True,
                DEFAULT_GATEKEEPER_PARAMS,
            )
            row["attempt"] = 0
            rows.append(row)
            append_csv(raw_path, row={k: row.get(k, "") for k in fieldnames}, header=fieldnames)
            print(
                f"done {done:03d}/{len(jobs)} map={m.map_id} n={n} algo={algo} att={att} "
                f"success={row['success']} coll={row['collision_or_infeasible']} vis={row['visibility_total']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_to_job = {}
            for m, n, algo, att in jobs:
                fut = ex.submit(
                    run_exploration_case,
                    m,
                    n,
                    algo,
                    att,
                    args.dt,
                    args.tf,
                    args.coverage_target,
                    True,
                    DEFAULT_GATEKEEPER_PARAMS,
                )
                future_to_job[fut] = (m.map_id, n, algo, att)
            done = 0
            for fut in as_completed(future_to_job):
                done += 1
                map_id, n, algo, att = future_to_job[fut]
                row = fut.result()
                row["attempt"] = 0
                rows.append(row)
                append_csv(raw_path, row={k: row.get(k, "") for k in fieldnames}, header=fieldnames)
                print(
                    f"done {done:03d}/{len(jobs)} map={map_id} n={n} algo={algo} att={att} "
                    f"success={row['success']} coll={row['collision_or_infeasible']} vis={row['visibility_total']}",
                    flush=True,
                )

    rows.sort(
        key=lambda r: (
            str(r["map_id"]),
            int(r["num_agent"]),
            str(r["algo"]),
            str(r["attitude"]),
        )
    )
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump([map_to_serializable(m) for m in trial_maps], f, indent=2)

    hero_map_id = find_hero_map_id(rows, max_gatekeeper_visibility=args.hero_max_gatekeeper_visibility)
    if hero_map_id is None:
        hero_map_id = trial_maps[0].map_id
    hero_map = next(m for m in trial_maps if m.map_id == hero_map_id)
    with open(hero_path, "w", encoding="utf-8") as f:
        json.dump(map_to_serializable(hero_map), f, indent=2)

    summary_rows = summarize_results(rows)
    render_summary_markdown(
        summary_rows=summary_rows,
        output_path=summary_path,
        hero_map_id=hero_map_id,
        gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
    )

    print(f"Saved {raw_path}")
    print(f"Saved {maps_path}")
    print(f"Saved {hero_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

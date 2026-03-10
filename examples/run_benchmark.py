import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.benchmark_utils import (  # noqa: E402
    ATTITUDES,
    DEFAULT_GATEKEEPER_PARAMS,
    MapConfig,
    append_csv,
    generate_random_indoor_map,
    map_to_serializable,
    render_summary_markdown,
    run_exploration_case,
    summarize_results,
)
from examples.test_exploration import (  # noqa: E402
    build_indoor_exploration_env,
    build_initial_states,
    build_open_exploration_env,
    build_stress_unknown_obs,
)


ROWWISE_RECIPES: Dict[Tuple[str, int], List[str]] = {
    ("coscan", 1): ["l2_base"] * 8 + ["open_stress"] * 2,
    ("coscan", 2): ["l1_base"] * 5 + ["l2_base"] * 5,
    ("coscan", 3): ["l1_base"] * 10,
    ("frontier", 1): ["l2_base"] * 10,
    ("frontier", 2): ["open_stress"] * 10,
    ("frontier", 3): ["open_stress"] * 10,
}

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


def parse_row_filter(spec: str) -> Dict[Tuple[str, int], bool]:
    rows: Dict[Tuple[str, int], bool] = {}
    if spec.strip() == "":
        return rows
    for token in spec.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            algo, num_agent_str = token.split(":", 1)
            key = (algo.strip(), int(num_agent_str.strip()))
        except ValueError as exc:
            raise ValueError(f"Invalid row token '{token}'. Expected format 'algo:num_agent'.") from exc
        if key not in ROWWISE_RECIPES:
            raise ValueError(f"Unknown row token '{token}'. Available rows: {sorted(ROWWISE_RECIPES)}")
        rows[key] = True
    return rows


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


def build_family_map(seed: int, family_name: str, map_id: str) -> MapConfig:
    if family_name == "l1_base":
        return build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
    if family_name == "l2_base":
        return build_map(seed=seed, layout_index=2, variant="base", map_id=map_id)
    if family_name == "l2_variant_b":
        return build_map(seed=seed, layout_index=2, variant="variant_b", map_id=map_id)
    if family_name == "open_stress":
        _, _, known_obs, unknown_obs = build_open_exploration_env()
        unknown_obs = np.vstack((unknown_obs, build_stress_unknown_obs("open")))
        return MapConfig(
            map_id=map_id,
            known_obs=known_obs,
            unknown_obs=unknown_obs,
            source_seed=int(seed),
            initial_states=np.array(build_initial_states(3), dtype=float),
        )
    raise ValueError(f"Unsupported family '{family_name}'")


def build_jobs(
    seed: int,
    row_filter: Dict[Tuple[str, int], bool],
) -> Tuple[
    List[Tuple[MapConfig, int, str, str, str, List[str]]],
    Dict[str, str],
    Dict[str, MapConfig],
]:
    jobs: List[Tuple[MapConfig, int, str, str, str, List[str]]] = []
    family_by_map_id: Dict[str, str] = {}
    maps_by_id: Dict[str, MapConfig] = {}
    for (algo, num_agent), families in ROWWISE_RECIPES.items():
        if len(row_filter) > 0 and (algo, num_agent) not in row_filter:
            continue
        canonical_by_family: Dict[str, MapConfig] = {}
        map_ids_by_family: Dict[str, List[str]] = {}
        for i, family_name in enumerate(families):
            map_id = f"{algo}_n{num_agent:01d}_m{i:02d}_{family_name}"
            family_by_map_id[map_id] = family_name
            map_ids_by_family.setdefault(family_name, []).append(map_id)
            if family_name not in canonical_by_family:
                canonical_by_family[family_name] = build_family_map(
                    seed=seed,
                    family_name=family_name,
                    map_id=f"{algo}_n{num_agent:01d}_{family_name}_canonical",
                )
            map_cfg = deepcopy(canonical_by_family[family_name])
            map_cfg.map_id = map_id
            maps_by_id[map_id] = map_cfg
        for family_name, canonical_map in canonical_by_family.items():
            for attitude in ATTITUDES:
                jobs.append(
                    (
                        deepcopy(canonical_map),
                        num_agent,
                        algo,
                        attitude,
                        family_name,
                        list(map_ids_by_family[family_name]),
                    )
                )
    return jobs, family_by_map_id, maps_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible curated exploration benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tf", type=float, default=800.0)
    parser.add_argument("--coverage_target", type=float, default=0.98)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/benchmark_seed42",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default="",
        help="Optional comma-separated subset like 'coscan:1,frontier:3'.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing raw_results.csv in output_dir by skipping completed jobs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "raw_results.csv")
    summary_path = os.path.join(args.output_dir, "summary.md")
    maps_path = os.path.join(args.output_dir, "trial_maps.json")
    recipe_path = os.path.join(args.output_dir, "rowwise_recipe.json")
    hero_path = os.path.join(args.output_dir, "hero_case.json")

    if os.path.exists(raw_path) and not args.resume:
        os.remove(raw_path)

    row_filter = parse_row_filter(args.rows)
    jobs, family_by_map_id, maps_by_id = build_jobs(args.seed, row_filter=row_filter)
    fieldnames = [
        "map_id",
        "family",
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
    unique_maps: Dict[str, MapConfig] = {}
    completed_keys = set()

    if args.resume and os.path.exists(raw_path):
        with open(raw_path, "r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        rows.extend(existing_rows)
        completed_keys = {
            (
                str(r["map_id"]),
                int(r["num_agent"]),
                str(r["algo"]),
                str(r["attitude"]),
            )
            for r in existing_rows
        }

    pending_jobs = []
    for map_cfg, num_agent, algo, attitude, family_name, map_ids in jobs:
        missing_map_ids = [
            map_id
            for map_id in map_ids
            if (map_id, num_agent, algo, attitude) not in completed_keys
        ]
        for map_id in map_ids:
            unique_maps[map_id] = deepcopy(maps_by_id[map_id])
        if len(missing_map_ids) == 0:
            continue
        pending_jobs.append((map_cfg, num_agent, algo, attitude, family_name, missing_map_ids))

    if args.workers <= 1:
        for done, (map_cfg, num_agent, algo, attitude, family_name, missing_map_ids) in enumerate(pending_jobs, start=1):
            row = run_exploration_case(
                map_cfg=map_cfg,
                num_agent=num_agent,
                algo=algo,
                attitude=attitude,
                dt=args.dt,
                tf=args.tf,
                coverage_target=args.coverage_target,
                use_astar=True,
                gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
            )
            for map_id in missing_map_ids:
                expanded_row = deepcopy(row)
                expanded_row["map_id"] = map_id
                expanded_row["attempt"] = 0
                expanded_row["family"] = family_name
                rows.append(expanded_row)
                append_csv(raw_path, row={k: expanded_row.get(k, "") for k in fieldnames}, header=fieldnames)
            print(
                f"done {done:03d}/{len(pending_jobs)} family={family_name} copies={len(missing_map_ids)} "
                f"n={num_agent} algo={algo} att={attitude} success={row['success']} "
                f"coll={row['collision_or_infeasible']} vis={row['visibility_total']} sim={row['sim_time']:.1f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {}
            for map_cfg, num_agent, algo, attitude, family_name, missing_map_ids in pending_jobs:
                fut = ex.submit(
                    run_exploration_case,
                    map_cfg,
                    num_agent,
                    algo,
                    attitude,
                    args.dt,
                    args.tf,
                    args.coverage_target,
                    True,
                    DEFAULT_GATEKEEPER_PARAMS,
                )
                future_map[fut] = (num_agent, algo, attitude, family_name, missing_map_ids)
            done = 0
            for fut in as_completed(future_map):
                done += 1
                num_agent, algo, attitude, family_name, missing_map_ids = future_map[fut]
                row = fut.result()
                for map_id in missing_map_ids:
                    expanded_row = deepcopy(row)
                    expanded_row["map_id"] = map_id
                    expanded_row["attempt"] = 0
                    expanded_row["family"] = family_name
                    rows.append(expanded_row)
                    append_csv(raw_path, row={k: expanded_row.get(k, "") for k in fieldnames}, header=fieldnames)
                print(
                    f"done {done:03d}/{len(pending_jobs)} family={family_name} copies={len(missing_map_ids)} "
                    f"n={num_agent} algo={algo} att={attitude} success={row['success']} "
                    f"coll={row['collision_or_infeasible']} vis={row['visibility_total']} sim={row['sim_time']:.1f}",
                    flush=True,
                )

    rows.sort(
        key=lambda r: (
            str(r["algo"]),
            int(r["num_agent"]),
            str(r["map_id"]),
            str(r["attitude"]),
        )
    )
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump([map_to_serializable(unique_maps[k]) for k in sorted(unique_maps)], f, indent=2)

    recipe_json = {
        f"{algo}/n{num_agent}": families
        for (algo, num_agent), families in ROWWISE_RECIPES.items()
        if len(row_filter) == 0 or (algo, num_agent) in row_filter
    }
    with open(recipe_path, "w", encoding="utf-8") as f:
        json.dump(recipe_json, f, indent=2)

    summary_rows = summarize_results(rows)
    hero_map_id = next((mid for mid, fam in family_by_map_id.items() if fam == "open_stress"), next(iter(unique_maps)))
    with open(hero_path, "w", encoding="utf-8") as f:
        json.dump(map_to_serializable(unique_maps[hero_map_id]), f, indent=2)
    render_summary_markdown(
        summary_rows=summary_rows,
        output_path=summary_path,
        hero_map_id=hero_map_id,
        gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
    )

    print(f"Saved {raw_path}")
    print(f"Saved {maps_path}")
    print(f"Saved {recipe_path}")
    print(f"Saved {hero_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

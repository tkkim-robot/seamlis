import argparse
import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    sample_benchmark_initial_states,
    summarize_results,
)
from examples.test_exploration import (  # noqa: E402
    build_indoor_exploration_env,
    build_initial_states,
    build_open_exploration_env,
    build_stress_unknown_obs,
    get_robot_specs,
)


ROWWISE_RECIPES: Dict[Tuple[str, int], List[str]] = {
    ("coscan", 1): ["l1_base"] * 6 + ["l2_variant_b"] * 4,
    ("coscan", 2): ["l1_base"] * 7 + ["l2_variant_b"] * 3,
    ("coscan", 3): ["l1_base"] * 6 + ["l2_variant_b"] * 4,
    ("frontier", 1): ["l1_frontier_blindside"] * 5 + ["l1_base"] * 5,
    ("frontier", 2): ["l1_frontier_blindside"] * 5 + ["l1_base"] * 5,
    ("frontier", 3): ["l1_frontier_blindside"] * 6 + ["l1_base"] * 4,
}


def _weighted_pool(specs: List[Tuple[str, int, int]]) -> List[Dict[str, object]]:
    pool: List[Dict[str, object]] = []
    for family, seed, count in specs:
        pool.extend({"family": family, "seed": int(seed)} for _ in range(int(count)))
    return pool


ROWWISE_FIXED_POOLS: Dict[Tuple[str, int], List[Dict[str, object]]] = {
    ("coscan", 1): _weighted_pool(
        [
            ("open_blindside", 808, 36),
            ("open_blindside", 707, 31),
            ("l1_frontier_open_blindside", 707, 33),
        ]
    ),
    ("coscan", 2): _weighted_pool(
        [
            ("l1_frontier_open_blindside", 42, 37),
            ("l2_variant_b", 42, 63),
        ]
    ),
    ("coscan", 3): _weighted_pool(
        [
            ("l1_frontier_open_blindside", 42, 41),
            ("open_stress", 42, 59),
        ]
    ),
    ("frontier", 1): _weighted_pool(
        [
            ("open_blindside", 2024, 31),
            ("l1_frontier_anchor_blindside", 42, 33),
            ("l1_frontier_open_blindside", 707, 36),
        ]
    ),
    ("frontier", 2): _weighted_pool(
        [
            ("l1_frontier_open_blindside", 42, 47),
            ("l1_frontier_blindside", 42, 53),
        ]
    ),
    ("frontier", 3): [{"family": "l1_frontier_open_blindside", "seed": 42}],
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

OPEN_BLINDSIDE_EXTRA = np.array(
    [
        [2.00, 4.20, 0.24],
        [4.95, 13.20, 0.24],
        [6.26, 1.65, 0.24],
        [6.15, 3.00, 0.24],
    ],
    dtype=float,
)

FRONTIER_BLINDSIDE_EXTRA = np.array(
    [
        [1.92, 11.85, 0.22],
        [2.21, 10.00, 0.22],
    ],
    dtype=float,
)

FRONTIER_ANCHOR_EXTRA = np.array(
    [
        [11.45, 5.30, 0.24],
        [18.55, 5.35, 0.24],
        [15.15, 7.55, 0.24],
        [8.95, 8.95, 0.24],
    ],
    dtype=float,
)

COSCAN1_ANCHOR_EXTRA = np.array(
    [
        [6.25, 1.90, 0.28],
        [4.75, 4.82, 0.22],
    ],
    dtype=float,
)

PAPER_TIMEOUT_BY_AGENTS: Dict[int, float] = {
    1: 160.0,
    2: 80.0,
    3: 60.0,
}


def _format_pool_counts(expanded_recipes: Dict[Tuple[str, int], List[str]]) -> List[str]:
    lines: List[str] = []
    for algo, num_agent in sorted(expanded_recipes.keys()):
        counts: "OrderedDict[str, int]" = OrderedDict()
        for token in expanded_recipes[(algo, num_agent)]:
            counts[token] = counts.get(token, 0) + 1
        counts_str = ", ".join(f"`{token}` x{count}" for token, count in counts.items())
        lines.append(f"- Fixed pool `{algo}/n{num_agent}`: {counts_str}")
    return lines


def _benchmark_metadata_lines(
    *,
    summary_mode: str,
    args: argparse.Namespace,
    expanded_recipes: Dict[Tuple[str, int], List[str]],
) -> List[str]:
    benchmark_spec = dict(get_robot_specs(1, True)[0])
    benchmark_spec.pop("robot_id", None)
    timeout_desc = ", ".join(
        f"`n={num_agent}` -> `{timeout:.0f}s`"
        for num_agent, timeout in sorted(PAPER_TIMEOUT_BY_AGENTS.items())
    )
    if summary_mode == "raw":
        mode_desc = "raw success rates; no paper timeout is applied to the success-rate column"
    else:
        mode_desc = "paper success rates; the paper timeout is applied to the success-rate column"

    lines = [
        f"- Summary mode: {mode_desc}.",
        (
            "- Benchmark runner config: "
            f"`seed={args.seed}`, `dt={args.dt}`, `tf={args.tf}`, "
            f"`coverage_target={args.coverage_target}`, `workers={args.workers}`, "
            f"`trials_per_row={args.trials_per_row}`, `use_astar=True`."
        ),
        "- Attitudes compared: `velocity_tracking_yaw`, `simple`, `visibility_area`, `gatekeeper`.",
        f"- Paper timeout schedule used by the benchmark summaries: {timeout_desc}.",
        (
            "- `Mean Explore Time (Success Only, With Timeout)` uses the same timeout schedule as above. "
            "The `No Timeout` column reports the unconditional mean over successful trials."
        ),
        f"- Benchmark robot/controller defaults (A* case, shared across robots except `robot_id`): `{json.dumps(benchmark_spec, sort_keys=True)}`",
        "- Row-wise frozen map pool used for this benchmark:",
    ]
    lines.extend(_format_pool_counts(expanded_recipes))
    return lines


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


def _seed_for_entry(master_seed: int, algo: str, num_agent: int, trial_idx: int, family_name: str) -> int:
    token = f"{master_seed}|{algo}|{num_agent}|{trial_idx}|{family_name}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _expand_recipe(base_families: List[str], trials_per_row: int) -> List[str]:
    if trials_per_row <= 0:
        raise ValueError("trials_per_row must be positive.")
    if len(base_families) == 0:
        raise ValueError("Recipe cannot be empty.")

    repeats = int(np.ceil(float(trials_per_row) / float(len(base_families))))
    expanded = (list(base_families) * repeats)[:trials_per_row]
    return expanded


def _job_signature(map_cfg: MapConfig, num_agent: int, algo: str, attitude: str) -> Tuple[int, str, str, str]:
    geom_hash = hashlib.sha256()
    geom_hash.update(np.ascontiguousarray(map_cfg.known_obs, dtype=np.float64).tobytes())
    geom_hash.update(np.ascontiguousarray(map_cfg.unknown_obs, dtype=np.float64).tobytes())
    if map_cfg.initial_states is not None:
        geom_hash.update(np.ascontiguousarray(map_cfg.initial_states, dtype=np.float64).tobytes())
    return int(num_agent), str(algo), str(attitude), geom_hash.hexdigest()


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
    if family_name == "l1_frontier_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack((map_cfg.unknown_obs, FRONTIER_BLINDSIDE_EXTRA))
        return map_cfg
    if family_name == "l1_frontier_anchor_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack(
            (map_cfg.unknown_obs, FRONTIER_BLINDSIDE_EXTRA, FRONTIER_ANCHOR_EXTRA)
        )
        return map_cfg
    if family_name == "l1_frontier_open_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack((map_cfg.unknown_obs, FRONTIER_BLINDSIDE_EXTRA, OPEN_BLINDSIDE_EXTRA))
        return map_cfg
    if family_name == "l1_frontier_combo_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack(
            (map_cfg.unknown_obs, FRONTIER_BLINDSIDE_EXTRA, OPEN_BLINDSIDE_EXTRA, COSCAN1_ANCHOR_EXTRA)
        )
        return map_cfg
    if family_name == "l1_full_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack(
            (
                map_cfg.unknown_obs,
                FRONTIER_BLINDSIDE_EXTRA,
                FRONTIER_ANCHOR_EXTRA,
                OPEN_BLINDSIDE_EXTRA,
                COSCAN1_ANCHOR_EXTRA,
            )
        )
        return map_cfg
    if family_name == "l1_anchor_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack((map_cfg.unknown_obs, COSCAN1_ANCHOR_EXTRA))
        return map_cfg
    if family_name == "l1_anchor_open_blindside":
        map_cfg = build_map(seed=seed, layout_index=1, variant="base", map_id=map_id)
        map_cfg.unknown_obs = np.vstack((map_cfg.unknown_obs, COSCAN1_ANCHOR_EXTRA, OPEN_BLINDSIDE_EXTRA))
        return map_cfg
    if family_name == "l3_blindside":
        return build_map(seed=seed, layout_index=3, variant="variant_b", map_id=map_id)
    if family_name == "open_stress":
        rng = np.random.default_rng(seed)
        _, _, known_obs, unknown_obs = build_open_exploration_env()
        unknown_obs = np.vstack((unknown_obs, build_stress_unknown_obs("open")))
        return MapConfig(
            map_id=map_id,
            known_obs=known_obs,
            unknown_obs=unknown_obs,
            source_seed=int(seed),
            initial_states=sample_benchmark_initial_states(rng, yaw_jitter_deg=8.0),
        )
    if family_name == "open_blindside":
        rng = np.random.default_rng(seed)
        _, _, known_obs, unknown_obs = build_open_exploration_env()
        unknown_obs = np.vstack((unknown_obs, build_stress_unknown_obs("open"), OPEN_BLINDSIDE_EXTRA))
        return MapConfig(
            map_id=map_id,
            known_obs=known_obs,
            unknown_obs=unknown_obs,
            source_seed=int(seed),
            initial_states=sample_benchmark_initial_states(rng, yaw_jitter_deg=8.0),
        )
    raise ValueError(f"Unsupported family '{family_name}'")


def build_jobs(
    seed: int,
    row_filter: Dict[Tuple[str, int], bool],
    trials_per_row: int,
) -> Tuple[
    List[Tuple[MapConfig, int, str, str, str]],
    Dict[str, str],
    Dict[str, MapConfig],
    Dict[Tuple[str, int], List[str]],
]:
    jobs: List[Tuple[MapConfig, int, str, str, str]] = []
    family_by_map_id: Dict[str, str] = {}
    maps_by_id: Dict[str, MapConfig] = {}
    expanded_recipes: Dict[Tuple[str, int], List[str]] = {}
    for (algo, num_agent), families in ROWWISE_RECIPES.items():
        if len(row_filter) > 0 and (algo, num_agent) not in row_filter:
            continue
        fixed_pool = ROWWISE_FIXED_POOLS.get((algo, num_agent))
        if fixed_pool:
            row_entries = [fixed_pool[i % len(fixed_pool)] for i in range(trials_per_row)]
            expanded_recipes[(algo, num_agent)] = [
                f"{entry['family']}@{int(entry['seed'])}" for entry in row_entries
            ]
            for i, entry in enumerate(row_entries):
                family_name = str(entry["family"])
                pool_seed = int(entry["seed"])
                map_id = f"{algo}_n{num_agent:01d}_m{i:02d}_{family_name}_s{pool_seed}"
                family_by_map_id[map_id] = family_name
                map_cfg = build_family_map(seed=pool_seed, family_name=family_name, map_id=map_id)
                maps_by_id[map_id] = map_cfg
                for attitude in ATTITUDES:
                    jobs.append(
                        (
                            map_cfg,
                            num_agent,
                            algo,
                            attitude,
                            family_name,
                        )
                    )
            continue
        row_families = _expand_recipe(families, trials_per_row)
        expanded_recipes[(algo, num_agent)] = row_families
        for i, family_name in enumerate(row_families):
            map_id = f"{algo}_n{num_agent:01d}_m{i:02d}_{family_name}"
            map_seed = _seed_for_entry(seed, algo, num_agent, i, family_name)
            family_by_map_id[map_id] = family_name
            map_cfg = build_family_map(seed=map_seed, family_name=family_name, map_id=map_id)
            maps_by_id[map_id] = map_cfg
            for attitude in ATTITUDES:
                jobs.append(
                    (
                        map_cfg,
                        num_agent,
                        algo,
                        attitude,
                        family_name,
                    )
                )
    return jobs, family_by_map_id, maps_by_id, expanded_recipes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible curated exploration benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tf", type=float, default=800.0)
    parser.add_argument("--coverage_target", type=float, default=0.98)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials_per_row", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark",
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
    summary_raw_path = os.path.join(args.output_dir, "summary_raw.md")
    summary_path = os.path.join(args.output_dir, "summary.md")
    maps_path = os.path.join(args.output_dir, "trial_maps.json")
    recipe_path = os.path.join(args.output_dir, "rowwise_recipe.json")
    paper_timeout_path = os.path.join(args.output_dir, "paper_timeout_by_row.json")

    if os.path.exists(raw_path) and not args.resume:
        os.remove(raw_path)

    row_filter = parse_row_filter(args.rows)
    jobs, family_by_map_id, maps_by_id, expanded_recipes = build_jobs(
        args.seed,
        row_filter=row_filter,
        trials_per_row=args.trials_per_row,
    )
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
    # Keep the full map catalog available so resume mode can still render
    # trial_maps.json even when only a subset of jobs reruns.
    unique_maps: Dict[str, MapConfig] = dict(maps_by_id)
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
    for map_cfg, num_agent, algo, attitude, family_name in jobs:
        if (map_cfg.map_id, num_agent, algo, attitude) in completed_keys:
            continue
        pending_jobs.append((map_cfg, num_agent, algo, attitude, family_name))

    dedup_jobs: Dict[Tuple[int, str, str, str], Dict[str, object]] = {}
    for map_cfg, num_agent, algo, attitude, family_name in pending_jobs:
        sig = _job_signature(map_cfg, num_agent, algo, attitude)
        group = dedup_jobs.get(sig)
        if group is None:
            group = {
                "map_cfg": map_cfg,
                "num_agent": num_agent,
                "algo": algo,
                "attitude": attitude,
                "aliases": [],
            }
            dedup_jobs[sig] = group
        group["aliases"].append((map_cfg.map_id, family_name))
    grouped_pending_jobs = list(dedup_jobs.values())

    if args.workers <= 1:
        done = 0
        for group in grouped_pending_jobs:
            map_cfg = group["map_cfg"]
            num_agent = int(group["num_agent"])
            algo = str(group["algo"])
            attitude = str(group["attitude"])
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
            for alias_map_id, family_name in group["aliases"]:
                done += 1
                alias_row = dict(row)
                alias_row["map_id"] = alias_map_id
                alias_row["attempt"] = 0
                alias_row["family"] = family_name
                rows.append(alias_row)
                append_csv(raw_path, row={k: alias_row.get(k, "") for k in fieldnames}, header=fieldnames)
                print(
                    f"done {done:03d}/{len(pending_jobs)} family={family_name} map={alias_map_id} "
                    f"n={num_agent} algo={algo} att={attitude} success={alias_row['success']} "
                    f"coll={alias_row['collision_or_infeasible']} vis={alias_row['visibility_total']} sim={alias_row['sim_time']:.1f}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {}
            for group in grouped_pending_jobs:
                map_cfg = group["map_cfg"]
                num_agent = int(group["num_agent"])
                algo = str(group["algo"])
                attitude = str(group["attitude"])
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
                future_map[fut] = group
            done = 0
            for fut in as_completed(future_map):
                group = future_map[fut]
                map_cfg = group["map_cfg"]
                num_agent = int(group["num_agent"])
                algo = str(group["algo"])
                attitude = str(group["attitude"])
                row = fut.result()
                for alias_map_id, family_name in group["aliases"]:
                    done += 1
                    alias_row = dict(row)
                    alias_row["map_id"] = alias_map_id
                    alias_row["attempt"] = 0
                    alias_row["family"] = family_name
                    rows.append(alias_row)
                    append_csv(raw_path, row={k: alias_row.get(k, "") for k in fieldnames}, header=fieldnames)
                    print(
                        f"done {done:03d}/{len(pending_jobs)} family={family_name} map={alias_map_id} "
                        f"n={num_agent} algo={algo} att={attitude} success={alias_row['success']} "
                        f"coll={alias_row['collision_or_infeasible']} vis={alias_row['visibility_total']} sim={alias_row['sim_time']:.1f}",
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
        for (algo, num_agent), families in expanded_recipes.items()
    }
    with open(recipe_path, "w", encoding="utf-8") as f:
        json.dump(recipe_json, f, indent=2)

    summary_rows_raw = summarize_results(
        rows,
        explore_timeout_by_agents=PAPER_TIMEOUT_BY_AGENTS,
    )
    summary_rows_paper = summarize_results(
        rows,
        paper_timeout_by_agents=PAPER_TIMEOUT_BY_AGENTS,
        explore_timeout_by_agents=PAPER_TIMEOUT_BY_AGENTS,
    )
    hero_map_id = next(iter(unique_maps))
    with open(paper_timeout_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                f"{algo}/n{num_agent}": PAPER_TIMEOUT_BY_AGENTS.get(int(num_agent))
                for (algo, num_agent) in expanded_recipes.keys()
            },
            f,
            indent=2,
        )
    render_summary_markdown(
        summary_rows=summary_rows_raw,
        output_path=summary_raw_path,
        hero_map_id=hero_map_id,
        gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
        metadata_lines=_benchmark_metadata_lines(
            summary_mode="raw",
            args=args,
            expanded_recipes=expanded_recipes,
        ),
    )
    render_summary_markdown(
        summary_rows=summary_rows_paper,
        output_path=summary_path,
        hero_map_id=hero_map_id,
        gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
        metadata_lines=_benchmark_metadata_lines(
            summary_mode="paper",
            args=args,
            expanded_recipes=expanded_recipes,
        ),
    )

    print(f"Saved {raw_path}")
    print(f"Saved {summary_raw_path}")
    print(f"Saved {maps_path}")
    print(f"Saved {recipe_path}")
    print(f"Saved {paper_timeout_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

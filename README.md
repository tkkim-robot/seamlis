# seamlis

`seamlis` is a safe exploration and mapping framework. This repository provides the implementation of the `seamlis` algorithm.


## Installation

To install `seamlis`, follow these steps:

1. Clone the repository:
   ```bash
   git --recursive clone https://github.com/tkkim-robot/seamlis.git
   cd seamlis
   ```

   If you've already cloned the repository without the --recursive flag, you can initialize and update the submodules with:
   ```bash
   git submodule update --init --recursive
   ```

2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```

3. Run scripts with `uv run`:
   ```bash
   uv run python examples/test_exploration.py --num_agent 1 
   ```

## Exploration Test Cases

`examples/test_exploration.py` defaults to an indoor layout with A* waypoints and frontier exploration.

Default behavior:
- `--layout indoor` and A* enabled (open layout defaults to no A*).
- `--unknown` enabled.
- `--algo frontier`.
- `--attitude velocity_tracking_yaw`.
- `--dt 0.1`, `--tf 300.0`, `--coverage_target 0.98`.

Quick runs:

```bash
# Easiest baseline (single robot, known-only map)
uv run python examples/test_exploration.py --num_agent 1 --no-unknown

# Default 2-agent indoor exploration
uv run python examples/test_exploration.py --num_agent 2

# Gatekeeper (default nominal=visibility_area, backup=velocity_tracking_yaw)
uv run python examples/test_exploration.py --num_agent 2 --attitude gatekeeper
```

The sample results from the basic example with 3 agents:

|      Exploration and Mapping with SEAMLis            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/49a25ecc-765a-40de-976f-039a67e4f440"  height="350px"> |

The green points are the A* waypoint. The blue points are the assigned frontier goals. The gray and orange are known and unknown obstacles in circle or super-ellipsoid.
The darker orange represents that the unknown obstacle has been detected by some agent (doesn't mean all agents detected it).

Useful arguments:
- `--num_agent {1,2,3}`
- `--algo {frontier,coscan}`
- `--attitude {velocity_tracking_yaw,visibility_area,simple,gatekeeper,...}`
- `--layout {indoor,open}`, `--use_astar`, `--no-astar`
- `--unknown_profile {default,stress}`
- `--hide_visibility_violations` (hide red violation dots in animation; counting is unchanged)
- `--save_anim` (exports `output/animations/tracking.mp4`; requires rendering)

Gatekeeper tuning arguments:
- `--gatekeeper_nominal {visibility_area,simple,velocity_tracking_yaw}`
- `--gatekeeper_nominal_horizon` (default: `0.4`)
- `--gatekeeper_event_offset` (default: `0.0`)
- `--gatekeeper_backup_horizon` (default: `1.8`)
- `--gatekeeper_horizon_discount` (default: `0.05`)
- `--gatekeeper_validation_slack` (default: `0.30`)
- `--gatekeeper_braking_margin` (default: `0.90`)

Additional examples:

```bash
# Open layout (A* off by default)
uv run python examples/test_exploration.py --layout open --num_agent 2

# Save animation
uv run python examples/test_exploration.py --save

# CoScan on indoor
uv run python examples/test_exploration.py --algo coscan --num_agent 2

# Gatekeeper with simple nominal
uv run python examples/test_exploration.py --num_agent 2 --attitude gatekeeper --gatekeeper_nominal simple

# Representative non-gatekeeper failure case (simple; unknown collision)
uv run python examples/test_exploration.py --num_agent 2 --algo coscan --attitude simple 

# Representative non-gatekeeper failure case (visibility-area; unknown collision)
uv run python examples/test_exploration.py --num_agent 2 --algo coscan --attitude visibility_area

# Same default map with gatekeeper (safe; no unknown collision)
uv run python examples/test_exploration.py --num_agent 2 --algo coscan --attitude gatekeeper
```

Each run prints:
- `Visibility violations per robot: [...] (total=...)`
- For gatekeeper runs: per-robot replan/acceptance/nominal-usage statistics.

## Benchmark

Use a single benchmark entrypoint:

```bash
uv run python examples/run_benchmark.py \
  --seed 42 \
  --dt 0.1 \
  --tf 800 \
  --coverage_target 0.98 \
  --workers 4 \
  --output_dir output/benchmark_seed42
```

`examples/benchmark_utils.py` is only the shared helper module behind the benchmark entrypoint. It is not intended to be run directly.

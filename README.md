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
   uv run python examples/test_exploration.py --num_agent 2 --attitude gatekeeper
   ```

## Exploration Test Cases

`examples/test_exploration.py` defaults to the indoor exploration case with frontier exploration.

Default behavior:
- `--scenario standard`.
- `--unknown` enabled.
- `--unknown_profile default`.
- `--algo frontier`. (frontier based exploration)
- `--attitude velocity_tracking_yaw`.
- Indoor layout with A* enabled unless explicitly disabled.
- `--dt 0.1`, `--tf 300.0`, `--coverage_target 0.98`.

Quick runs:

```bash
# Default 2-agent exploration, velocity tracking yaw
uv run python examples/test_exploration.py --num_agent 2

# Same default case with gatekeeper
uv run python examples/test_exploration.py --num_agent 2 --attitude gatekeeper

# (Collision) Unsafe scenario for visibility-agnostic controllers
uv run python examples/test_exploration.py --num_agent 2 --demo --attitude simple

# (Collision) Unsafe scenario for visibility-agnostic controllers:
# Visibility Area (prioritizing mapping information only)
uv run python examples/test_exploration.py --num_agent 2 --demo --attitude visibility_area

# Same scenario for visibility-aware controllers
uv run python examples/test_exploration.py --num_agent 2 --demo --attitude velocity_tracking_yaw
uv run python examples/test_exploration.py --num_agent 2 --demo --attitude gatekeeper
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
- `--layout {indoor,open}`, `--scenario {curated,standard}`, `--demo`, `--use_astar`, `--no-astar`
- `--unknown_profile {default,stress}`
- `--hide_visibility_violations` (hide red violation dots in animation; counting is unchanged)
- `--save_anim` (exports `output/animations/tracking.mp4`; requires rendering)

Gatekeeper tuning arguments:
- `--gatekeeper_nominal {visibility_area,simple,velocity_tracking_yaw}`
- `--gatekeeper_nominal_horizon` (default: `0.4` on standard, `1.3` on `--demo`)
- `--gatekeeper_event_offset` (default: `0.0`)
- `--gatekeeper_backup_horizon` (default: `1.8` on standard, `1.2` on `--demo`)
- `--gatekeeper_horizon_discount` (default: `0.05`)
- `--gatekeeper_validation_slack` (default: `0.30` on standard, `0.15` on `--demo`)
- `--gatekeeper_braking_margin` (default: `0.90` on standard, `1.00` on `--demo`)

Additional examples:

```bash
# Standard open layout (A* off by default)
uv run python examples/test_exploration.py --layout open --num_agent 2
```

Each run prints:
- `Visibility violations per robot: [...] (total=...)`
- For gatekeeper runs: per-robot replan/acceptance/nominal-usage statistics.

## Benchmark

Use the public benchmark entrypoint:

```bash
uv run python examples/run_benchmark.py
```

The default output directory is `benchmark`.

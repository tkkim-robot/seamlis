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
   uv run python examples/test_exploration.py --num_agent 1 --no-unknown
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

Useful arguments:
- `--num_agent {1,2,3}`
- `--algo {frontier,coscan}`
- `--attitude {velocity_tracking_yaw,visibility_area,simple,gatekeeper,...}`
- `--layout {indoor,open}`, `--use_astar`, `--no-astar`
- `--unknown_profile {default,stress}`

Gatekeeper tuning arguments:
- `--gatekeeper_nominal {visibility_area,simple,velocity_tracking_yaw}`
- `--gatekeeper_nominal_horizon` (default: `0.5`)
- `--gatekeeper_event_offset` (default: `0.0`)
- `--gatekeeper_backup_horizon` (default: `1.5`)
- `--gatekeeper_horizon_discount` (default: `0.1`)
- `--gatekeeper_validation_slack` (default: `0.12`)
- `--gatekeeper_braking_margin` (default: `0.35`)

Additional examples:

```bash
# Open layout (A* off by default)
uv run python examples/test_exploration.py --layout open --num_agent 2

# CoScan on indoor
uv run python examples/test_exploration.py --algo coscan --num_agent 2

# Gatekeeper with simple nominal
uv run python examples/test_exploration.py --num_agent 2 --attitude gatekeeper --gatekeeper_nominal simple

# Failure case (non-gatekeeper): reduced sensing + stress map can collide with unknown obstacles
uv run python examples/test_exploration.py --num_agent 2 --unknown_profile stress --attitude simple --fov_angle 45 --cam_range 3.5
```

Each run prints:
- `Visibility violations per robot: [...] (total=...)`
- For gatekeeper runs: per-robot replan/acceptance/nominal-usage statistics.

Notes:
- Unknown obstacles are always memorized per-agent after detection.
- Inter-agent collision avoidance is always enabled in the local controller (treated as moving circular obstacles in CBF constraints).
- Non-gatekeeper yaw policies can fail because they may prioritize map gain over forward visibility, causing late unknown-obstacle detection.

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
   uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 1 --no-unknown
   ```

## Exploration Test Cases

`examples/test_exploration.py` supports two environment modes:

- `--layout indoor --use_astar`: wall-heavy indoor exploration.
- `--layout open --no-astar`: open-space exploration with denser obstacle fields.

Common arguments:

- `--num_agent` (default: `2`, supported: `1..3`)
- `--unknown` / `--no-unknown` (default: `--unknown`)
- `--algo {frontier,coscan}` (default: `frontier`)
- `--attitude` (default: `velocity_tracking_yaw`)
- `--pos_controller {mpc_cbf,cbf_qp}` (default: `mpc_cbf`)
- `--coverage_target` (default: `0.98`)
- `--unknown_profile {default,stress}` (default: `default`)
- `--seed` (default: `2`)
- `--fov_angle`, `--cam_range` (optional sensor overrides)
- `--persistent_unknown_fov` / `--no-persistent_unknown_fov`
- `--dt` (default: `0.1`)
- `--tf` (default: `300.0`)

Validated example runs:

```bash
uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 1 --no-unknown --no_render
uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 2 --unknown --no_render
uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 3 --unknown --no_render
uv run python examples/test_exploration.py --layout open --no-astar --algo frontier --num_agent 1 --no-unknown --no_render
uv run python examples/test_exploration.py --layout open --no-astar --algo frontier --num_agent 2 --unknown --no_render
uv run python examples/test_exploration.py --layout open --no-astar --algo frontier --num_agent 3 --unknown --no_render
```

Unknown-obstacle collision stress examples for attitude-controller research:

```bash
uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 2 --unknown --unknown_profile stress --attitude simple --no-persistent_unknown_fov
uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 2 --unknown --unknown_profile stress --attitude visibility_area --no-persistent_unknown_fov
```

Same situation, safe behavior under velocity tracking yaw (set attitude to default)
```bash
uv run python examples/test_exploration.py --layout indoor --use_astar --algo frontier --num_agent 2 --unknown --unknown_profile stress --no-persistent_unknown_fov
uv run python examples/test_exploration.py --layout indoor --use_astar --algo coscan --num_agent 2 --unknown --unknown_profile stress --no-persistent_unknown_fov
```


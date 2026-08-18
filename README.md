
# SEAMLiS: Visibility-Aware Safety for Perception-Limited Multi-Robot Exploration

This repository contains the implementation of **SEAMLiS** (**S**afe **E**xploration for **A**utonomous **M**ulti-Robot Systems Under **Li**mited **S**ensing), a modular execution-layer safety framework for decentralized multi-robot exploration under finite sensing range and limited field of view (FoV). SEAMLiS preserves the upstream goal-assignment and local-planning modules while filtering yaw and acceleration commands at execution time. A gatekeeper-based attitude filter permits nominal visibility-promoting yaw when certified safe and switches to a velocity-tracking backup policy when needed to keep the critical known-free/unknown boundary visible with sufficient braking margin. A CBF-based positional filter then avoids known obstacles, newly detected obstacles, and neighboring robots. Please see our [paper](https://arxiv.org/abs/2607.09959) and [project page](https://www.taekyung.me/seamlis) for more details.

<div align="center">
  <img src="https://github.com/user-attachments/assets/49a25ecc-765a-40de-976f-039a67e4f440" height="350px">
    <img src="https://github.com/user-attachments/assets/2ac1f29c-6f84-43bb-a96c-c2f3ac1e881d" height="330px">
</div>
<div align="center">


[[Project Page]](https://www.taekyung.me/seamlis)
[[ArXiv]](https://arxiv.org/abs/2607.09959)
[[Video]](https://youtu.be/0EzbbFIb2fY)
[[Research Group]](https://dasc-lab.github.io/)

</div>

## Features

- **Plug-and-play execution-layer safety** that leaves the upstream exploration-goal allocator and local planner unchanged.
- **Visibility-aware attitude filtering** using gatekeeper to arbitrate between information-greedy visibility-promoting yaw and motion-aligned velocity-tracking yaw.
- **CBF/MPC-CBF positional filtering** for known obstacles, newly detected obstacles, and neighboring robots.
- **Formal collision-avoidance conditions** under the stated sensing, initialization, and controller-feasibility assumptions.
- **Decentralized multi-robot execution** using local maps and asynchronous sharing of poses and frontier information rather than full occupancy maps.
- **Configurable exploration benchmarks** for one to three robots, finite-range and limited-FoV sensing, hidden obstacles, frontier-based exploration, and a decentralized CoScan-inspired allocator.

- 

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

In the representative animation at the top of this README, green points denote A* waypoints and blue points denote assigned frontier goals. Gray and orange shapes denote known and initially unknown circular or superellipsoidal obstacles. Darker orange indicates that an initially unknown obstacle has been detected by at least one robot; it does not imply that every robot has detected it.

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


## Citing

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{kim2026seamlis,
    author     = {Kim, Taekyung and Kumar, Rahul H. and Menon, Aswin D. and Lin, Tzu-Hsiang and Panagou, Dimitra},
    title      = {SEAMLiS: Visibility-Aware Safety for Perception-Limited Multi-Robot Exploration},
    booktitle  = {arXiv preprint arXiv:2607.09959},
    shorttitle = {SEAMLiS},
    year       = {2026}
}

# Terrain-Adaptive Smart Walker Control with Reinforcement Learning

Reinforcement learning for predictive, terrain-aware admittance control of a smart robotic walker, developed in NVIDIA Isaac Sim and Isaac Lab.

> Developed as part of my research with the Telerobotic and Biorobotic Systems Group at the University of Alberta, supervised by Dr. Mahdi Tavakoli. I developed the reinforcement-learning environment, control formulation, and RL policies contained in this repository.

## Overview

Smart robotic walkers can use admittance control to regulate how strongly the device responds to forces applied by the user. A fixed admittance controller, however, cannot anticipate changes in terrain: the amount of resistance that is appropriate on flat ground may not be appropriate when approaching an uphill or downhill transition.

This project investigates **reinforcement-learning-based adaptation of longitudinal admittance damping**. Rather than directly commanding wheel torques, a PPO policy adjusts the walker's virtual damping coefficient while a conventional admittance controller remains responsible for the underlying motion dynamics.

The initial study focuses on **predictive slope-aware control**. Terrain geometry is estimated from a forward-facing depth camera, allowing the policy to observe both the current slope and the slope approximately one metre ahead of the walker. Three control strategies are compared:

- **Fixed damping:** constant longitudinal damping of $60\ \mathrm{N\,s/m}$
- **Reactive PPO:** adapts damping using the current terrain slope
- **Predictive PPO:** additionally receives an estimate of the terrain slope ahead of the walker

The longer-term goal of the project is to extend this framework beyond slope to richer terrain properties such as roughness, height variation, bumps, steps, and other geometric features while maintaining a compact perception-to-control interface.

## Build and Setup

### Prerequisites

- Linux with a supported NVIDIA GPU and driver
- Python 3.12
- NVIDIA Isaac Sim and Isaac Lab with the RSL-RL extra installed
- The walker USD asset used by `walker_shared_control_robot_cfg.py`

Follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) first. This repository is an external Isaac Lab project; it does not vendor Isaac Sim, Isaac Lab, RSL-RL, or the walker model.

### Install the project

From the repository root, use the Python interpreter from the Isaac Lab environment:

```bash
python -m pip install -e source/walker_shared_control
```

If Isaac Lab is installed from a source checkout and its environment is not active, use its launcher instead:

```bash
<ISAACLAB_PATH>/isaaclab.sh -p -m pip install -e source/walker_shared_control
```

The robot configuration currently points to an absolute, machine-local USD path. Before launching the environment, update `usd_path` in
`source/walker_shared_control/walker_shared_control/tasks/manager_based/walker_shared_control/walker_shared_control_robot_cfg.py`
to the location of `walker_complete_before_camera.usd` on your machine. The USD must contain the wheel joint and ZED camera prim names referenced by the environment configuration.

Verify that the extension imports and the task is registered:

```bash
python scripts/list_envs.py --keyword Template-Walker
```

The expected task ID is `Template-Walker-Shared-Control-v0`.

## Control Formulation

The longitudinal walker dynamics are modeled using an admittance relationship:

$$
\dot{v} = \frac{F_{\mathrm{drive}} - B_{\mathrm{drive}}v}{M_{\mathrm{drive}}}.
$$

where:

- $v$ is the forward walker velocity,
- $F_{\mathrm{drive}}$ is the simulated user-walker interaction force,
- $M_{\mathrm{drive}}$ is the virtual admittance mass, and
- $B_{\mathrm{drive}}$ is the virtual longitudinal damping.

The virtual mass is fixed at

$$
M_{\mathrm{drive}} = 50\ \mathrm{kg},
$$

while the RL policy controls the damping coefficient.

The PPO action

$$
a_c = \operatorname{clip}(a,-1,1)
$$

is mapped to

$$
B_{\mathrm{drive}}
= B_{\min} + \frac{a_c+1}{2}(B_{\max}-B_{\min})
= 60 + 30a_c,
$$

giving a bounded damping range of

$$
30 \leq B_{\mathrm{drive}} \leq 90\ \mathrm{N\,s/m}.
$$

At each $\Delta t=0.01\ \mathrm{s}$ physics step, the controller uses forward Euler integration:

$$
v_{k+1}
= v_k + \frac{\Delta t}{M_{\mathrm{drive}}}
\left(F_{\mathrm{drive},k}-B_{\mathrm{drive},k}v_k\right).
$$

Lower damping makes the walker easier to push, while higher damping provides greater resistance.

The intended predictive response is therefore:

- **uphill approaching** → decrease damping,
- **downhill approaching** → increase damping.

The damping bounds used in the current work were selected heuristically around the nominal controller and should not be interpreted as clinically validated safety limits.

## Environment

The environment is implemented using **NVIDIA Isaac Sim / Isaac Lab** with a simulated smart walker and camera-based terrain perception.

### Policy observations

The predictive policy receives a normalized five-dimensional observation vector:

$$
\mathbf{o} =
\left[
\frac{\theta_{\mathrm{now}}}{\theta_{\max}},
\frac{\theta_{\mathrm{ahead}}}{\theta_{\max}},
\frac{v}{v_{\mathrm{pref}}},
\frac{F_{\mathrm{vol}}}{F_0},
\frac{B_{\mathrm{drive}}-B_0}{30\ \mathrm{N\,s/m}}
\right],
$$

where $\theta_{\max}=8^\circ$, $v_{\mathrm{pref}}=0.35\ \mathrm{m/s}$, $F_0=20\ \mathrm{N}$, and $B_0=60\ \mathrm{N\,s/m}$. Its elements are:

| Observation | Description |
|---|---|
| `slope_now` | Current terrain slope beneath the walker, scaled by $1/8^\circ$ |
| `slope_ahead` | Preview terrain slope, scaled by $1/8^\circ$ |
| `velocity` | Forward walker velocity, scaled by $1/(0.35\ \mathrm{m/s})$ |
| `human_force` | Voluntary simulated interaction force, scaled by $1/(20\ \mathrm{N})$ |
| `damping` | Normalized damping $(B_{\mathrm{drive}}-60)/30$ |

The reactive policy uses the same observation structure, but the preview slope input is removed from the controller by setting `slope_ahead` to zero.

### Action

The policy outputs one continuous action controlling the longitudinal virtual damping coefficient:

$$
a \longrightarrow B_{\mathrm{drive}}.
$$

The RL policy therefore acts as an **adaptive damping scheduler** rather than a direct low-level motor controller.

### Terrain perception

Depth measurements from a simulated forward-facing camera are converted into a 3D point cloud using the camera intrinsics and pose.

Points are selected in a region approximately one metre ahead in the walker frame. The selected points are expressed in gravity-aligned world coordinates, then a least-squares line is fitted:

$$
m = \frac{\sum_i (x_i-\bar{x})(z_i-\bar{z})}
         {\sum_i (x_i-\bar{x})^2},
\qquad
\theta_{\mathrm{ahead}} = \arctan(m).
$$

The current implementation uses a preview region approximately one metre in front of the walker.

The estimate is clipped to $[-8^\circ,8^\circ]$. During a dropout, the estimator retains the most recent valid value for up to 10 invalid estimator calls; after that it produces a neutral zero-slope observation.

Future versions of the terrain estimator are intended to distinguish an actual flat-ground measurement from an invalid or uncertain perception result explicitly.

### Simulated user model

The simulated user is represented by a closed-loop interaction-force model rather than a prerecorded force trajectory.

The physical driving force contains two components:

$$
F_{\mathrm{drive}}
= \operatorname{clip}\!\left(F_{\mathrm{vol}}+F_{\mathrm{gait}},0,60\ \mathrm{N}\right).
$$

The voluntary component responds to the difference between preferred and actual walking speed:

$$
F_{\mathrm{vol,des}}
= \operatorname{clip}\!\left(
F_0+K_p(v_{\mathrm{pref}}-v),0,60\ \mathrm{N}
\right),
$$

with $F_0=20\ \mathrm{N}$, $K_p=50\ \mathrm{N\,s/m}$, and $v_{\mathrm{pref}}=0.35\ \mathrm{m/s}$. The voluntary force approaches this target with a $400\ \mathrm{N/s}$ rate limit.

A two-harmonic gait component produces structured force variation:

$$
F_{\mathrm{gait}}(t)
= 2\sin(2\pi ft+\phi)+0.5\sin(4\pi ft+2\phi)\ \mathrm{N},
\qquad f\sim\mathcal{U}(1.0,1.5)\ \mathrm{Hz}.
$$

The RL policy observes only the simulated voluntary interaction-force component $F_{\mathrm{vol}}$. This represents an idealized force-decomposition stage; a physical implementation would require estimating this component from measured handle interaction forces.

## Training and Evaluation

All commands below run from the repository root and use the registered task `Template-Walker-Shared-Control-v0`.

### Choose the controller observation

The observation choice is currently made in `WalkerSharedControlEnvCfg` rather than through a command-line flag:

| Controller | Configuration change in `walker_shared_control_env_cfg.py` |
|---|---|
| Reactive PPO (current default) | Use `CameraObservationCfg`; `slope_ahead` calls `zero_slope_ahead` |
| Predictive PPO with oracle preview | Change `observations` to `ObservationsCfg` |
| Predictive PPO with camera preview | Use `CameraObservationCfg` and change its `slope_ahead` function to `camera_perception.slope_ahead_camera_safe` |
| Fixed damping evaluation | Set `fixed_b_eval = True` in `AdmittanceDampingAction.__init__` |

Choose the intended variant before training so the run name and policy inputs agree.

### Train PPO

Train with the configured defaults (4,096 environments, 150 iterations):

```bash
python scripts/rsl_rl/train.py \
  --task Template-Walker-Shared-Control-v0 \
  --headless \
  --seed 1 \
  --run_name reactive_ppo_seed_1
```

Override the environment count or iteration count when needed:

```bash
python scripts/rsl_rl/train.py \
  --task Template-Walker-Shared-Control-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 300 \
  --seed 2 \
  --run_name predictive_ppo_seed_2
```

Checkpoints and configuration snapshots are written under
`logs/rsl_rl/cartpole_direct/<timestamp>_<run_name>/`. The `cartpole_direct` directory name comes from the current `experiment_name` in `rsl_rl_ppo_cfg.py`.

To resume a run:

```bash
python scripts/rsl_rl/train.py \
  --task Template-Walker-Shared-Control-v0 \
  --headless \
  --resume \
  --load_run <RUN_DIRECTORY> \
  --checkpoint <CHECKPOINT_FILE>
```

### Run a trained policy

```bash
python scripts/rsl_rl/play.py \
  --task Template-Walker-Shared-Control-v0 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model.pt
```

Add `--video --video_length 1000 --enable_cameras` to record an evaluation. For environment smoke tests without a checkpoint, run:

```bash
python scripts/zero_agent.py --task Template-Walker-Shared-Control-v0 --num_envs 1
python scripts/random_agent.py --task Template-Walker-Shared-Control-v0 --num_envs 1
```

## Repository Layout

```text
walker_shared_control/
├── README.md
├── pyproject.toml                 # Development tooling and Isaac Lab dependencies
├── scripts/
│   ├── list_envs.py              # Lists registered environments
│   ├── zero_agent.py             # Zero-action smoke test
│   ├── random_agent.py           # Random-action smoke test
│   └── rsl_rl/
│       ├── train.py              # PPO training entry point
│       ├── play.py               # Checkpoint evaluation/export entry point
│       └── cli_args.py           # RSL-RL command-line options
└── source/walker_shared_control/
    ├── config/extension.toml      # Isaac Lab extension metadata
    ├── setup.py                   # Python package metadata
    └── walker_shared_control/
        └── tasks/manager_based/walker_shared_control/
            ├── __init__.py                       # Gym task registration
            ├── walker_shared_control_env_cfg.py  # Scene, observations, rewards, and timing
            ├── walker_shared_control_robot_cfg.py # Walker USD and actuator configuration
            ├── agents/rsl_rl_ppo_cfg.py          # PPO network and optimizer settings
            └── mdp/
                ├── actions.py                    # User model and damping action
                ├── admittance.py                 # Longitudinal admittance integration
                ├── camera_perception.py          # Depth-to-slope estimator
                ├── observations.py               # Policy observation terms
                └── rewards.py                    # Task reward terms
```

Generated `logs/`, `eval_logs/`, plot exports, and Python cache files are runtime artifacts rather than source-code components.

## Results

The slope-aware controllers were evaluated on a simulated course containing flat terrain, an uphill ramp, a level section, and a downhill ramp.

Each learned controller was evaluated across three independently trained seeds.

| Controller | Velocity RMSE (m/s) | Mean voluntary force (N) | Peak voluntary force (N) | Maximum velocity error (m/s) |
|---|---:|---:|---:|---:|
| Fixed damping | 0.02159 | 20.433 | 24.056 | 0.08111 |
| Reactive PPO | 0.01805 ± 0.00083 | 20.101 ± 0.176 | 24.149 ± 0.150 | 0.08298 ± 0.00300 |
| Predictive PPO | 0.01834 ± 0.00241 | 20.174 ± 0.162 | 23.649 ± 0.740 | 0.07298 ± 0.01480 |

Both learned controllers improved average velocity tracking relative to the fixed-damping controller.

The predictive controller achieved average tracking performance comparable to the reactive controller rather than consistently outperforming it. Its main benefits appeared around particular terrain transitions and extreme responses.

Compared with the reactive controller, the predictive controller produced approximately:

- **10.5% lower transition RMSE** when exiting the uphill section,
- **10.5% lower transition RMSE** when exiting the downhill section,
- **12.1% lower maximum velocity error** overall,
- **2.1% lower peak voluntary interaction force**, and
- **8.8% lower mean positive overspeed**.

Two of the three predictive seeds also learned the expected anticipatory damping behavior before terrain transitions. The remaining seed learned a qualitatively different response, indicating that exploitation of preview information remains sensitive to training seed.

These results support the use of preview terrain information for anticipatory admittance adaptation, while also showing that further work is needed to improve the consistency and generalization of the learned behavior.

## Current Development

The project is being extended from slope-aware control toward a more general terrain-adaptive framework.

Current and planned work includes:

- migrating perception to a simulated ZED stereo camera,
- improving camera-based terrain reconstruction,
- estimating terrain roughness and height variation,
- detecting bumps, steps, and other geometric discontinuities,
- representing perception confidence explicitly,
- procedural terrain generation and domain randomization,
- evaluation on held-out terrain distributions,
- robustness testing under camera noise, dropout, and latency,
- sim-to-real transfer to the physical smart walker, and
- real-world validation of terrain perception and controller behavior.

The intended long-term architecture is:

```text
Stereo / depth perception
          │
          ▼
   Local terrain map
          │
          ├── slope
          ├── roughness
          ├── height variation
          ├── steps / obstacles
          └── confidence
          │
          ▼
Terrain + walker state
          │
          ▼
        PPO
          │
          ▼
Desired admittance damping
          │
          ▼
Safety / validity supervisor
          │
          ▼
Conventional admittance controller
```

## Limitations 

The reported results are currently based on simulation.

In particular:

- the damping range has not been clinically validated,
- the work does not measure physiological or metabolic user effort,
- voluntary interaction force is directly available from the simulated user model,
- the published experiments consider slope rather than arbitrary terrain,
- the perception system uses simulated depth measurements,
- no human-subject or clinical conclusions are made, and
- validation on the physical walker remains future work.

The results should therefore be interpreted as an investigation of **predictive terrain-aware control and reinforcement-learning-based admittance adaptation**, rather than a clinical evaluation of walker assistance.

## Acknowledgments

Thank you to Hans Jarales and the Telerobotic and Biorobotic Systems Group for their guidance throughout this project and for permission to release this code.

Thanks also to Dr. Mahdi Tavakoli for supervising the research.

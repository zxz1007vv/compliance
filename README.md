# Code Release for Learning Force Control for Legged Manipulation


# Table of contents
1. [Overview](#overview)
2. [System Requirements](#requirements)
3. [Training a Model](#simulation)
    1. [Installation](#installation)
    2. [Environment and Model Configuration](#configuration)
    3. [Training and Logging](#training)
    4. [Analyzing the Policy](#analysis)

## Overview <a name="introduction"></a>

This repository provides an implementation of the paper:


<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://tif-twirl-13.github.io/learning-compliance.html" target="_blank">
      <b> Learning Force Control for Legged Manipulation </b>
      </a>
      <br>
      <a href="https://tif-twirl-13.github.io/Home.html" target="_blank">Tifanny Portela</a> and <a href="https://gmargo11.github.io/" target="_blank">Gabriel B. Margolis</a> and <a href="https://yandongji.github.io/" target="_blank">Yandong Ji</a> and <a href="https://people.csail.mit.edu/pulkitag" target="_blank">Pulkit Agrawal</a>
      <br>
      <em>International Conference on Robotics and Automation</em>, 2024
      <br>
      <a href="https://arxiv.org/abs/2405.01402">paper</a> /
      <a href="https://tif-twirl-13.github.io/learning-compliance.html" target="_blank">project page</a>
    <br>
</td>

<br>

This environment builds on the [legged gym environment](https://leggedrobotics.github.io/legged_gym/) by Nikita
Rudin, Robotic Systems Lab, ETH Zurich (Paper: https://arxiv.org/abs/2109.11978) and the Isaac Gym simulator from 
NVIDIA (Paper: https://arxiv.org/abs/2108.10470). Training code builds on the 
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) repository, also by Nikita
Rudin, Robotic Systems Lab, ETH Zurich. All redistributed code retains its
original [license](LICENSES/legged_gym/LICENSE).

Our initial release provides the following features:
* Train a force control and locomotion policy for the Unitree B1 with Z1 arm.

## System Requirements <a name="requirements"></a>

**Simulated Training and Evaluation**: Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. The code can run on a smaller GPU if you decrease the number of parallel environments (`Cfg.env.num_envs`). However, training will be slower with fewer environments.

## Training a Model <a name="simulation"></a>

### Installation <a name="installation"></a>

#### Install pytorch 1.10 with cuda-11.3:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    python examples/1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

#### Install the `b1_gym` package

In this repository, run `pip install -e .`

### Verifying the Installation

If everything is installed correctly, you should be able to run the test script with:

```bash
python scripts/test.py
```

You should see a GUI window with 10 B1+Z1 robots standing in place.

### Environment and Model Configuration <a name="configuration"></a>


**CODE STRUCTURE** The reusable simulator implementation remains in
[legged_robot.py](b1_gym/envs/base/legged_robot.py). The B1+Z1 task and all of
its environment/training overrides live together in
[b1_z1_config.py](b1_gym/envs/b1_z1/b1_z1_config.py). `train.py` only parses
arguments and composes the registered task:

```text
train.py -> TaskRegistry -> B1Z1Env -> OnPolicyRunner -> PPO -> ActorCritic
```

The RL framework is organized by responsibility under
`b1_gym_learn/{runners,algorithms,modules,storage}`. The historical
`b1_gym_learn.ppo_cse` imports remain available for compatibility.

Environment-facing task boundaries are exposed through
`b1_gym/{commands,rewards,sensors,curriculum}`. Historical reward and
curriculum import paths remain exact compatibility aliases. The command module
owns the shared 23-dimensional command-vector contract and command lifecycle;
all 28 sensor implementations are consolidated in `sensors/sensors.py`, with
the historical per-sensor modules retained as exact compatibility imports.

The completed V2 refactor, frozen numerical contracts, and GPU equivalence
results are documented in
[the final execution report](docs/refactor_v2_final_report.md).

The main scripts in [scripts](scripts/) are:

```bash
scripts
├── __init__.py
├── export_policy.py
├── play.py
├── test.py
└── train.py
```

You can run the `test.py` script to verify your environment setup. If it runs then you have installed the gym
environments correctly. To train an agent, run `train.py`. To evaluate a trained agent, run `play.py`. 


### Training and Logging <a name="training"></a>

To train the compliant whole-body controller for B1+Z1, run: 

```bash
python scripts/train.py --task b1_z1_ik
```

The script prints `Saved checkpoint 0` after the first update. TensorBoard is
the default logger. W&B is optional and can be enabled with `--logger wandb`
or `--logger both` after installing `pip install -e '.[wandb]'`.

Common experiment overrides are available on the command line:

```bash
python scripts/train.py \
  --task b1_z1_ik \
  --num-envs 4000 \
  --max-iterations 100000 \
  --save-interval 400 \
  --run-name force_tracking
```

Each run owns all of its artifacts:

```text
logs/b1_z1_ik/<date>_<run-name>/
├── config.json
├── tensorboard/
├── checkpoints/
│   ├── model_000400.pt
│   └── model_latest.pt
└── exported/
    └── policies/
        ├── policy_000400.pt
        └── policy_latest.pt
```

Checkpoints contain policy and optimizer states, iteration/runner statistics,
the full task configuration, and RNG states. Historical checkpoints stored in
the run root or an older `checkpoints/` directory remain loadable. Resume with:

```bash
python scripts/train.py \
  --resume-run-dir logs/b1_z1_ik/<run-directory> \
  --checkpoint latest
```

The GUI is off during training by default. Pass `--viewer` to enable it.
Each checkpoint save also exports a self-contained TorchScript inference policy;
checkpoint and deployment artifacts remain separate.

Training with the default configuration requires about 12GB of GPU memory. If you have less memory available, you can 
still train by reducing the number of parallel environments used in simulation (the default is `Cfg.env.num_envs = 4000`).

### Analyzing the Policy <a name="analysis"></a>

Run a saved checkpoint by passing its local run directory and iteration:

```bash
python scripts/play.py \
  --run-dir logs/b1_z1_ik/<run-directory> \
  --checkpoint 5000 \
  --control-mode position \
  --seed 1
```

Running `python scripts/play.py` without arguments automatically selects the
most recently saved run for `b1_z1_ik`, loads its highest numbered checkpoint,
and opens the Isaac Gym viewer. Pass `--headless` when no window is wanted.

Use `--checkpoint latest` to select the highest numbered checkpoint. At play
startup the policy is exported as one TorchScript file under
`<run-directory>/exported/policies/`. Its filename combines the run name and
checkpoint, for example `wbc_release_5000.pt`.

`--control-mode` accepts only `position`, `force`, `binary`, or `mixed`.
Headless evaluations can use multiple environments and automatically write a
machine-readable JSON report under `<run-directory>/evaluations/`:

```bash
python scripts/play.py \
  --run-dir logs/b1_z1_ik/<run-directory> \
  --checkpoint 5000 \
  --control-mode force \
  --force-amplitude 70 \
  --seed 1 \
  --num-envs 32 \
  --steps 2000 \
  --headless \
  --print-every 0
```

See [the first-round evaluation protocol](docs/round1_closure_evaluation_protocol.md)
for the three-seed position/force matrix, acceptance thresholds, and required
comparison artifacts.

Export without constructing an Isaac Gym environment:

```bash
python scripts/export_policy.py \
  --run-dir logs/b1_z1_ik/<run-directory> \
  --checkpoint latest
```

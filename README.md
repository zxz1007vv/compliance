# WBC Compliance Gym

本仓库用于在 Isaac Gym 中训练和评估足式机器人全身柔顺控制策略。当前提供
Unitree B1+Z1 和 ZGWSARM 轮足四足+机械臂两套独立任务，复用 locomotion +
position/force compliance 控制框架。代码来源于
论文 [Learning Force Control for Legged Manipulation](https://arxiv.org/abs/2405.01402)，
并参考了 legged_gym、rsl_rl 以及
[HIMLoco-for-Go2W](https://github.com/TrackinBIT/HIMLoco-for-Go2W) 的 TaskRegistry 组织方式。

## 1. 当前约定

- 当前注册任务：`b1_z1_ik`、`zgwsarm_compliance`
- 仿真与任务包：`wbc_compliance_gym`
- 强化学习包：`wbc_compliance_rl`
- 正式实验目录：`logs/<task>/<timestamp>_<run-name>/`
- `train.py` 必须使用 `--task` 显式选择注册任务
- `play.py` 必须提供 `--task` 或 `--run-dir`
- play 指定 task 但不指定 run/checkpoint 时，自动使用该任务的最新 run 和最高编号 checkpoint
- C++ MuJoCo sim2sim 与 Logitech F710 遥操作入口：`mujoco/`

所有命令均建议在仓库根目录执行：

```bash
cd /home/ubuntu/zskj/learning-compliance
conda activate compliance
```

## 2. 安装与环境

当前代码已在以下环境验证：

- Ubuntu Linux
- Python 3.8
- Isaac Gym Preview 4
- PyTorch 1.13.1 + CUDA 11.7
- NVIDIA GPU

先安装 Isaac Gym，再安装本项目的 editable package：

```bash
cd /path/to/isaacgym/python
python -m pip install -e .

cd /home/ubuntu/zskj/learning-compliance
python -m pip install -e . --no-deps
```

运行基础验证：

```bash
python -m unittest discover -s tests -v
```

### C++ MuJoCo sim2sim

仓库现已同时提供 `zgwsarm_compliance` 和 `b1_z1_ik` 的 C++ 部署端。它从每个训练
run 导出精确的 TorchScript、观测/关节/控制合同，支持 F710 位置/力模式遥操作、
三种接触场景、无界面运行和 MuJoCo 官方 classic `simulate` Viewer。完整依赖、导出、
构建、手柄映射和运行命令见
[mujoco/README.md](mujoco/README.md)。

ZGWSARM 的 force mode 现在把末端锚点独立于 position trajectory 锁存，并默认保存在
随底盘 X/Y/yaw、固定世界高度的命令基坐标系中；Isaac Gym 与 MuJoCo 使用相同的
`robot_relative_static` 生命周期。力命令是策略跟踪目标，不会被直接施加为外力；显式
free 状态才会关闭虚拟弹簧，零力命令本身不会。Viewer 也支持用
`Ctrl+右键` 拖动选中的机械臂刚体进行零力命令柔顺性测试。

训练端现在按每个新的 force segment 独立采样锚点模式：
`robot_relative_static/robot_relative_moving = 0.8/0.2`。`world_fixed` 不进入训练随机化，
避免世界固定接触点与持续的底盘速度/yaw-rate 跟踪形成额外对抗任务。
moving 模式以配置的低速分段运动连续更新局部锚点，并限制相对初始锁存点的最大偏移；
模式不会在每个物理步重新采样。free/position/reset 会清空锚点状态，下次进入有效 force
segment 时重新采样和锁存。

`boxes`/`boxes_tm` 地形现在使用两组互不复用的概率配置：
`heightfield_terrain_proportions` 包含 10 类基础高度场，
`box_terrain_proportions` 包含 5 类 Box 结构地形。两者的固定类别顺序写在任务配置旁，
启动时会检查维数、非负性及概率和是否为 1，避免同一概率槽在两个生成阶段表示不同地形。

这次锚点改动保持 23 维命令、观测、奖励、动作、PD 增益和位置轨迹逻辑不变。旧的
`2026-08-24_18-44-10_wbc_release` 策略仍可用于修改前后的 A/B 基线评估，但它没有针对
新锚点随机化重新训练，因此不应单独作为最终训练效果结论。当前弹簧阻尼仍为
`Kd * (0 - ee_velocity)`；moving anchor velocity 尚未进入阻尼项，以保持本轮实验变量单一。

导出策略到 `mujoco/policies/` 后，使用
`mujoco/build/mujoco_sim --task zgwsarm_compliance` 或
`mujoco/build/mujoco_sim --task b1_z1_ik` 显式选择任务。机器人 MJCF 路径、
TorchScript 策略路径、Viewer 和手柄参数均在 `mujoco/config/<task>.yaml` 中修改。

### 关于 Gym 警告

运行时可能看到：

```text
Gym has been unmaintained since 2022 ...
Please upgrade to Gymnasium ...
```

这是旧版 `gym` 打出的迁移提示，不是本次 play 失败的原因。Isaac Gym Preview 4 及当前
wrapper 仍使用 `gym` API，因此不要仅为消除提示直接全局替换成 Gymnasium。Gymnasium
迁移应作为单独兼容性任务验证。

## 3. 代码框架

```text
learning-compliance/
├── scripts/
│   ├── train.py                 # 训练入口
│   ├── play.py                  # 本地 checkpoint 评估入口
│   ├── export_policy.py         # 独立 TorchScript 导出
│   ├── export_deployment_bundle.py # C++ sim2sim 完整部署包导出
│   ├── generate_mujoco_models.py # 双任务 MuJoCo 场景生成
│   └── test.py                  # 无训练权重的仿真检查
├── mujoco/                       # C++ MuJoCo sim2sim、F710、场景与测试
├── wbc_compliance_gym/
│   ├── envs/
│   │   ├── base/                # 通用 Isaac Gym 环境与物理流程
│   │   ├── wrappers/            # observation history 等 wrapper
│   │   └── b1_z1_compliance/
│   │       ├── b1_z1_config.py  # 当前任务全部 Environment/PPO 配置
│   │       └── b1_z1_env.py     # 当前任务环境类型
│   ├── robots/
│   │   ├── configs/             # 机器人配置片段
│   │   └── *.py                 # Isaac Gym robot asset loader
│   ├── commands/                # 23 维 command schema、采样与生命周期
│   ├── curriculum/              # curriculum 算法
│   ├── rewards/b1_z1.py         # B1+Z1 reward 公式
│   ├── sensors/                 # observation/privileged observation sensor
│   ├── terrains/                # 地形实现
│   └── utils/                   # TaskRegistry、config、artifact 等工具
├── wbc_compliance_rl/
│   ├── algorithms/              # PPO
│   ├── modules/                 # ActorCritic、adaptation module
│   ├── runners/                 # rollout、更新、保存与自动导出
│   ├── storage/                 # rollout storage
│   └── logging/                 # TensorBoard/W&B logger
├── resources/robots/b1_z1/      # B1、Z1 和 B1+Z1 URDF/mesh/xacro
└── logs/                        # 正式训练与评估产物
```

训练链路：

```text
train.py --task
  -> TaskRegistry
  -> task env config + train config
  -> task env + wrappers
  -> OnPolicyRunner
  -> PPO
  -> ActorCritic / AdaptationModule
```

play 链路：

```text
play.py --task/--run-dir
  -> 解析 task、latest run、latest checkpoint
  -> TaskRegistry
  -> 加载该 run 的 config.json
  -> task play_cfg_hook
  -> task env + wrappers
  -> 加载 ActorCritic state_dict
  -> rollout + evaluation JSON
```

## 4. 配置在哪里修改

日常训练参数的唯一主要入口是：

```text
wbc_compliance_gym/envs/b1_z1_compliance/b1_z1_config.py
```

配置与实现的边界如下：

| 修改目标 | 修改位置 |
|---|---|
| 环境数量、episode、terrain、control、domain randomization | `b1_z1_config.py` |
| command 范围、重采样、hybrid mode、课程开关/阈值 | `b1_z1_config.py` |
| reward 权重、tracking sigma、目标值 | `b1_z1_config.py` |
| PPO、ActorCritic、runner、保存间隔 | `B1Z1CfgPPO` |
| command 采样/更新算法或新增 command 维度 | `commands/commands.py` |
| curriculum 更新算法 | `curriculum/curriculum.py` |
| reward 数学公式或新增 reward | `rewards/b1_z1.py` |
| observation sensor 实现 | `sensors/sensors.py` |

不要手动修改 `logs/.../config.json`。它是某一次训练解析完成后的完整只读配置快照，
用于复现、resume 和 play 历史 checkpoint。

## 5. TaskRegistry 与任务名称

查看当前注册任务：

```bash
python scripts/train.py --list-tasks
# 或
python scripts/play.py --list-tasks
```

当前输出：

```text
b1_z1_ik
zgwsarm_compliance
```

任务名称必须显式使用注册名。未知任务会直接报错并列出可用任务。

## 6. 训练命令

### 6.1 启动训练

```bash
python scripts/train.py --task b1_z1_ik
# ZGWSARM
python scripts/train.py --task zgwsarm_compliance
```

`--task` 是必填参数，避免多机器人环境下误用默认任务。直接运行
`python scripts/train.py` 会返回参数错误。

训练默认 headless。默认运行名称分别由任务配置顶部的
`B1_Z1_TRAINING_NAME` / `ZGWSARM_TRAINING_NAME` 控制；单次运行推荐使用
`--run-name` 覆盖，不需要修改文件。环境数量、训练轮数和保存间隔也由对应任务配置管理。

### 6.2 常用覆盖参数

```bash
python scripts/train.py \
  --task b1_z1_ik \
  --num-envs 4000 \
  --max-iterations 100000 \
  --save-interval 400 \
  --run-name force_tracking \
  --logger tensorboard
```

打开 viewer：

```bash
python scripts/train.py --task b1_z1_ik --viewer
```

### 6.3 CUDA/PhysX 崩溃调试

物理调试模式默认关闭。复现 CUDA 非法访存、PhysX contact kernel 崩溃或不明
NaN/Inf 时，可先缩小环境数量并开启同步检查：

```bash
COMPLIANCE_CUDA_DEBUG=1 \
COMPLIANCE_CUDA_DEBUG_INTERVAL=100 \
CUDA_LAUNCH_BLOCKING=1 \
python scripts/train.py --task zgwsarm_compliance --num-envs 512
```

该模式会打印最终使用的环境数和 PhysX contact buffer 配置，在每个物理子步检查
action、torque、force、DOF、root/body state 和 contact force，并在首次发现 NaN/Inf
时报告 tensor 名称、索引和环境 ID。它还会列出最严重的硬限位越界、目标夹紧、
速度超限和接触力，包含对应的环境 ID、关节名或刚体名。若 CUDA 异步错误在同步点
暴露，日志会附上具体物理阶段以及最后一组有效的 tensor、DOF 和接触诊断。

- `COMPLIANCE_CUDA_DEBUG_INTERVAL`：周期统计输出间隔，默认 `100` 个控制步。
- `COMPLIANCE_CUDA_DEBUG_SYNC`：是否在关键阶段主动同步 CUDA，默认 `1`。

完整 tensor 检查和 CUDA 同步会明显降低训练速度，仅在复现问题时开启。正常训练
不要设置 `COMPLIANCE_CUDA_DEBUG`。

ZGWSARM 任务还启用了常驻保护：腿和机械臂的位置目标被夹紧到 URDF 硬限位，
机械臂目标的逐物理步变化量受 URDF 速度限制，reset 后 position-drive target 与随机
初始关节位置对齐，连续轮关节不参与位置夹紧；`asset.self_collisions = 1`（Isaac Gym
中 `1` 表示禁用自碰撞）；有限位关节越界超过 `0.05 rad` 或关节速度超过 URDF
限制两倍时只重置对应环境，非足端刚体接触力超过 `5000 N` 时也会重置对应环境；
GPU contact pair 容量提升为 `2 ** 24`。这些设置在不开启 debug 时同样生效。

### 6.4 从该任务最新 run 续训

```bash
python scripts/train.py \
  --task b1_z1_ik \
  --resume \
  --checkpoint latest
```

`--resume` 未指定目录时，会自动选取 `logs/b1_z1_ik/` 下最新的有效 run。

### 6.5 从指定 run/checkpoint 续训

```bash
python scripts/train.py \
  --task b1_z1_ik \
  --resume-run-dir logs/b1_z1_ik/2026-08-13_11-32-29_wbc_release \
  --checkpoint 4800
```

`--resume-run-dir` 隐含 resume。程序会验证 run 中记录的任务是否与 `--task` 一致。

## 7. Play 与评估命令

### 7.1 指定任务的最新模型

```bash
python scripts/play.py --task b1_z1_ik --checkpoint latest
# ZGWSARM
python scripts/play.py --task zgwsarm_compliance --checkpoint latest
```

`play.py` 不设置默认任务。直接运行 `python scripts/play.py` 会提示必须提供
`--task` 或 `--run-dir`。

行为是：

1. 选择 `logs/b1_z1_ik/` 下最新的有效 run；
2. 选择该 run 中最高编号的 `model_*.pt`；
3. 默认打开 Isaac Gym viewer；
4. 默认运行 2000 steps；
5. 将统计结果写入该 run 的 `evaluations/`。

### 7.2 指定 run，自动推断 task

```bash
python scripts/play.py \
  --run-dir logs/b1_z1_ik/2026-08-13_11-32-29_wbc_release \
  --checkpoint latest
```

显式 `--run-dir` 时可以省略 `--task`，程序会从该 run 的 `config.json` 读取任务名。

### 7.3 同时指定 task 和 run

```bash
python scripts/play.py \
  --task b1_z1_ik \
  --run-dir logs/b1_z1_ik/2026-08-13_11-32-29_wbc_release \
  --checkpoint 4800
```

如果 task 与 run 不匹配，程序会拒绝加载，避免错误组合模型和环境。

### 7.4 单独评估 position/force mode

不传 `--control-mode` 时，使用对应 task 的 `configure_*_play()` 中定义的
默认模式；`--control-mode` 只对当次运行做临时覆盖。当前
`zgwsarm_compliance` 和 `b1_z1_ik` 的 play 默认模式都是 `position`。

Position：

```bash
python scripts/play.py \
  --task b1_z1_ik \
  --control-mode position \
  --seed 1
```

Force：

```bash
python scripts/play.py \
  --task b1_z1_ik \
  --control-mode force \
  --force-amplitude 70 \
  --seed 1
```

`--control-mode` 只接受：

```text
position | force | binary | mixed
```

### 7.5 Headless 批量评估

```bash
python scripts/play.py \
  --task b1_z1_ik \
  --checkpoint latest \
  --control-mode force \
  --force-amplitude 70 \
  --seed 1 \
  --num-envs 32 \
  --steps 2000 \
  --headless \
  --print-every 0
```

也可以用 `--output /path/to/result.json` 指定评估 JSON 路径。

### 7.6 Teacher / student A/B 评估

同一个训练 checkpoint 内包含共享 actor 和 adaptation module：

- `student`（默认）用观测历史经过 adaptation module 估计 16 维 privileged latent，
  与导出的 TorchScript/MuJoCo 策略一致；
- `teacher` 在 Isaac Gym 中直接把真实 privileged observation 输入同一个 actor，
  用于判断误差来自 adaptation/反馈瓶颈，还是 actor、奖励和任务本身。

使用相同 run、checkpoint、seed 和诊断参数分别运行：

```bash
python scripts/play.py \
  --run-dir logs/zgwsarm_compliance/<run> --checkpoint 6500 \
  --diagnostic-scenario force --control-mode force \
  --force-amplitude 10 --policy-mode student --headless

python scripts/play.py \
  --run-dir logs/zgwsarm_compliance/<run> --checkpoint 6500 \
  --diagnostic-scenario force --control-mode force \
  --force-amplitude 10 --policy-mode teacher --headless
```

Teacher 只用于 Gym 诊断，不能导出到 MuJoCo，因为部署环境不存在这些真实 privileged
量。Teacher 显著优于 student 表明 adaptation/可观测性是主要瓶颈；两者都差则应优先检查
actor、奖励、命令课程和虚拟力场设计。两次运行会把 `policy_mode` 写入评估 JSON。

纯 yaw 的确定性 Gym 检查可以分别运行：

```bash
python scripts/play.py --run-dir logs/zgwsarm_compliance/<run> --checkpoint 6500 \
  --diagnostic-scenario velocity_arm_fixed --diagnostic-lin-vel-x 0 \
  --diagnostic-ang-vel-yaw 0.2 --policy-mode student --headless

python scripts/play.py --run-dir logs/zgwsarm_compliance/<run> --checkpoint 6500 \
  --diagnostic-scenario velocity_arm_fixed --diagnostic-lin-vel-x 0 \
  --diagnostic-ang-vel-yaw -0.2 --policy-mode student --headless
```

MuJoCo 中更适合使用 `mujoco/scripts/run_diagnostic.sh --scenario yaw`，它会在一次运行中
自动完成正负命令并汇总左右轮符号和 `wheel_yaw_hat`；详见 `mujoco/README.md`。



## 8. 日志、checkpoint 与 policy

每次训练创建独立目录：

```text
logs/<task>/<timestamp>_<run-name>/
├── config.json
├── tensorboard/
├── checkpoints/
│   ├── model_000000.pt
│   ├── model_000400.pt
│   └── model_latest.pt
├── exported/
│   └── policies/
│       ├── policy_000000.pt
│       ├── policy_000400.pt
│       └── policy_latest.pt
└── evaluations/
```

- `model_*.pt`：完整训练 checkpoint，包含网络、优化器、runner、配置和 RNG 状态；用于续训。
- `policy_*.pt`：TorchScript 推理模型；用于部署或独立推理。
- 每次 runner 保存 checkpoint 时会自动导出对应 policy 并刷新 `policy_latest.pt`。
- `play.py` 只负责评估，不重复导出 policy。

手动补导指定 checkpoint：

```bash
python scripts/export_policy.py \
  --run-dir logs/b1_z1_ik/2026-08-13_11-32-29_wbc_release \
  --checkpoint 4800
```

TensorBoard：

```bash
tensorboard --logdir logs
```

## 9. 机器人资源

当前 B1、Z1 和组合模型统一位于：

```text
resources/robots/b1_z1/
├── urdf/
│   ├── b1.urdf
│   ├── z1.urdf
│   └── b1_plus_z1.urdf
├── meshes/
│   └── z1/
└── xacro/
    └── z1/
```

旧 run 的 `config.json` 仍可能记录 `resources/robots/b1/...`。加载器会在内存中自动迁移到
`resources/robots/b1_z1/...`，不会修改历史日志文件。

ZGWSARM 资源位于：

```text
resources/robots/zgwsarm/
├── urdf/zgwsarm.urdf       # Isaac Gym 训练资产，BASE_LINK 为浮动根
├── meshes/                 # STL，仅使用仓库内相对路径
├── zgwsarm.xml             # 原模型的 MuJoCo 版本
├── scene_terrain.xml
└── *.png                   # MuJoCo height-field 依赖
```

ZGWSARM 的实际 Isaac Gym DOF 顺序是 `FR(FAR) 4 + FL(FBL) 4 + RR(RAR) 4 +
RL(RBL) 4 + arm 6`，每条轮足链依次为 `ABAD/HIP/KNEE/FOOT`。策略 action
严格使用这一 22 维顺序；末端为 `ROBOT_ARM_LINK7`，base 为 `BASE_LINK`。该任务的
维度为 observation=96、privileged observation=16、history=960、action=22、command=23。

关键入口：

| 内容 | 路径 |
|---|---|
| Asset loader | `wbc_compliance_gym/robots/zgwsarm.py` |
| 机器人名称、DOF 分组、默认姿态、PD | `wbc_compliance_gym/robots/configs/zgwsarm.py` |
| 训练/PPO/Play 配置 | `wbc_compliance_gym/envs/zgwsarm_compliance/zgwsarm_compliance_config.py` |
| 环境与轮力矩控制 | `wbc_compliance_gym/envs/zgwsarm_compliance/zgwsarm_compliance_env.py` |
| 独立 reward 索引适配 | `wbc_compliance_gym/rewards/zgwsarm.py` |

ZGWSARM 仿真步长为 `0.002 s`、控制 decimation 为 `5`，对应 500 Hz 物理仿真和
100 Hz 策略频率。出生高度为 `0.55 m`，目标工作高度为 `0.54 m`。ABAD/HIP/KNEE
使用位置 PD，增益分别为 `90/1`、`120/1`、`120/1`；轮关节不跟踪角度，使用
`torque = 15 * action - 0.2 * wheel_velocity` 并按 URDF 裁剪到 `±28 N·m`。
轮角不进入策略关节位置观测，真实轮速仍保留。

ZGWSARM 的 base velocity 不再进入多维 bin grid，而使用“先采 locomotion mode、再在
激活维度内连续采样”的课程。当前 Phase A 使用 `stand/pure_yaw=0.1/0.9`，yaw 初始范围
为 `±0.3 rad/s`，并排除 `|yaw_rate|<0.15 rad/s` 的近零命令；最终 limit 为
`±1.5 rad/s`。yaw 根据 pure-mode MAE 独立按 `0.1` 扩张，运行时状态随 checkpoint 保存。

pure-yaw 使用与 `ClockSensor` 同源的 1.2 Hz 四拍 crawl：
`FAR → RBL → FBL → RAR`。每拍由 `yaw_gait_support` 引导一足卸载、三足承重，
并使用 15% 卸载、70% 摆动、15% 重载的连续接触包络。`yaw_foothold_tracking`
引导 swing 足沿 yaw 切向移动：最大步长为 3 cm，在 `|yaw|=0.3 rad/s` 达到最大值，
较小命令按比例缩短；有界切向速度 reward 只在中间运动段提供进度信号。pure-yaw 下
旧的四轮接触和逐轮最低载荷项关闭，其他 mode 保持原逻辑；机身与轮速 yaw tracking
继续提供最终任务目标。

Phase A 使用平地、固定机械臂 nominal pose、零 force command，并关闭 roughness、push、
COM、gravity 和 motor randomization。形成稳定转向机构后，再从该 checkpoint 进入
Phase B/C，逐步恢复直线/弧线命令、机械臂 compliance 和域随机化。

ZGWSARM 原地 Position/Force/Binary 测试：

```bash
python scripts/play.py --task zgwsarm_compliance --checkpoint latest --control-mode position
python scripts/play.py --task zgwsarm_compliance --checkpoint latest --control-mode force --force-amplitude 70
python scripts/play.py --task zgwsarm_compliance --checkpoint latest --control-mode binary --force-amplitude 70
```

Play hook 会把底盘速度命令固定为零。`config.json`、checkpoint、续训和 TorchScript
policy 导出仍使用通用格式，且 play 会拒绝 task 与 run 不匹配的组合。

## 10. 增加新机器人/任务

例如增加 `b2_z2_compliance`：

1. 将 URDF、mesh、xacro 放入 `resources/robots/b2_z2/`；
2. 在 `wbc_compliance_gym/robots/` 增加 asset loader；
3. 在 `wbc_compliance_gym/robots/configs/` 增加机器人配置片段；
4. 新建任务目录：

   ```text
   wbc_compliance_gym/envs/b2_z2_compliance/
   ├── __init__.py
   ├── b2_z2_config.py
   └── b2_z2_env.py
   ```

5. 如 reward 公式不同，在 `rewards/` 增加对应实现并注册 reward container；
6. 在 `wbc_compliance_gym/envs/__init__.py::register_tasks()` 注册：
   environment、environment config、training config、runner、wrappers 和 play config hook；
7. 增加 observation/action/command 维度、配置指纹、资源加载和短训练测试；
8. 运行：

   ```bash
   python scripts/train.py --task b2_z2_compliance
   python scripts/play.py --task b2_z2_compliance
   ```

任务特有的 play 覆盖应放在该任务的 `*_config.py` 中，并通过 `play_cfg_hook` 注册；不要把
新机器人的参数硬编码回 `scripts/play.py`。

# mujoco C++ sim2sim

`mujoco/` 是本仓库的 C++ MuJoCo 部署端，同时支持：

- `zgwsarm_compliance`：22 维动作、96 维单帧观测、10 帧历史、100 Hz 控制；
- `b1_z1_ik`：19 维动作、87 维单帧观测、10 帧历史、50 Hz 控制；
- TorchScript adaptation module + actor 的 CPU 推理；
- Logitech F710（XInput/X 模式）遥操作与热插拔；
- flat、wall、block 三种基础场景、ZGWSARM terrain 场景，以及 MuJoCo 官方 classic
  `simulate` Viewer。所有场景都带天空盒、灯光和棋盘地面，不再依赖 Viewer 的黑色默认背景。

这里的 `runtime.cfg` 不是手写参数文件。它由训练 run 的 `config.json` 和 checkpoint
一起导出，固定关节顺序、观测缩放、命令范围、PD 增益、动作缩放、硬限位和控制频率。
因此 ZGWSARM 使用本机机器人参数，B1+Z1 使用论文原始任务参数，两者不会混用。

## 1. 依赖与构建

Ubuntu 上安装基础依赖：

```bash
sudo apt install build-essential cmake libsdl2-dev libglfw3-dev libyaml-dev
```

下载 MuJoCo 的 Linux x86-64 binary release 并设置根目录，例如：

```bash
export MUJOCO_ROOT=/opt/mujoco-3.9.0
conda activate compliance
mujoco/scripts/configure.sh
```

构建脚本会从当前 Python 环境定位 Torch。若使用官方 CPU LibTorch，也可以直接配置：

```bash
cmake -S mujoco -B mujoco/build \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DMUJOCO_ROOT="$MUJOCO_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build mujoco/build --parallel
ctest --test-dir mujoco/build --output-on-failure
```

SDL2 缺失时仍可构建策略一致性测试和无手柄仿真，但运行时会明确报告控制器支持未编译；
MuJoCo 缺失时只构建 core、契约测试和 TorchScript parity 工具。

## 2. 策略格式与导出目录

这里有两种不要混淆的“模型”：

- `mujoco/models/` 保存机器人和场景的 MJCF/XML 物理模型；
- `mujoco/policies/` 保存神经网络策略部署包。

C++ 端当前使用 LibTorch，因此策略文件是 TorchScript `policy.pt`，不是 ONNX。TorchScript
可以直接在 C++ 加载，并且当前导出已经把 adaptation module 和 actor 合成一个完整推理图。
ONNX 只有在改用 ONNX Runtime 后端时才需要，同时会新增运行库和 parity 验证工作。

导出器会生成 `policy.pt`、`manifest.json`、`runtime.cfg`、3 组 float32 黄金向量和
SHA-256 校验文件：

```bash
python scripts/export_deployment_bundle.py \
  --run-dir logs/zgwsarm_compliance/<run> --checkpoint latest \
  --output-dir mujoco/policies/zgwsarm

python scripts/export_deployment_bundle.py \
  --run-dir logs/b1_z1_ik/<run> --checkpoint latest \
  --output-dir mujoco/policies/b1_z1
```

在启动仿真前验证 Python/C++ 推理一致性：

```bash
mujoco/build/mujoco_policy_parity \
  mujoco/policies/zgwsarm
```

## 3. YAML 启动配置与运行

每次启动都必须显式选择任务，程序随后读取同名 YAML：

- `--task zgwsarm_compliance` → `mujoco/config/zgwsarm_compliance.yaml`
- `--task b1_z1_ik` → `mujoco/config/b1_z1_ik.yaml`

ZGWSARM 可视化遥操作：

```bash
mujoco/build/mujoco_sim --task zgwsarm_compliance
```

ZGWSARM 默认加载 `models/zgwsarm/scene_terrain.xml`；其中的高度场 PNG 和机器人 XML
都保存在同一模型目录内，不依赖仓库外路径。需要回到无障碍平地时，在 YAML 中把
`paths.scene_path` 改为 `../models/zgwsarm/scene_flat.xml`，也可临时使用：

```bash
mujoco/build/mujoco_sim --task zgwsarm_compliance \
  --scene mujoco/models/zgwsarm/scene_flat.xml
```

B1+Z1：

```bash
mujoco/build/mujoco_sim --task b1_z1_ik
```

配置里的相对路径以配置文件所在目录为基准，因此从其他工作目录启动也不会改变路径含义。
也可以用 `--config /path/custom.yaml` 为选定任务加载另一份外挂配置；YAML 内的
`task_name` 必须与 `--task` 一致。命令行参数仍可临时覆盖配置，例如：

```bash
mujoco/build/mujoco_sim --task zgwsarm_compliance \
  --headless --steps 500 --no-realtime
mujoco/build/mujoco_sim --task zgwsarm_compliance \
  --scene mujoco/models/zgwsarm/scene_wall.xml
```

可直接修改的启动配置如下：

| 配置项 | 含义 |
|---|---|
| `paths.deployment_bundle` | 包含 `policy.pt` 和 `runtime.cfg` 的策略目录 |
| `paths.scene_path` | MJCF/XML 机器人与场景路径 |
| `paths.policy_path` | TorchScript 策略路径，通常位于部署包内 |
| `runtime.viewer/realtime/steps` | 界面、实时节拍和有限步运行 |
| `runtime.status_interval_seconds` | 终端状态输出周期，`0` 表示关闭 |
| `teleoperation.*` | 手柄 deadzone、力/位置/腕/夹爪速度 |
| `teleoperation.force_field_enabled` | force mode 下启用虚拟弹簧力场；关闭即为显式 free 状态 |
| `teleoperation.force_anchor_mode` | `robot_relative_static`、`robot_relative_moving` 或 `world_fixed` |
| `teleoperation.force_anchor_velocity_range` | moving 模式的局部锚点速度模长范围，单位 m/s |
| `teleoperation.force_anchor_motion_duration_s` | moving 模式每段恒速运动的持续时间范围 |
| `teleoperation.force_anchor_offset_limit` | moving 模式相对初始锁存点的局部 XYZ 最大偏移 |
| `teleoperation.force_field_stiffness/damping/limit` | 力场的 K、D 和逐轴力限幅 |
| `startup.*` | A 键匍匐/起立姿态、PD 增益和两段插值时间 |

`policy_path` 可以是相对路径，也可以直接写绝对路径。例如下面这个文件是仓库导出器生成的
TorchScript，能够直接加载：

```yaml
paths:
  policy_path: /home/user/PROJECT/compliance/logs/zgwsarm_compliance/2026-08-21_16-11-44_wbc_release/exported/policies/policy_001200.pt
```

但 `checkpoints/model_001200.pt` 是训练 checkpoint，不能直接交给 C++ 推理器。直接切换
`policy_*.pt` 时，新策略必须来自相同任务和兼容的训练配置；`deployment_bundle` 仍提供它
所需的观测维度、关节顺序和控制合同。启动日志因此把部署包中的 checkpoint 标记为
`contract_checkpoint`，实际加载的策略以打印出的 `policy=` 路径为准。

部署包中的 `runtime.cfg` 仍由 checkpoint 和训练配置自动导出，保存观测顺序、关节顺序、动作缩放、
PD 增益、物理步长和安全限位等训练合同。不要把这些值复制到启动配置中随意修改；训练参数
发生变化时应重新导出整个部署包。命令行 `--bundle`、`--scene`、`--policy` 等旧接口仍然
保留，便于自动化回归。

服务器/回归测试可使用 `--headless`。`--steps N --no-realtime` 用于有限步快速回归。
单独检查手柄原始输入可运行 `mujoco/build/mujoco_gamepad_probe`。

### 确定性 force/yaw 诊断

`mujoco_diagnostic` 与正常的 `mujoco_sim` 并列，不读取手柄，也不改变正常遥操作路径。
它会自动完成匍匐、起立和 RL 接管，并默认使用平地场景。薄脚本会定位已构建的程序：

```bash
# Z 轴力：0 -> 5 -> 10 -> -5 N，每段 5 s，目标转换使用 1 s 线性渐变
mujoco/scripts/run_diagnostic.sh \
  --scenario force --axis z --values 0,5,10,-5 \
  --segment-seconds 5 --settle-seconds 2 --ramp-seconds 1 \
  --stiffness 200 --damping 20 --force-limit 30 \
  --viewer --realtime

# 纯 yaw：先 +0.2，再 -0.2 rad/s；vx 始终为 0
mujoco/scripts/run_diagnostic.sh \
  --scenario yaw --yaw-values 0.2,-0.2 \
  --segment-seconds 5 --settle-seconds 2 \
  --viewer --realtime
```

服务器上去掉 `--viewer --realtime` 即为无界面快速测试。可用 `--policy FILE`、
`--bundle DIR` 或 `--scene FILE` 临时覆盖 YAML，但不会改写 YAML。

Force 状态行给出 yaw 命令系下的 `cmd/actual/abs_error`、锚点位移、世界系未限幅力
`raw_world` 和限幅后力 `clipped_world`。每段末尾汇总 XYZ MAE、锚点位移均值/最大值、
世界 XYZ 分轴饱和率和任一轴饱和率。默认 `K=200,D=20` 对应训练中的
`prop_kd=0.1`；普通遥操作 YAML 的 `D=6` 不会影响这项独立诊断。

Yaw 状态行及分段汇总给出实际底盘 yaw rate、左右轮平均滚动线速度、按训练公式计算的
`wheel_yaw_hat` 和方向检查。正 yaw 的期望符号是左轮负、右轮正；负 yaw 相反。
完整参数可运行：

```bash
mujoco/scripts/run_diagnostic.sh --help
```

`--viewer` 直接复用 MuJoCo binary release 内的 `simulate/` 源码，因此界面与常用的
官方 MuJoCo Viewer 一致：左右侧栏、鼠标相机、关节/执行器面板、传感器和 profiler
都可用。它不是只绘制模型的简化 GLFW 窗口。构建可视化程序时，`MUJOCO_ROOT` 必须
指向完整的 MuJoCo binary release（其中应包含 `include/`、`lib/` 和 `simulate/`）；
只安装 MuJoCo 动态库时仍可运行无界面 sim2sim，但不能启用这个官方 Viewer。

## 4. Logitech F710 映射

先把 F710 背面的模式开关置于 `X`，再启动程序。程序启动后处于
`dog_zero_arm_hold`：四足输出零力矩，机械臂立即用部署合同中的 PD 增益保持默认姿态。

1. 按 A，四足用 PD 从当前姿态插值到匍匐姿态，再插值到起立姿态；每段 1 秒，机械臂继续保持默认姿态。
2. 终端显示 `control_state=standby` 后按 B，四足和机械臂立即一起交给 RL 策略。
3. 显示 `control_state=rl` 后进入完整策略遥操作。过早按 B 会被忽略并打印提示。
4. 任意时刻按 Y 都会重置仿真，回到“四足零力矩、机械臂默认姿态保持”。

RT、LT、Start、Back 不参与控制，也不需要按住任何使能键。

| 输入 | 位置模式 | 力模式 |
|---|---|---|
| 左摇杆 Y | 基座前进/后退 `vx` | 同左 |
| 左摇杆 X | 基座偏航 `yaw rate` | 同左 |
| 右摇杆 Y | 末端球坐标 pitch | `Fx` |
| 右摇杆 X | 末端球坐标 yaw | `Fy` |
| RB - LB | 末端球坐标 radius | `Fz` |
| A | 执行匍匐→起立 PD 序列 | 同左 |
| B | 起立完成后将整机交给 RL 策略 | 同左 |
| X | 切换到力模式 | 切换到位置模式 |
| Y | 重置：四足零力矩、机械臂默认姿态保持 | 同左 |
| D-pad 左/右 | 腕关节 q6 | 同左 |
| D-pad 上/下 | B1+Z1 夹爪；ZGWSARM 无可驱动夹爪自由度 | 同左 |

这里的 D-pad 是手柄左侧十字方向键，不是左摇杆。摇杆有 0.10 deadzone；位置目标采用
积分并受训练命令范围限制，力目标是直接映射并受各任务力范围限制。默认启用
`teleoperation.force_field_enabled`：进入 force mode 且 RL 接管时，程序锁存当时的末端位置。
ZGWSARM 默认使用 `force_anchor_mode: robot_relative_static`，把锚点保存在与训练端一致的
“固定世界高度、随底盘 X/Y/yaw”的命令基坐标系中；底盘 roll/pitch 或地形倾斜不会旋转锚点。
也可在 YAML 中选择 `robot_relative_moving`，让局部锚点以受限、连续的分段低速轨迹移动；
`world_fixed` 则保持锁存时的世界坐标，但只保留为墙面/固定物体的显式诊断选项，不进入
当前训练随机化。训练端按 0.8/0.2 为每个 force segment 随机采样 static/moving，MuJoCo
单机器人部署端通过 YAML 固定一种模式，便于分别评估。
robot-relative 模式在每个物理子步先用当前底盘 X/Y/yaw 更新世界锚点，然后施加
`F_spring = clamp(K (p_anchor - p_ee) - D v_ee)`，与训练环境的虚拟弹簧—阻尼力场一致。
力命令 XYZ 只作为策略要跟踪的目标，不会被直接写成外力；策略需要移动末端、压缩虚拟弹簧，
使实际 `F_spring` 接近期望力。切回 position mode、退出 RL 或 reset 时力场关闭，下次进入
force mode 会在新的当前位置重新锁存锚点。
`force_field_enabled: false`（或 `--no-force-field`）是显式 free 测试；单纯把 `Fx/Fy/Fz`
设为零并不会关闭弹簧，因此“零力命令较稳”不能直接等价为策略本身不抖。
力模式观测会把位置命令 15:18 清零，与训练端 `RCSensor` 一致。状态行里的 `gamepad=1`
表示 SDL 已识别手柄；若一直为 `0`，可运行 `mujoco/build/mujoco_gamepad_probe` 查看原始轴
和按键。可用 `--no-force-field` 临时关闭虚拟力场，只保留策略观测和鼠标扰动。

### 鼠标柔顺性测试

MuJoCo 官方 Viewer 保留右键给相机，因此拖动刚体使用标准 perturbation 操作：

1. 切到 force mode，并把 `Fx/Fy/Fz` 保持为零；
2. 在机械臂连杆上双击鼠标左键，选中的刚体会高亮；
3. 按住 `Ctrl` 和鼠标右键拖动，对选中刚体施加平移扰动力；
4. 松开鼠标后扰动力立即消失，观察机械臂和四足是否柔顺恢复且没有失稳。

`Shift+Ctrl+右键` 改变拖动平面。鼠标 perturbation 与虚拟弹簧力可以叠加，
但做零命令柔顺性测试时应先确认日志中的 `force_yaw` 命令项为零。状态输出同时给出
`anchor` 的 mode、active、local/world 坐标及世界系位移，便于核对锚点生命周期。

状态输出按位置跟踪、力/柔顺性和关节状态分块。位置部分同时打印球坐标、机械臂 XYZ
的命令值、实际值和误差；force mode 下位置目标对策略无效，日志会明确标记 inactive。
力部分压缩为一行 `force_yaw`，只显示目标力、实际虚拟弹簧力和 XYZ 逐轴绝对误差，便于
连续观察。接触反力不会被误当成训练中的实际力。`arm_q` 按启动时 `arm_q_order` 的顺序
排列。这样可以直接判断右摇杆改变了哪个命令、策略是否让末端跟随。ZGWSARM 启动时会明确打印
`gripper=unavailable`；其部署动作维度为
16 个四足关节加 6 个机械臂关节，当前模型没有夹爪关节或执行器，不能通过映射产生闭合动作。

ZGWSARM 状态还会分别打印 `lin_vel_x` 和 `yaw_rate` 的目标值、底盘局部坐标系实际值及
绝对误差，便于直接判断前进和持续转向是否真正跟踪命令。

## 5. 关键一致性约束

- reset 后策略第一次输入是全零 10 帧历史，第一物理步后才追加真实观测；
- 观测顺序固定为 gravity、23 commands、q-default、qd、last action、4 clocks；
- ZGWSARM 腿为位置 PD，轮子为直接力矩，机械臂为带速度限幅的位置目标；
- B1+Z1 使用 19 关节位置 PD，并保留训练端闭合夹爪的 action baseline；
- 每物理步执行 URDF 硬位置/速度约束，以复现 PhysX articulation 合同；
- 任意 NaN/Inf、明显限位或速度越界都会触发 safety stop。

场景由 `scripts/generate_mujoco_models.py` 确定性生成。修改 URDF/MJCF 后应重新生成并
运行全部测试：

```bash
python scripts/generate_mujoco_models.py
python -m unittest tests.test_mujoco_deployment -v
ctest --test-dir mujoco/build --output-on-failure
```

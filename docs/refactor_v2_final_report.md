# learning-compliance 重构 V2 最终执行报告

## 结论

V2 方案内规划的框架整理已经全部落地。当前训练主链路为：

```text
scripts/train.py
  -> TaskRegistry
  -> B1Z1Cfg / B1Z1CfgPPO
  -> B1Z1Env + HistoryWrapper
  -> OnPolicyRunner
  -> PPO
  -> ActorCritic + AdaptationModule
  -> RolloutStorage
```

Environment 侧已经形成 `rewards`、`sensors`、`commands`、`curriculum` 四个稳定边界。
控制器、力注入、力矩计算、domain randomization 和 PhysX step 仍保留在
`LeggedRobot`，符合 V2 中“不大拆仿真核心”的约束。

## 最终代码归属

```text
b1_gym/
├── commands/commands.py       # 23 维 schema、采样、EE 轨迹、gait/curriculum lifecycle
├── curriculum/curriculum.py   # curriculum 实现
├── rewards/b1_z1_rewards.py   # B1+Z1 全部 reward
├── sensors/sensors.py         # 全部 28 个 sensor 与 catalog
├── envs/b1_z1/                # task 与 Environment/PPO config
└── envs/base/legged_robot.py  # 仿真、reset、control、force、randomization 主体

b1_gym_learn/
├── runners/on_policy_runner.py
├── algorithms/ppo_cse.py
├── modules/{actor_critic,adaptation_module}.py
├── storage/rollout_storage.py
└── logging/experiment_logger.py
```

旧的 reward、sensor、curriculum 和 `b1_gym_learn.ppo_cse.*` 模块路径均保留为
兼容导入，并指向新入口中的同一个类对象，不复制第二套实现。

## 本次最终批次完成内容

### Command lifecycle

- 将 `_resample_commands()`、command curriculum 更新、category/bin 状态、heading
  command、gait phase 后处理、command sum reset 和 `_step_contact_targets()` 移入
  `CommandLifecycleMixin`。
- 将 EE 球坐标测量、姿态测量、轨迹插值、可行性检查和 position/force mode 重采样
  移入同一 command 边界。
- `LeggedRobot` 通过 mixin 使用原有环境 buffer；没有 manager 中转，也没有额外 tensor
  copy，因此调用顺序和 RNG 消耗顺序保持不变。
- 保留原实现中 fixed gait command 对最后一个 category 的历史索引行为；本轮不夹带算法
  修复。

### Sensor 收敛

- 28 个 sensor 的实现统一到 `b1_gym/sensors/sensors.py`。
- 原 29 个小模块变为兼容 wrapper；例如旧路径
  `b1_gym.sensors.orientation_sensor.OrientationSensor` 与新 catalog 对象严格相同。
- Environment、play 和 test 的正式依赖改为 `b1_gym.sensors` 公共入口。
- sensor 名称、构造参数、observation 顺序、noise vector 和 privileged observation 顺序
  均未修改。

### Artifact 与兼容加载

- 新训练只使用 run-owned 的 `checkpoints/model_*.pt`、`model_latest.pt` 和
  `exported/`。
- Runner 每次保存 checkpoint 时同步生成 `exported/policies/policy_<iteration>.pt`
  和 `policy_latest.pt`；checkpoint 继续保存完整训练状态，export 只用于 inference。
- W&B 评估优先下载 run 中的 canonical checkpoint 并构造完整 ActorCritic；老 run 的
  split JIT 文件只作为兼容 fallback。
- 本地历史 checkpoint、旧 config schema 和运行期已乘 `dt` 的 reward config 仍可加载。

## 冻结契约

- Environment config SHA-256：
  `5bbc3f75471ac679952e1b6029a0639fe35e0513957d5f85426665f08db751d7`
- Training config SHA-256：
  `3971bb0ad9795963582e9f46a46121722055b5390850bee43e38798b6596df9b`
- Command curriculum grid + seed=100 首批 sample SHA-256：
  `44e68ff3ab777f43aaace74a9901bb962e0f2f4614a7e59b9620231552be09a2`
- Observation / privileged / history / action / command 维度：
  `87 / 16 / 870 / 19 / 23`
- ActorCritic state-dict、参数量、旧 checkpoint strict-load 和 TorchScript 输出契约继续
  由测试覆盖。

## 最终验证结果

### 自动化测试

- `29/29` 项单元与结构契约测试通过。
- 全部 Python 文件通过 `compileall`。
- `git diff --check` 通过。

### 固定 seed GPU 回放

固定条件：`model_005000.pt`、seed `7`、`8` environments、`600` steps、binary mode、
force amplitude `70 N`。该长度覆盖约 step 500 的 command resampling。

最终 JSON 与 command lifecycle 抽取前的冻结 JSON 对比：

```text
settings_equal = True
config_load_equal = True
metrics_equal = True
metric_diffs = []
```

也就是说所有 metric 的 `mean`、`maximum` 和 `samples` 都逐值相等。主要结果为：

| 指标 | 数值 |
|---|---:|
| reward/env-step | 0.0567226671 |
| reset rate | 0 |
| EE position error | 0.0889228433 m |
| EE orientation error | 0.1105821133 rad |
| EE XY force error | 8.360039711 N |
| EE Z force error | 12.038392067 N |
| active X/Y/Z force error | 9.590505 / 9.354184 / 16.506735 N |
| joint torque RMS | 21.88768768 N·m |
| peak absolute torque | 187.5082397 N·m |

### GPU 训练链路

最终代码完成 `1 environment × 48 rollout steps × 1 PPO/adaptation update`：

```text
run: logs/b1_z1_ik/2026-08-13_11-25-13_refactor_v2_final_export_smoke
total timesteps: 48
model_000000.pt: saved
model_000001.pt: saved
model_latest.pt: saved
exported/policies/policy_000000.pt: saved
exported/policies/policy_000001.pt: saved
exported/policies/policy_latest.pt: saved
process exit: 0
```

## 验收判断

从框架等价性角度，V2 重构可以验收。当前没有证据表明此次代码归属调整改变了 policy
输入、reward、command 分布、PPO 更新或控制输出。

下一步建议停止继续做结构修改，进入一次独立的训练回归。单个新长训练可用于工程确认；
若要对训练效果作统计结论，应使用相同超参数完成至少 3 个独立 run 对比旧版本，并继续
使用 `round1_closure_evaluation_protocol.md` 中的 position/force 评估矩阵。当前训练入口
没有统一的全局 seed 参数，本轮没有为了复现实验而新增 seed 初始化，以免改变旧训练的
RNG 行为；这可以作为后续独立、显式评审的实验基础设施改动。

## 保留的原始问题

启动仍会报告：

```text
reward _reward_dof_pos_limits has nonzero coefficient but was not found
```

这是重构前已有的配置/实现不一致。补函数或删除权重都会改变 reward 语义，因此不属于
V2 结构重构，未在本轮修改。Gym、`torch.meshgrid` 和 terrain tensor copy 的提示同样是
依赖或旧实现告警，不影响本次等价性结论。

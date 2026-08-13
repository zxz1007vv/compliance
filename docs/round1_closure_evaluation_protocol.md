# 第一轮重构收尾与复测规程

## 结论

第一轮框架重构的训练链路可以进入冻结候选状态。当前不需要因为收尾修复重新训练
`model_005000.pt`：修复的是配置保存/加载与评估统计，不是 policy、observation、reward
公式、PPO 更新或控制计算。

此前新 run 的 `play` 输出中，`reward/env-step` 约为 `0.0011`，主要原因不是策略退化，
而是训练配置在环境初始化后保存，reward 系数已乘过一次 `dt=0.02`；`play` 加载后又乘
一次。现已完成以下闭环：

- 环境使用独立的运行期 reward 字典，不再原地修改 `cfg.reward_scales`。
- TaskRegistry 在环境初始化前保存原始配置，并写入 `ConfigMeta` 和配置指纹。
- 历史污染 run 自动检测整体 `dt` 缩放并恢复；本次 run 检出 `31/31` 个非零系数
  匹配 `0.02` 缩放。
- `play.py` 用 `--control-mode` 限定合法模式，`postion` 之类拼写错误会在启动前报错。
- 评估固定 Python、NumPy、PyTorch/CUDA 与 command curriculum seed。
- 每次评估自动保存 JSON，并增加 active-force 条件下的 X/Y/Z 逐轴误差。
- plane 多环境评估的 CPU/CUDA 网格设备不一致已修复。

旧 run 的 reward 加载修复只改变 rollout 中报告的 reward 数值。policy 动作、位置误差、
姿态误差、力误差和力矩统计不依赖 play reward，因此此前这些跟踪观测仍可参考。

## 已执行验证

- `20` 项单元/契约测试全部通过。
- 同一 `model_005000.pt` 完成 position 模式：8 environments × 100 steps，0 reset，
  JSON 正常生成，reward 均值恢复为 `0.0458`。
- 同一 checkpoint 完成 force 模式：8 environments × 150 steps，0 reset，JSON 正常生成；
  active-force X/Y/Z 指标在与训练 reward 相同的机身 yaw 局部坐标系计算。
- 完成 1 environment × 48 rollout steps × 1 PPO/adaptation update 的 GPU 短训练；
  `model_000000.pt` 和 `model_000001.pt` 正常生成。
- 短训练保存的配置确认原始权重为 `manip_pos_tracking=3.0`、`ee_force_z=3.0`，
  零 reward 项保留，运行期派生字段未写入。

短冒烟包含启动瞬态，不能代替下述正式 2000-step 统计，也不用于最终比较策略优劣。

## 复测前准备

从仓库根目录执行：

```bash
cd /home/ubuntu/zskj/learning-compliance
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate compliance
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/compliance/lib:/usr/local/cuda/lib64
export TORCH_EXTENSIONS_DIR=/tmp/learning_compliance_torch_extensions
export MPLCONFIGDIR=/tmp/learning_compliance_matplotlib
```

以下命令中的 run 固定为：

```bash
RUN_DIR=logs/b1_z1_ik/2026-08-12_17-09-53_wbc_release
```

## 先做单环境可视化

position 模式：

```bash
python scripts/play.py \
  --run-dir "$RUN_DIR" \
  --checkpoint 5000 \
  --control-mode position \
  --seed 1 \
  --num-envs 1 \
  --steps 2000 \
  --viewer \
  --print-every 100
```

force 模式（显式固定训练使用的 `[-70, 70] N` 目标范围）：

```bash
python scripts/play.py \
  --run-dir "$RUN_DIR" \
  --checkpoint 5000 \
  --control-mode force \
  --force-amplitude 70 \
  --seed 1 \
  --num-envs 1 \
  --steps 2000 \
  --viewer \
  --print-every 100
```

可视化阶段重点记录：是否跌倒或明显抖动、末端是否持续偏离目标、force 变化时是否出现
发散/饱和，以及异常开始的大致 step。需要录像时添加 `--record-video`。

## 正式无头统计

单环境轨迹受一次随机命令影响较大。正式判断建议每个模式运行 3 个 seed，每次使用
32 个环境和 2000 steps，即每个 seed 产生 64,000 个 env-step 样本。

position：

```bash
python scripts/play.py --run-dir "$RUN_DIR" --checkpoint 5000 --control-mode position --seed 1 --num-envs 32 --steps 2000 --headless --print-every 0
python scripts/play.py --run-dir "$RUN_DIR" --checkpoint 5000 --control-mode position --seed 2 --num-envs 32 --steps 2000 --headless --print-every 0
python scripts/play.py --run-dir "$RUN_DIR" --checkpoint 5000 --control-mode position --seed 3 --num-envs 32 --steps 2000 --headless --print-every 0
```

force：

```bash
python scripts/play.py --run-dir "$RUN_DIR" --checkpoint 5000 --control-mode force --force-amplitude 70 --seed 1 --num-envs 32 --steps 2000 --headless --print-every 0
python scripts/play.py --run-dir "$RUN_DIR" --checkpoint 5000 --control-mode force --force-amplitude 70 --seed 2 --num-envs 32 --steps 2000 --headless --print-every 0
python scripts/play.py --run-dir "$RUN_DIR" --checkpoint 5000 --control-mode force --force-amplitude 70 --seed 3 --num-envs 32 --steps 2000 --headless --print-every 0
```

未指定 `--output` 时，结果自动写入：

```text
<run-dir>/evaluations/<timestamp>_<policy>_<mode>_seed<seed>.json
```

需要与重构前 checkpoint 做严格 A/B 时，对旧 run 使用完全相同的 6 条命令，只替换
`--run-dir` 和 `--checkpoint`。不要用旧日志中不同 seed、不同环境数或不同 force 范围的
一次 rollout 直接做百分比比较。

## JSON 中重点字段

- `config_load.legacy_reward_scale_repaired`：本次历史 run 应为 `true`；新训练 run 应为
  `false`。
- `settings`：核对 checkpoint、mode、seed、环境数、steps 和 force range。
- `metrics.reset.mean`：env-step reset rate。
- `metrics.reward.mean`：修复后的每 env-step reward。
- position：`ee_position_error.mean`、`ee_orientation_error.mean`。
- force：`ee_xy_force_error.mean`、`ee_z_force_error.mean`。
- force 逐轴：`ee_x/y/z_force_abs_error_active.mean`。该指标只统计目标合力大于 1 N 的
  active-force 样本，避免大量零命令稀释误差。
- `active_force_command.mean`：有效 force 命令样本占比；比较两次实验前应先确认其接近。
- `joint_torque_rms.mean` 和 `joint_torque_abs.maximum`：力矩均值与峰值。

## 第一轮冻结建议阈值

将相同 seed 的新旧 JSON 配对，再对 3 个 seed 的相对变化取平均。建议满足：

- 不出现 NaN/Inf、仿真异常退出或肉眼可见持续发散。
- reset rate 不高于旧版 `0.5` 个百分点；若旧版为 0，新版也应接近 0。
- reward 均值相对变化在 `±10%` 内。
- position error 不恶化超过 `10%`，且绝对增量不超过 `0.01 m`。
- orientation error 不恶化超过 `15%`，且绝对增量不超过 `0.02 rad`。
- active-force X/Y/Z 任一轴不恶化超过 `15%` 或 `2 N`；Z 轴单独判定，不再只看
  XY 合成误差。
- torque RMS 不增加超过 `10%`；peak torque 若异常增加超过 `15%`，需要结合录像定位。

这些是本项目第一轮工程重构的冻结门槛，不是算法性能上限。如果某项越界但 3 个 seed
方向不一致，应先扩大 seed，而不是立即调整 reward 或 policy。

## 需要回传的材料

最低材料：

1. 新 run 的 6 个 evaluation JSON。
2. 对应 run 的 `config.json`、checkpoint 编号和执行时的 git commit/status。
3. position/force 各一段可视化观察；有异常时附视频和首次异常 step。
4. TensorBoard 中最后 10% 训练区间的 mean reward、episode length、value loss、
   adaptation loss、policy noise std。

若要做严格新旧 A/B，再提供旧 run 的同一组 6 个 JSON 和旧 `config.json`。收到这些后
再决定是否冻结第一轮并进入第二轮代码整理；不要在基线采集期间调整 reward、控制参数
或 domain randomization。

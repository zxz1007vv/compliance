# 第二步 Environment 边界重构执行报告

## 结论

第一轮正式复测没有发现阻塞性异常，第一轮框架重构基线已冻结。本文记录第二步前两批
Environment 边界整理；后续 command lifecycle 与 sensor 收敛已完成，最终状态见
[refactor_v2_final_report.md](refactor_v2_final_report.md)。

本批只改变代码归属与创建入口，不改变 reward 函数体、command 向量布局、随机采样
操作、sensor 实现、curriculum 算法、observation 顺序或控制器。

注：本文记录第二阶段当时的渐进迁移状态。最终包名与唯一正式入口以
[refactor_v2_final_report.md](refactor_v2_final_report.md) 为准；阶段性兼容转发模块已在
最终目录收敛时删除。

## 第一轮正式复测验收

正式统计采用 `32 environments × 2000 steps × 3 seeds`。目录中另有一份 position
单环境可视化 JSON，不计入三 seed 汇总。

| 模式 | 指标 | 3-seed 均值 | seed 范围 |
|---|---:|---:|---:|
| position | reward/env-step | 0.056414 | 0.056257–0.056555 |
| position | reset rate | 0 | 0–0 |
| position | EE position error | 0.075384 m | 0.072171–0.077898 m |
| position | EE orientation error | 0.099406 rad | 0.098754–0.100238 rad |
| position | torque RMS | 20.518426 N·m | 20.275257–20.856169 N·m |
| force | reward/env-step | 0.073339 | 0.073191–0.073612 |
| force | reset rate | 0.0036% | 0.0016%–0.0047% |
| force | EE XY force error | 9.296098 N | 8.980330–9.474693 N |
| force | EE Z force error | 19.455587 N | 18.494152–20.144894 N |
| force | active X/Y/Z error | 13.412 / 7.741 / 25.230 N | 跨 seed CV 3.31% / 3.79% / 4.77% |
| force | torque RMS | 23.626757 N·m | 23.463551–23.883677 N·m |

所有 JSON 数值有限，无 NaN/Inf。position 三个 seed 均无 reset；force 总体 reset 极低。
各主要指标的跨 seed 变异系数约为 `0.22%–4.77%`。Z force error 明显高于 XY，但
三个 seed 同方向且稳定，判定为当前策略特征，列入算法优化观察项，不属于框架重构异常。

force peak torque 三个 seed 均约为 `210 N·m`，表现为一致的上限触达。第二步不修改
torque limit、reward 或 controller，仅继续记录该现象。

本次 run 是第一轮早期生成的 schema，配置内 reward scale 已经乘过 `dt=0.02`。所有
正式评估均由兼容加载器成功识别 `31/31` 个非零系数并恢复；原 run 的 `config.json`
没有被修改。

## 本批实施内容

### Command 边界

新增 `wbc_compliance_gym/commands/commands.py`：

- 固化 23 维 command 向量的全部索引；
- 统一合法 control modes；
- 集中 position / force / binary / mixed 的采样操作；
- `LeggedRobot`、`play.py` 和 `scripts/utils.py` 使用同一个契约。

相同 seed 下，mixed 仍调用一次 `torch.rand`，binary 仍调用一次 `torch.randint`；测试已
确认输出逐元素一致。

### Reward 边界

任务 reward 的正式实现入口改为：

```text
wbc_compliance_gym/rewards/b1_z1.py::B1Z1Rewards
```

历史类名和模块路径继续可用，并且是同一个类对象：

```text
B1LocoZ1GaitfreeRewards is B1Z1Rewards
```

配置中的 `reward_container_name="B1LocoZ1GaitfreeRewards"` 暂不修改，因此环境配置
指纹和历史 run 均保持兼容。环境通过 reward registry 创建任务 reward，不再在
`legged_robot.py` 内硬编码 import。

### Sensor 与 Curriculum 边界

- 新增 `wbc_compliance_gym/sensors/sensors.py` 作为统一 sensor catalog 与创建入口。
- curriculum 正式入口迁移到 `wbc_compliance_gym/curriculum/curriculum.py`。

这一步先统一依赖方向，没有为了减少文件数机械合并 28 个 sensor 类。

## 等价性验证

- Python `compileall` 通过。
- 单元/契约测试由 20 项增加到 24 项，全部通过。
- 环境配置与训练配置 SHA-256 契约保持不变。
- command 索引严格等于 `0..22`，维度仍为 23。
- mixed/binary 采样与原始 PyTorch 操作在固定 seed 下逐元素相同。
- reward、curriculum 的旧导入与新实现是同一对象。
- 1 environment × 48 rollout steps × 1 PPO/adaptation update 的 GPU 训练通过，
  checkpoint `model_000000.pt` / `model_000001.pt` 正常保存。
- 使用 `model_005000.pt` 执行 8 environments × 150 steps force play。与第二步前
  相同条件的 JSON 对比结果：所有 metric 的 mean、maximum、samples 数值差异为空。

## 第一批当时刻意未修改

- 不搬动完整 `_resample_commands()`，避免同时触碰 curriculum、category、gait 和
  heading 的 RNG 顺序。
- 不搬动 `_compute_torques()`、PD control 或 force injection。
- 不修改任何 reward 函数、权重和 reward container 配置字符串。
- 不调整 Z force tracking 或 210 N·m peak torque。
- 不合并 sensor 实现文件；先稳定 catalog，再决定是否有实际维护收益。
- 不处理历史 `_reward_dof_pos_limits` 告警，因为修复它会改变算法语义。

## 第二批：Command Distribution Schema

第二步第二批已继续完成：

- 新增不可变 `CommandDimension` 和有序 `COMMAND_SCHEMA`，统一描述全部 23 个 command
  slot 的索引、curriculum key、limit/bin 配置来源和 active range 来源。
- 新增 `command_curriculum_ranges()` 与 `command_active_bounds()`，替代
  `LeggedRobot` 中两份容易发生顺序漂移的手写 23 维列表。
- 新增 `build_command_curricula()`，负责 curriculum 类型解析、category 创建、实例化和
  active bin 初始化。
- `_init_command_distribution()` 现在只保留环境运行状态的归属：接收 factory 结果，
  创建 `env_command_bins`、`env_command_categories` 和 GPU control-mode buffer。
- 未知 curriculum 类型现在显式抛出 `ValueError`，避免旧实现中未绑定局部变量导致的
  不透明错误；当前任务配置仍使用原 `RewardThresholdCurriculum`。

### 第二批数值契约

修改前先冻结以下对象：curriculum `grid`、`idx_grid`、`weights`、`lows/highs`、类别、
keys、bin shape，以及 seed=100 的首次 64 次 `sample()` 输出。组合 SHA-256 为：

```text
44e68ff3ab777f43aaace74a9901bb962e0f2f4614a7e59b9620231552be09a2
```

提取后哈希完全相同，并已写入自动化测试。测试总数由 24 增至 26，全部通过。

GPU 再次完成 1 environment × 48 steps × 1 PPO/adaptation update。随后使用同一
`model_005000.pt`、seed=1、8 environments、150 steps 的 force play，与第一批
Environment 边界重构后的 JSON 比较：`settings` 相同，全部 metric 的 mean、maximum、
samples 精确相等。

## 后续批次执行结果

原计划的下一批已经执行：

1. `_resample_commands()`、gait phase、heading 和 command-sum reset 已移入 command
   boundary，并维持原调用和 RNG 顺序。
2. EE command 插值、测量和 position/force mode lifecycle 已一并迁移。
3. 28 个 sensor 实现已收敛到单文件，并统一使用公共 catalog 入口。
4. force push、controller、torque 与 physics step 继续留在 `LeggedRobot`。

最终固定 seed、600-step 回放覆盖 command resampling，全部 JSON metric 与迁移前逐值
一致；最终 GPU PPO/adaptation 更新也已通过。完整证据见最终报告。

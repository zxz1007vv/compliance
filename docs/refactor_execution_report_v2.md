# 训练框架重构执行报告 V2

## 结论

本轮已完成低风险框架重构，训练主链路收敛为：

```text
scripts/train.py
  -> TaskRegistry
  -> B1Z1Env + HistoryWrapper
  -> OnPolicyRunner
  -> PPO
  -> ActorCritic + AdaptationModule
  -> RolloutStorage
```

重构采用 HIMLoco 的 Task / Config / Runner 组织思路，但保留了本仓库原有的
whole-body compliance 算法、观测、奖励、命令、随机化和 PPO 更新逻辑。

## 已执行内容

1. 将 `ppo_cse` 按职责机械拆分为：
   `algorithms/`、`modules/`、`runners/`、`storage/`、`logging/`。
2. 保留 `b1_gym_learn.ppo_cse.*` 兼容层，旧脚本和历史 checkpoint 无需改名。
3. 将 `scripts/train.py::configure_env()` 的任务配置迁移到
   `b1_gym/envs/b1_z1/b1_z1_config.py`，形成隔离的 `B1Z1Cfg` 和
   `B1Z1CfgPPO`。
4. 新增 `TaskRegistry`，统一 task、环境、wrapper、配置和 Runner 的创建。
5. 将训练入口缩减为参数解析、Registry 组装和 `runner.learn()`。
6. 统一本地 artifact 结构，并同时识别新旧 checkpoint 布局。
7. checkpoint 增加完整配置、Runner 状态与 CPU/CUDA/NumPy/Python RNG 状态。
8. W&B 改为可选依赖；TensorBoard 仍为默认日志后端。
9. 新增独立 `scripts/export_policy.py`，无需创建仿真环境即可导出 TorchScript。

## 等价性保护

以下训练契约已固化为自动化测试：

- 环境配置 SHA-256：
  `5bbc3f75471ac679952e1b6029a0639fe35e0513957d5f85426665f08db751d7`
- 训练配置 SHA-256：
  `3971bb0ad9795963582e9f46a46121722055b5390850bee43e38798b6596df9b`
- ActorCritic state-dict 结构 SHA-256：
  `b5c1d0bb801d0cde86c11b843ae5b39b31431f2f9069539157bb203cde90716a`
- ActorCritic 参数量：`1,497,271`
- 维度：observation `87`、privileged observation `16`、history `870`、
  action `19`、command `23`
- adaptation labels / dimensions / weights 保持原值
- 旧导入路径与新实现为同一 Python 对象
- 历史 raw state-dict 和完整 checkpoint 均可 strict load
- 显式 policy config 与旧全局 config 的初始化和前向输出逐张量一致
- 导出 TorchScript 与 Python inference 输出一致

## 验证结果

- 标准库单元测试：`20` 项通过。
- 全部 Python 文件通过 `compileall`。
- 使用历史 `model_4999.pt` 完成独立 TorchScript 导出。
- GPU 短训练：`1` 个环境、`48` 个 rollout step、`1` 次 PPO/adaptation
  更新，完整训练链路返回码 `0`。
- 短训练 run 正确生成 `config.json`、`model_000000.pt`、
  `model_000001.pt` 和 `model_latest.pt`。

## 第一轮测评后的收尾修复

第一轮模型训练曲线与重构前基本一致，训练主链路可判为等价候选。play 测评进一步发现
配置在环境初始化后保存会包含已经乘过 `dt` 的 reward scale，导致加载时再次缩放。
该问题只污染 play 报告的 reward，不改变已经训练出的 policy，也不改变此前记录的位置、
姿态、力跟踪和力矩指标。

现已改为保存环境初始化前的原始配置快照，并隔离环境的运行期 reward 字典。历史污染
run 由加载器按多系数一致比例自动检测并恢复。本次
`2026-08-12_17-09-53_wbc_release` 实测 `31/31` 个非零系数匹配 `dt=0.02`。

同时完成：

- `play.py --control-mode {position,force,binary,mixed}`，不再编辑源码切模式；
- 固定 seed、可选 force amplitude、自动 evaluation JSON；
- active-force X/Y/Z 逐轴误差，且坐标系与训练 reward 一致；
- plane 地形多环境 CUDA 网格修复；
- 新 run 配置 schema、保存阶段和 SHA-256 元数据。

正式复测命令、阈值与回传清单见
[round1_closure_evaluation_protocol.md](round1_closure_evaluation_protocol.md)。

## Artifact 约定

```text
logs/<task>/<timestamp>_<run-name>/
├── config.json
├── tensorboard/
├── checkpoints/
│   ├── model_000000.pt
│   ├── model_000400.pt
│   └── model_latest.pt
└── exported/
    └── policies/
```

读取逻辑仍兼容历史 run 根目录下的 `model_<iteration>.pt`，不会迁移或删除旧实验。

## 本轮刻意不做的改动

- 不修改 reward 公式、权重或调用顺序。
- 不修改 observation / privileged observation / history 的拼接顺序。
- 不修改 command 采样、curriculum、domain randomization 或 control 流程。
- 不修改 PPO loss、mini-batch、optimizer step 或 adaptation loss。
- 不大拆 `legged_robot.py`，也不合并现有 sensor 文件。
- 不引入 HIM estimator、prototype、sinkhorn 等参考仓库算法。

这些内容若继续整理，需要先增加固定 seed 的短训练轨迹对照，再单独分批执行。

## 已知原始告警

GPU 冒烟测试仍会报告：

```text
reward _reward_dof_pos_limits has nonzero coefficient but was not found
```

该告警来自重构前已有的 reward 配置/实现不一致。本轮未修正它，因为直接补函数或改
权重会改变训练算法，应作为独立问题评审。

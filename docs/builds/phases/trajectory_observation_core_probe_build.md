# trajectory_observation_core_probe 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段验证轨迹观测是否提供与 endpoint evidence 不同源的路径证据, 并将路径证据升级为时间重参数化不变的 Flow Matching trajectory observation。

### 1.2 输入

```text
configs/protocol/trajectory_observation_core.json
configs/trajectory/trajectory_observation.json
configs/trajectory/trajectory_controls.json
configs/trajectory/trajectory_time_grid.json
main/trajectory/
experiments/trajectory_observation_core/
```

### 1.3 构建任务

1. 捕获或构造 trajectory trace。
2. 计算 velocity projection 与 key-conditioned path response。
3. 构造 `time_reparameterization_invariant_path_observation`。
4. 与 endpoint evidence 计算独立性和互补性。
5. 构造 wrong key、wrong sampler、time shuffled、trajectory-only 和 endpoint-only controls。
6. 将 trajectory evidence 接入 state-space posterior。

### 1.4 必须 baseline / control

```text
endpoint_only_control
trajectory_only_score
trajectory_time_shuffled_control
wrong_key_trajectory_control
wrong_sampler_replay_control
explicit_temporal_alignment_baseline
```

### 1.5 通过标准

1. path evidence 与 endpoint evidence 不应高度冗余。
2. time-reparameterization-invariant path score 在不同 time grid 下保持可比。
3. wrong sampler replay 不能获得与 correct replay 等价的路径证据。
4. trajectory-only score 不能绕过 fixed-FPR 和 admissibility 直接支撑最终判定。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有 trajectory observation 相关模块:

```text
main/trajectory/trajectory_observation.py
main/trajectory/trajectory_reconstruction.py
main/trajectory/trajectory_runtime.py
main/trajectory/trajectory_statistic.py
main/trajectory/trajectory_trace.py
main/trajectory/velocity_projection.py
experiments/trajectory_observation_core/runner.py
experiments/trajectory_observation_core/trajectory_builder.py
experiments/trajectory_observation_core/control_runner.py
experiments/trajectory_observation_core/correlation_audit.py
experiments/trajectory_observation_core/mechanism_audit.py
```

### 2.2 当前阶段补充要求

按照新的整体流程, 该阶段需要显式覆盖:

```text
time_reparameterization_invariant_path_observation
wrong_sampler_replay_control
trajectory_only_score
endpoint_only_control
```

这些要求用于证明 SSTW 不是固定 step-index 对齐, 也不是 endpoint-only 或 trajectory-only 分数堆叠。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 部分完成 |
| 主要差距项 | path evidence 独立性和 fixed-FPR marginal gain 仍需 pilot_paper / full_paper validation。 |
| 下一步构建方向 | 强化 endpoint/path redundancy audit、wrong sampler replay control 和 trajectory-only safety。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 部分完成
gap_item: path evidence 独立性和 fixed-FPR marginal gain 仍需 pilot_paper / full_paper validation。
next_action: 强化 endpoint/path redundancy audit、wrong sampler replay control 和 trajectory-only safety。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

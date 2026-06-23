# replay_and_authenticated_sketch_gate 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段规范 owner-side trajectory audit、model-side replay verification 和 video-only proxy observation 的证据等级, 防止 SSTW 被误解为未认证服务端日志或普通 replay 后处理。

### 1.2 输入

```text
main/core/digest.py
main/core/manifests.py
main/trajectory/trajectory_trace.py
main/generation/trajectory_capture.py
experiments/trajectory_observation_core/
experiments/generative_video_model_probe/
```

### 1.3 构建任务

1. 定义 authenticated trajectory sketch 的记录、摘要和验证状态。
2. 为 replay trajectory 记录 scheduler、time grid 和 uncertainty weight。
3. 定义 wrong sampler replay control。
4. 将 evidence level 写入 manifest 和 claim audit。
5. 限制 video-only proxy 的论文主张边界。

### 1.4 证据等级

```text
level_1_authenticated_cached_trajectory
level_2_model_side_replay_with_uncertainty
level_3_video_only_proxy_observation
```

### 1.5 必须字段

```text
authenticated_trajectory_sketch_status
trajectory_sketch_digest_random
trajectory_sketch_verification_status
replay_uncertainty_weight
replay_scheduler_id
replay_time_grid_id
wrong_sampler_replay_control
```

### 1.6 通过标准

1. level 1 可以支撑高置信 owner-side trajectory audit。
2. level 2 必须记录 replay uncertainty。
3. level 3 只能作为补充证据, 除非经过 fixed-FPR 独立验证。
4. 未认证 trajectory logging 只能作为 control。

### 1.7 Claim-3 降级规则

若本阶段不能闭合, Claim-3 必须降级, 不得写成完全鲁棒 replay verification。

```text
authenticated_sketch_not_ready -> owner-side audit claim 降级为 deployment protocol sketch
replay_uncertainty_not_ready -> model-side replay claim 降级为 exploratory replay analysis
wrong_sampler_replay_not_ready -> 不声明 sampler-mismatch robustness
wrong_prompt_replay_not_ready -> 不声明 prompt-mismatch robustness
video_only_proxy_not_validated -> 不声明 black-box video-only trajectory detection
```

### 1.8 必须输出的审计 artifacts

```text
records/trajectory_sketch_verification_records.jsonl
records/replay_uncertainty_records.jsonl
records/wrong_sampler_replay_records.jsonl
records/wrong_prompt_replay_records.jsonl
tables/replay_verification_table.csv
reports/replay_and_sketch_gate_report.md
artifacts/replay_and_sketch_gate_decision.json
```

## 2. 当前阶段具体完成情况

### 2.1 已有工程基础

仓库中已经存在 digest、manifest、trajectory trace 和 trajectory capture 基础模块, 可作为 authenticated sketch 与 replay uncertainty 的实现基础。

### 2.2 当前阶段补充要求

该阶段需要在后续工程中进一步补齐 authenticated trajectory sketch 的正式字段、manifest 记录和 checker 规则。未认证 logging 不得直接支撑 supported claims。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 未完成 |
| 主要差距项 | authenticated sketch、replay uncertainty、wrong prompt replay 与 checker 未闭合。 |
| 下一步构建方向 | 补齐 sketch 签名验证、replay uncertainty records、wrong sampler / wrong prompt replay control。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 未完成
gap_item: authenticated sketch、replay uncertainty、wrong prompt replay 与 checker 未闭合。
next_action: 补齐 sketch 签名验证、replay uncertainty records、wrong sampler / wrong prompt replay control。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

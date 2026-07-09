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

最新 small-scale pilot 已经写出 replay uncertainty、wrong-sampler replay control 和 runtime detection 相关记录, 可以作为 probe_paper 继续扩展的工程入口。但这些记录仍属于 pilot / proxy / runtime workflow 证据, 不能替代 authenticated trajectory sketch 或论文级 replay posterior gate。

### 2.2 当前阶段补充要求

该阶段需要在后续工程中进一步补齐 authenticated trajectory sketch 的正式字段、manifest 记录和 checker 规则。未认证 logging 不得直接支撑 supported claims。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 未完成 |
| 主要差距项 | pilot 级 replay uncertainty 已有记录, 但 authenticated sketch、wrong prompt replay、replay/sketch checker 与 full_paper evidence level 仍未闭合。 |
| 下一步构建方向 | 在 probe_paper 补齐 sketch 签名验证、replay uncertainty records、wrong sampler / wrong prompt replay control 和 gate decision。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 未完成
gap_item: pilot 级 replay uncertainty 已有记录, 但 authenticated sketch、wrong prompt replay、replay/sketch checker 与 full_paper evidence level 仍未闭合。
next_action: 在 probe_paper 补齐 sketch 签名验证、replay uncertainty records、wrong sampler / wrong prompt replay control 和 gate decision。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

### 3.2 2026-06-23 最新阶段边界

当前项目可以继续推进 replay / sketch 工程实现, 但必须保持以下 claim 边界:

```text
method_mechanism_validation_passed = true
pilot_replay_uncertainty_record_available = true
wrong_sampler_replay_control_available_at_pilot_level = true
authenticated_trajectory_sketch_status = not_ready
trajectory_sketch_verification_status = not_ready
wrong_prompt_replay_records_ready = false
replay_and_authenticated_sketch_gate_closed = false
claim3_full_support_allowed = false
```

因此, Claim-3 目前只能表述为“小规模 pilot 中 replay uncertainty 与 wrong-sampler control 已进入 governed records”, 不能表述为“攻击后视频均可通过 authenticated replay 恢复轨迹后验”。

### 3.3 最终创新性要求

短期可以通过 `Claim-3 downgrade gate` 保护 probe_paper 流程不被 unsupported claim 阻断, 但这只是 claim 边界收缩, 不是本阶段完成。若项目最终要把 replay posterior / authenticated sketch 作为足够强的创新性贡献, 必须实现并通过 `replay/sketch gate`。

该要求在项目构建流程中对应如下边界:

```text
probe_paper_short_term: claim3_downgrade_gate_allowed
full_paper_strong_claim: replay/sketch gate_required
top_tier_innovation_claim: replay/sketch gate_required
```

`replay/sketch gate` 完成前, full_paper 或投稿叙述不得把 Claim-3 写成强 supported claim。

## 2026-06-24 validation proxy runner 接入

当前已新增 replay/sketch gate 的 probe_paper 工程入口:

```text
experiments/generative_video_model_probe/replay_and_sketch_gate.py
records/trajectory_sketch_verification_records.jsonl
records/replay_uncertainty_records.jsonl
records/wrong_sampler_replay_records.jsonl
records/wrong_prompt_replay_records.jsonl
tables/replay_verification_table.csv
artifacts/replay_and_sketch_gate_decision.json
reports/replay_and_sketch_gate_report.md
```

该 runner 从 `generation_records.jsonl` 与 `trajectory_trace.jsonl` 构造 authenticated trajectory sketch digest、replay uncertainty weight、wrong sampler replay control 和 wrong prompt replay control。它不读取 `S_final` 或最终检测判定分数来决定样本是否可用, 因此不会把最终检测结果反向用于污染过滤。

当前 evidence level 为:

```text
replay_and_sketch_evidence_level: validation_runtime_trace_proxy
claim_support_status: replay_and_sketch_validation_proxy_only
claim3_full_support_allowed: false
```

这表示 probe_paper 的 replay/sketch records、table、decision 和 report 已具备工程闭环入口, 但还不能把 Claim-3 写成 full_paper 强 supported claim。后续若要解除 Claim-3 降级, 仍需要 full_paper 级 authenticated replay、held-out negative replay split、wrong sampler / wrong prompt / wrong time-grid FPR 审计和质量约束。

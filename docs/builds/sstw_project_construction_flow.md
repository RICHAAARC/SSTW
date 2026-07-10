# SSTW 项目整体构建流程指引：状态空间同步 Flow Matching 轨迹水印

> 当前实现基线以 `docs/builds/complete_paper_mechanism_implementation.md` 为准。
> 本文档后半部分保留的早期 proxy、preflight 或 claim 收缩讨论仅是历史设计记录,
> 不属于 `probe_paper`、`pilot_paper` 或 `full_paper` 的可执行正式阶段。正式阶段
> 只接受真实 velocity constraint、key 无关固定 inversion、校准概率后验、逐视频
> adaptive attack 和三层完整门禁; 历史替代路径不得进入 PASS。

## 0. 文档定位

## 0.1 文档关系与独立阅读边界

本文档是项目级构建手册, 只规定从机制、数据、baseline、消融、验证门禁到论文结果包的构建顺序, 不记录任何阶段已完成或未完成状态。阶段进度只能记录在 `docs/builds/sstw_phase_completion_status.md` 和 `docs/builds/phases/` 下的分阶段文档中。

`sstw_method_mechanism_design.md` 与 `sstw_algorithm_primitives_design.md` 均可独立阅读。前者定义完整方法机制和论文 claim 边界, 后者定义算法原语、体系创新性和实验映射。`sstw_top_tier_experimental_sufficiency_checklist.md` 独立定义顶刊顶会实验充分性核查清单。本文档只引用这些文档作为规范来源, 不把这些内容压缩成某一次运行结论。


本文档定义 SSTW 项目的整体构建流程。它的职责不是记录项目进度, 也不是列出某一轮实验的执行状态, 而是规定从方法机制到实验产物、论文证据和投稿判断之间的固定构建顺序。

本文档依赖的核心方法机制为:

```text
velocity_field_weak_watermark_constraint
+ endpoint_aware_minimum_energy_flow_control
+ time_reparameterization_invariant_path_observation
+ replay_uncertainty_aware_flow_state_inference
+ flow_state_evidence_admissibility
+ fixed_low_fpr_calibrated_detection
```

因此, SSTW 的项目构建必须围绕以下问题展开:

```text
密钥条件速度场弱约束是否真实参与 Flow Matching 生成轨迹;
路径积分轨迹证据是否在 fixed-FPR 条件下提供独立增益;
状态空间后验推断是否优于普通时序聚合器、显式时间对齐和通用状态模型;
真实 Flow Matching 模型是否能暴露、记录和复现可审计 trajectory proxy;
小规模真实模型 pilot 是否显示 Claim-1、Claim-2 和受限 Claim-3 有成立迹象;
该机制在真实视频生成链路、真实 VAE 链路和攻击协议下是否保持低误报;
所有可支持论文主张的结果是否来自 governed records、tables、figures、reports 和 manifests。
```

本文档属于整体流程指引。具体阶段构建文档、Colab notebook、runner、checker 和 packager 必须服从本文档, 但不得把某次运行结果、临时输出或人工结论写入本文档。

分阶段构建文档统一放置于:

```text
docs/builds/phases/
```

阶段完成情况汇总记录放置于:

```text
docs/builds/sstw_phase_completion_status.md
```

分阶段构建文档可以记录该阶段的当前工程状态; 本整体流程文档只记录流程规范, 不记录执行进度。

---

## 1. 顶会投稿版本的最低定义

### 1.1 方法贡献最低定义

SSTW 的主贡献必须被定义为:

```text
面向 Flow Matching 视频生成轨迹的密钥条件状态空间同步水印。
```

不能将主贡献写成以下内容:

```text
显式时间对齐水印
普通视频后处理水印
endpoint-only latent watermark
通用 temporal aggregator
通用 SSM 检测器
服务端日志审计系统
只依赖最终视频帧的 black-box watermark detector
```

这些内容可以作为 baseline、control、部署组件或降级解释, 但不能成为 SSTW 的方法主体。

### 1.2 必须成立的机制条件

SSTW 作为主论文方法时, 至少需要满足以下机制条件:

1. `velocity_field_weak_watermark_constraint` 必须在采样阶段改变与密钥相关的 flow / velocity trajectory 统计量。
2. `endpoint_aware_minimum_energy_flow_control` 必须说明速度场扰动不是只存在于中间日志, 而是与 endpoint payload 或 endpoint evidence 保持一致。
3. `time_reparameterization_invariant_path_observation` 必须避免把路径证据退化为固定 step index 对齐。
4. `replay_uncertainty_aware_flow_state_inference` 必须显式处理 replay 与原始生成轨迹之间的不确定性。
5. `flow_state_evidence_admissibility` 必须限制状态搜索空间, 避免 negative sample 中的伪轨迹抬高 false positive tail。
6. `fixed_low_fpr_calibrated_detection` 必须使用 calibration negative 固定阈值, 不能在 test split 上更新阈值。

### 1.3 最低论文证据

最低论文证据必须来自以下 governed artifacts:

```text
records/event_scores.jsonl
records/trajectory_traces.jsonl
records/thresholds.jsonl
tables/*.csv
figures/*.json 或 figures/*.png
reports/*.json 或 reports/*.md
manifests/*.json
```

人工整理的表格、手工截图、未绑定 provenance 的结论不能支撑 supported claims。

---

## 2. 方法边界与非重叠规则

### 2.1 与显式同步水印的边界

SSTW 可以比较显式同步水印或时间对齐方法, 但主方法不能依赖显式 temporal matching、DTW、edit-distance matching、segment ordering 或 frame-wise sequence matching。

项目流程中必须保留以下对照:

```text
explicit_temporal_alignment_baseline
frame_matching_temporal_registration_baseline
wrong_sampler_replay_control
trajectory_time_shuffled_control
```

这些对照用于证明 SSTW 的增益来自 Flow Matching 轨迹状态空间机制, 而不是普通同步或普通序列匹配。

### 2.2 与普通视频水印的边界

普通视频水印 baseline 可以用于证明 SSTW 在时间扰动、重采样和生成式攻击下的优势, 但不能使用已经在相关并行论文中作为主 baseline 的方法作为 SSTW 的唯一外部 baseline。

项目流程中必须区分三类 baseline:

```text
mechanism_control_baseline: endpoint_only, trajectory_only, generic_ssm, key_agnostic_state_space
explicit_synchronization_baseline: explicit_dtw_temporal_alignment, frame_matching_temporal_registration
```


### 2.3 与服务端日志审计的边界

`authenticated_trajectory_sketch` 可作为 owner-side trajectory audit 的可信证据来源, 但不能把 SSTW 叙述成单纯的日志审计系统。

认证 sketch 的作用是:

```text
证明轨迹证据来源可信;
限制 replay 轨迹被伪造或替换的风险;
为 model-side replay verification 提供校验锚点。
```

它不是水印本体, 也不能替代 velocity-field watermark constraint。

---

## 3. 全局治理规则

### 3.1 必须固定的 split

```text
calibration
test
stress
ablation
pilot
```

其中:

- `calibration` 只能用于阈值、标准化统计量和 gate 参数冻结;
- `test` 用于主表结果;
- `stress` 用于强攻击、错配 replay 和 adaptive attack;
- `ablation` 用于机制消融;
- `pilot` 用于 small-scale claim pilot, 不得直接进入主论文表格。

### 3.2 必须固定的 sample role

```text
clean_positive
attacked_positive
clean_negative
attacked_negative
replay_negative
sampler_mismatch_negative
wrong_prompt_replay_negative
wrong_time_grid_replay_negative
wrong_key_negative
wrong_key_positive
endpoint_only_control
trajectory_only_control
trajectory_time_shuffled_control
trajectory_key_shuffled_control
wrong_sampler_replay_control
wrong_prompt_replay_control
video_only_proxy_control
```

### 3.3 negative family

`calibration_negative` 不应作为单一 sample role 使用, 而应由多个 negative family 组成。必须新增或映射字段:

```text
negative_family
```

允许取值为:

```text
none
clean_negative
attacked_negative
replay_negative
sampler_mismatch_negative
wrong_prompt_negative
wrong_time_grid_negative
wrong_key_negative
trajectory_shuffle_negative
```

其中, `none` 仅用于 positive 或非负样本 control。

### 3.4 字段闭包原则

所有 event-level records 应逐步闭包以下字段族。若某字段在当前阶段不可用, 必须显式写入 placeholder 状态、unavailable reason 或在阶段文档中说明尚未进入该阶段字段闭包, 不得静默缺失。

```text
sample_id
content_id
prompt_id_placeholder
seed_id_placeholder
generation_model_id_placeholder
sampler_id_placeholder
sampler_signature_placeholder
method_variant
key_id
sample_role
negative_family
split
attack_name
attack_strength
attack_family
trajectory_source_level
trajectory_source_status
trajectory_source_unavailable_reason
threshold_id
threshold_value
threshold_source_split
target_fpr
test_time_threshold_update_blocked
S_endpoint_raw
S_end_state
S_velocity
S_path_raw
S_path_inv
S_path_endpoint_consistency
S_replay
S_state_posterior
S_final_conservative
S_final_weighted_optional
flow_velocity_proxy_available
flow_velocity_alignment_gain
path_marginal_gain_at_fixed_fpr
trajectory_payload_redundancy
replay_uncertainty_mean
replay_uncertainty_curve_id_placeholder
replay_scheduler_id_placeholder
replay_time_grid_id_placeholder
time_grid_reliability
flow_state_admissibility_status
admissibility_failure_reason
negative_tail_status
quality_guard_status
semantic_projection_status
endpoint_controllability_status
authenticated_trajectory_sketch_status
trajectory_sketch_verification_status
decision
decision_reason
claim_support_status
```

### 3.5 旧字段兼容映射

旧字段可在兼容层保留, 但正式表格和新阶段记录应逐步映射为新语义字段:

| 旧字段 | 新字段或新字段族 |
|---|---|
| `key_state_admissibility_status` | `flow_state_admissibility_status` |
| `S_payload_raw` | `S_endpoint_raw` |
| `S_trajectory_observation` | `S_path_inv` 或 `S_velocity` |
| `S_final` | `S_final_conservative` |

字段迁移必须遵守 `docs/field_registry.md` 与 record schema 治理。placeholder 字段必须以 `_placeholder` 结尾, random trace 字段必须以 `_random` 或 `_digest_random` 结尾。

---

## 4. 总体构建里程碑

SSTW 项目采用以下语义阶段。阶段名称表达构建职责, 不表达完成状态。编号只表示推荐阅读与执行顺序, 不作为正式阶段名称的一部分。

```text
protocol_governance_foundation
mechanism_validation
probe_paper
probe_paper
pilot_paper
full_paper
submission_package_freeze
```

`mechanism_validation` 聚合以下机制前置与实现 phase: `synthetic_state_inference_sanity`、`real_video_latent_transfer_check`、`state_space_inference_formalization`、`trajectory_observation_core_probe`、`flow_model_adapter_preflight`、`sampling_time_constraint_probe`、`motion_threshold_calibration`、`small_scale_mechanism_pilot_check` 和真实生成式视频模型实现包。`small_scale_mechanism_pilot_check` 只属于机制层检查, 不能单独放行 `pilot_paper` 或 `full_paper`; `generative_video_model_probe` 只表示真实生成式视频模型实验的实现 package。

`probe_paper` 是 `target_fpr=0.1` 的小样本论文闭合层。它必须使用 `configs/protocol/probe_paper_generative_probe.json` 中 `target_fpr=0.1` 指定的口径跑通与论文协议同构的全部产物链路, 并在 FPR=10% 口径下判断 SSTW 是否成立以及是否相对 5 个现代 external baseline 具备优势证据。`replay_and_authenticated_sketch_gate`、`flow_specific_adaptive_attack_gate`、external baseline、internal ablation、CI reporter、artifact rebuild 和 claim audit 都必须在 `probe_paper` 中形成可落盘、可检查、可失败闭环, 不得推迟到 `pilot_paper` 或 `full_paper` 后再补。`probe_paper` 必须能在小样本规模上产出 paper 相关的全部 governed artifact 类型: generation / detection records、主方法 measured_formal 结果、完整外部 baseline 对比、内部消融、46 个 runtime attack、11 个 non-runtime/adaptive 协议、完整 replay/authenticated sketch gate、fixed-FPR CI、tables、figures、reports、package manifest 和 claim audit。只有 `probe_paper` 通过并生成 `probe_paper_to_pilot_paper_transition_decision` 后, 才允许进入 `pilot_paper`; 不得直接进入 `full_paper`。`pilot_paper` 使用代表性 paper 协议执行 FPR=1% 中等规模结果运行并报告 pilot 级 `TPR@FPR=0.01`。`full_paper` 必须等待 `pilot_paper`、`pilot_paper_to_full_paper_transition_decision`、`full_paper_result_checker`、CI、claim audit、artifact rebuild 和 submission freeze 相关门禁通过。

核心原则是:

```text
先闭合协议, 再闭合状态推断;
先验证路径证据, 再接入真实 Flow sampler;
先验证 Flow 模型接口能记录轨迹, 再验证 velocity constraint 进入采样过程;
先在 mechanism_validation 中完成机制前置检查, 再让 probe_paper 以 FPR=10% 小样本论文闭合全部 paper 产物链路并验证三层主张, 然后通过 probe_paper_to_pilot_paper_transition_decision 进入 pilot_paper; full_paper 仍需 pilot_paper、pilot_paper_to_full_paper_transition_decision 和 full_paper_result_checker 通过;
先控制 negative tail, 再写论文主张;
所有 supported claims 必须由 frozen records 自动重建。
```

每个阶段必须满足三类约束:

1. 方法约束: 与 `sstw_method_mechanism_design.md` 中的算法原语一致。
2. 工程约束: 由 repository modules、runner、checker 和 packager 产生可复现产物。
3. 论文约束: supported claims 必须映射到 governed records、tables、figures、reports 或 manifests。

### 4.1 Colab workflow profile 配置规则

生成式视频主线的 Colab 入口不得在多个 Notebook cell 中硬编码 `probe_paper`、
`pilot_paper` 或未来 `full_paper` 的 run_root、package 目录、样本数量和 gate
阶段。统一配置文件为:

```text
configs/paper_workflow/generative_video_notebook_workflows.json
```

该配置负责定义:

```text
workflow_profile: motion_calibration / probe_paper / pilot_paper / full_paper
result_tier
runtime_profile
protocol_config_path
drive_run_root_relative
drive_package_dir_relative
drive_log_dir_relative
motion_threshold_artifact_run_root_relative
method_sample_count
baseline_sample_count
max_content_records
max_source_records
target_fpr
minimum_clean_negative_count
bootstrap_iteration_count
notebook_role
workflow_stage_plan
```

Notebook 只能通过 `workflows/generative_video_paper.py`
读取上述配置, 再调用 `experiments/`、`scripts/` 或 `main/` 中的正式模块。禁止
在 Notebook 中为不同结果层级单独维护一套路径和样本上限。

当前推荐 Notebook 编排为:

```text
motion_threshold_calibration_colab.ipynb
-> generative_video_generation_colab.ipynb
-> generative_video_quality_scoring_colab.ipynb
-> runtime_attack_colab.ipynb
-> runtime_detection_colab.ipynb
-> 5 个主实验 modern external baseline formal reference Notebook
-> formal_comparison_scoring_colab.ipynb
-> paper_evidence_postprocess_colab.ipynb
-> paper_gate_and_package_colab.ipynb
```

当前不保留 `probe_paper` 单 Notebook 全流程入口。严格门禁必须通过拆分 Notebook
逐段执行, 并通过本项目完成 5 个主实验现代 baseline 的 clone / build / run / adapt / record。
若某个 baseline 只能在高显存或特殊依赖环境中运行, 也必须通过本项目记录 source intake、
构建命令、运行命令、adapter 输出和 governed non-run reason, 不接受外部补交结果。

其中:

1. `motion_threshold_calibration_colab.ipynb` 只负责独立 calibration split 和阈值冻结。
2. 4 个 SSTW runtime 拆分 Notebook 负责 Wan2.1 生成、formal metrics、阈值复用、
   runtime attack 和 detection, 不执行现代 baseline command 预检或 baseline scoring。
3. 5 个主实验 modern external baseline formal reference Notebook 分别负责对应 baseline 的 source intake、clone / build / run / adapt 和 official bundle 生成, 不默认调用全量 runner 转写 `measured_formal` records。
4. `formal_comparison_scoring_colab.ipynb` 负责恢复 5 个主实验 baseline official reference 阶段包, 重新执行全量统一转写、self-containment 判定、公平校准和差值区间统计。
5. `paper_evidence_postprocess_colab.ipynb` 负责恢复 runtime、motion threshold 和 formal comparison scoring 阶段包, 再执行正式内部消融、formal adaptive attack 证据整理、完整 replay/authenticated sketch gate、CI、low-FPR formal statistics 和数据切分泄漏检查。
6. `paper_gate_and_package_colab.ipynb` 负责恢复 runtime、motion threshold、formal comparison scoring 和 paper evidence postprocess 阶段包, 只执行最终 fixed-FPR gate、transition decision、artifact rebuild dry run、figure/package manifest builder 和 Drive package。

旧的通用 external baseline scoring Notebook 已删除。正式主流程应以 5 个主实验
baseline 专用 Notebook 产生的 official bundle 为输入, 由
`formal_comparison_scoring_colab.ipynb` 统一重建最终 `measured_formal` records,
避免单 baseline 临时 records 互相覆盖。

旧综合 Notebook 已移除。正式推进只使用拆分 Notebook, 因为拆分后可以在同一 `workflow_profile` 下分阶段复跑、检查和打包,
避免 runtime、baseline 与 gate 的失败原因混在一个长 Notebook 中。

`probe_paper`、`pilot_paper` 和 `full_paper` 的协议差异必须由 protocol config 显式记录。当前三个 paper profile
共享同一套 46 个 runtime attack 与 11 个 non-runtime / adaptive 协议; 差异只允许是样本规模、统计功效和 target_fpr。当前 `full_paper`
已通过 `configs/protocol/full_paper_generative_probe.json` 登记正式协议要求, 但 workflow profile
在前置门禁闭合前只能作为协议配置和 checker 目标, 不允许作为 claim profile 或 submission source 使用。

必须区分以下 3 个状态, 避免形成“full_paper 必须先运行, 但又要等 full_paper_result_checker 先通过”的循环表述:

```text
full_paper_run_allowed: 只表示允许启动 full_paper 规模结果生产, 需要 paper_profile_gate、probe_paper_gate、probe_paper_to_pilot_paper_transition_decision、pilot_paper_gate、pilot_paper_to_full_paper_transition_decision 和 full_paper 运行前置门禁通过。
full_paper_claim_allowed: 只表示允许把 full_paper 结果写成正式论文 claim, 需要 full_paper records 生成后由 full_paper_result_checker 判定 PASS。
submission_freeze_allowed: 只表示允许冻结投稿包, 需要 full_paper_result_checker、reviewer_evidence_index_builder、full_paper_to_submission_freeze_transition_decision、claim audit 和 artifact rebuild 全部通过。
```

`configs/protocol/full_paper_generative_probe.json` 是正式规模论文协议配置的唯一入口。该配置至少必须固定:

```text
target_fpr == 0.001
threshold_protocol == calibration_split_to_frozen_threshold_to_heldout_test_split
minimum_prompt_count == 125
minimum_seed_per_prompt == 8
minimum_unique_video_count == 1000
minimum_calibration_seed_per_prompt == 4
minimum_test_seed_per_prompt == 4
minimum_calibration_unique_video_count == 500
minimum_test_unique_video_count == 500
minimum_calibration_negative_event_count >= 50000
minimum_heldout_test_negative_event_count >= 50000
minimum_heldout_attacked_positive_event_count >= 46000
minimum_negative_event_count_per_family >= 12500
minimum_calibration_negative_event_count_per_family >= 12500
minimum_heldout_negative_event_count_per_family >= 12500
minimum_attack_event_count_per_attack >= 1000
minimum_modern_external_baseline_formal_adapter_count >= 5
minimum_external_baseline_measured_adapter_count >= 5
minimum_full_paper_external_baseline_trace_count >= 500
minimum_full_paper_internal_ablation_trace_count >= 500
minimum_internal_ablation_variant_count >= 8
require_paper_profile_gate_passed == false
require_probe_paper_gate_passed == true
require_probe_paper_to_pilot_paper_transition_decision == true
require_pilot_paper_gate_passed == true
require_pilot_paper_to_full_paper_transition_decision == true
require_external_baseline_self_containment_decision == true
require_data_split_and_leakage_guard == true
require_external_baseline_self_contained_outputs == true
require_confidence_interval_report == true
require_statistical_confidence_interval_decision == true
require_artifact_rebuild_dry_run == true
require_claim_audit_report == true
require_artifact_rebuild_report == true
```

若 full_paper profile 与该配置不一致, 以协议配置为阻断依据, 不能用 Notebook 参数临时放行。

注意: profile-specific run_root 用于防止不同结果层级混写。motion threshold calibration
artifact 通过 `motion_threshold_artifact_run_root_relative` 显式共享给 `probe_paper` 与
`pilot_paper`, 而不是把 calibration 输出复制进 evaluation run_root。

---

## 5. protocol_governance_foundation

### 5.1 目标

建立全项目共享的协议、字段、命名、输出、测试和 artifact rebuild 规则。

### 5.2 必须实现

```text
configs/protocol/sstw_protocol.json
configs/protocol/fixed_low_fpr.json
configs/records/event_record_schema.json
configs/records/state_trace_schema.json
configs/records/threshold_schema.json
docs/field_registry.md
tools/harness/run_all_audits.py
```

### 5.3 必须冻结的协议

```text
split 定义
sample role 定义
negative family 定义
target_fpr
threshold_source_split
test_time_threshold_update_blocked
baseline 与 control 命名
placeholder 与 random 字段命名
```

### 5.4 通过标准

1. 默认 `pytest -q` 不运行重型 GPU 测试。
2. `python tools/harness/run_all_audits.py` 不报告命名、字段、依赖边界或 artifact 治理违规。
3. 正式 claims 均能映射到 governed artifacts。

---

## 6. synthetic_state_inference_sanity

### 6.1 目标

在可控 synthetic latent 中验证密钥条件状态空间推断是否优于普通时序聚合器与显式时间对齐 baseline。

### 6.2 必须实现

```text
key_conditioned_state_inference
state_transition_model
state_observation_model
fixed_low_fpr_calibration
wrong_key_control
state_shuffle_control
negative_family_calibration
```

### 6.3 必须比较

```text
mean_temporal_aggregator
max_temporal_aggregator
conv1d_temporal_aggregator
explicit_temporal_alignment_baseline
generic_ssm_baseline
key_agnostic_ssm_baseline
```

### 6.4 必须证明

1. key-conditioned posterior 在 `TPR@FPR=0.01` 下优于普通 aggregator。
2. wrong-key score 分布不能接近 correct-key positive 分布。
3. 状态模型不能通过扩大搜索空间显著抬高 negative score tail。

### 6.5 论文使用边界

该阶段只能支撑方法合理性和机制 sanity, 不能单独支撑真实视频生成模型主 claim。

---

## 7. real_video_latent_transfer_check

### 7.1 目标

验证 synthetic 阶段成立的密钥条件状态空间推断在真实视频 VAE encode-decode-reencode 链路中是否仍然有效并保持低误报。

### 7.2 实验流程

```text
video frames
-> VAE encode
-> latent watermark or latent trace construction
-> VAE decode
-> video attack
-> VAE re-encode
-> state-space detection
```

### 7.3 必须实现

```text
vae_backend_adapter
encode_decode_reencode_runner
reconstruction_quality_audit
fixed_low_fpr_threshold_reuse
endpoint_consistency_score
```

### 7.4 必须比较

```text
endpoint_only_latent_score
frame_prc_baseline
explicit_temporal_alignment_baseline
generic_ssm_baseline
key_conditioned_state_space_method
```

### 7.5 必须证明

1. VAE 重建误差不会破坏低 FPR 校准。
2. state-space evidence 在 attacked positive 中仍有有效增益。
3. endpoint-only 方法不足以解释 SSTW 的整体分数提升。

---

## 8. state_space_inference_formalization

### 8.1 目标

将密钥条件状态空间推断整理为可审计、可消融、可复现实验模块。

### 8.2 必须实现

```text
formal_state_variable_definition
key_conditioned_transition
key_conditioned_observation_likelihood
filtering_and_smoothing
state_posterior_score
flow_state_evidence_admissibility
negative_tail_audit
```

### 8.3 必须比较

```text
without_key_condition
without_state_transition
without_observation_likelihood
without_admissibility
generic_ssm
mamba_style_temporal_fusion_control
```

### 8.4 必须证明

1. 状态变量不是普通后处理特征堆叠。
2. key conditioning 对检测有独立贡献。
3. admissibility 降低 false positive tail。

---

## 9. trajectory_observation_core_probe

### 9.1 目标

验证轨迹观测是否提供与 endpoint evidence 不同源的路径证据。

### 9.2 必须实现

```text
trajectory_trace_capture
velocity_projection_operator
path_integral_statistic
time_reparameterization_invariant_path_observation
trajectory_state_adapter
endpoint_consistency_score
path_marginal_gain_at_fixed_fpr
trajectory_payload_redundancy
```

### 9.3 必须比较

```text
endpoint_only_control
trajectory_only_score
trajectory_time_shuffled_control
wrong_key_trajectory_control
wrong_sampler_replay_control
explicit_temporal_alignment_baseline
```

### 9.4 必须证明

1. path evidence 与 endpoint evidence 不能高度冗余。
2. time-reparameterization-invariant path score 在不同 scheduler / time grid 下保持可比。
3. wrong sampler replay 不能获得与 correct replay 等价的路径证据。
4. trajectory-only score 不能绕过 fixed-FPR 和 admissibility 直接支撑最终判定。

### 9.5 顶会 gate

若路径证据无法在 fixed-FPR 条件下提供独立边际增益, 项目应降级为状态空间同步视频水印或 endpoint-aware Flow latent watermark, 不应宣称完整的 Flow Matching 轨迹水印。

---

## 10. flow_model_adapter_preflight

### 10.1 目标

在进入昂贵的真实生成模型实验前, 确认目标 Flow Matching / Rectified Flow 视频生成模型是否能够暴露、记录和复现 SSTW 所需的 trajectory proxy。

### 10.2 主线模型

主线模型应优先使用 velocity-field sampler 或 Flow Matching 机制清晰的 DiT 视频生成模型。项目主线模型配置为:

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

轻量模型可以作为接口预验证或 fallback probe, 但不能单独支撑完整 Flow Matching trajectory watermark 主 claim。

### 10.3 必须实现

```text
flow_model_backend_adapter
sampler_callback_adapter
latent_state_capture
velocity_or_displacement_proxy_capture
sampler_signature_recorder
time_grid_recorder
prompt_seed_manifest_reader
generation_reproducibility_check
trajectory_trace_schema_writer
```

### 10.4 必须检查

```text
model_load_success
single_prompt_generation_success
callback_latent_capture_success
velocity_proxy_available
latent_displacement_available
sampler_signature_available
time_grid_available
replay_adapter_feasibility
gpu_memory_budget_status
colab_runtime_budget_status
```

### 10.5 通过标准

1. `trajectory_trace_capture_success_rate >= 0.95`。
2. 每个样本均能记录 sampler signature 与 time grid。
3. 至少一种 velocity proxy 或 latent displacement proxy 可用。
4. 生成结果可由 prompt、seed、model id、sampler id 与 config 复现。
5. 若 velocity field 原始值不可访问, 必须记录 `flow_velocity_proxy_available=false` 与 proxy 类型, 不得伪称拥有真实 velocity field。

### 10.6 失败处理

若该阶段失败, 不得进入 sampling-time constraint 或 `generative_video_model_probe` 后续实现 package 运行。应优先修复 backend adapter、callback、scheduler hook 或选择替代 Flow Matching / velocity-field 模型。

---

## 11. sampling_time_constraint_probe

### 11.1 目标

验证 sampling-time weak constraint 是否真正进入 Flow Matching 采样动力学, 而不是只在后处理阶段改变最终视频。

### 11.2 必须实现

```text
velocity_field_weak_watermark_constraint
endpoint_aware_minimum_energy_flow_control
lambda_schedule
quality_guard
semantic_projection
flow_velocity_proxy_record
callback_latent_displacement_record
```

### 11.3 必须比较

```text
no_constraint_control
endpoint_only_constraint
constant_lambda_constraint
no_endpoint_aware_control
no_semantic_projection
wrong_key_constraint
```

### 11.4 必须证明

1. keyed flow velocity alignment gain 大于 baseline。
2. endpoint payload 或 endpoint evidence 与路径证据一致。
3. 质量保护约束没有被关闭或绕过。
4. semantic projection 不只是注释, 必须有可审计字段或可替代的 placeholder 字段边界。

### 11.5 失败处理

若 velocity constraint 不能提供有效增益, 不得宣称 SSTW 是 Flow Matching trajectory watermark。可降级为 endpoint-aware latent watermark 或 state-space synchronized video watermark。

---

## 12. small_scale_mechanism_pilot_check

`small_scale_mechanism_pilot_check` 属于 `mechanism_validation` 的子检查, 不能单独放行 `pilot_paper` 或 `full_paper`。

### 12.1 目标

在进入 `probe_paper` 前, 以较小成本验证 Claim-1、Claim-2 和部分 Claim-3 是否有成立迹象。该阶段不产生主论文表格, 也不直接决定是否进入 paper 级结果运行; `small_scale_mechanism_pilot_check` 通过后只允许进入 `probe_paper`。只有 `probe_paper` 通过并生成 `probe_paper_to_pilot_paper_transition_decision` 后, 才允许进入 `pilot_paper`。

### 12.2 推荐规模

```text
N_prompt >= 8
N_seed_per_prompt >= 2
N_attack >= 3
N_calibration_negative_family >= 4
N_method_variant >= 6
```

### 12.3 必须覆盖的 method variant

```text
sstw_full_method
endpoint_only_control
trajectory_only_score
without_velocity_constraint
without_endpoint_aware_control
without_replay_uncertainty_weighting
generic_ssm_baseline
explicit_dtw_temporal_alignment
```

### 12.4 必须覆盖的攻击和错配

```text
probe_paper_required_runtime_attack_names:
  video_compression_runtime
  temporal_crop_runtime
  frame_rate_resampling_runtime
pilot_paper_additional_runtime_attack_names:
  frame_drop_uniform_runtime
  spatial_resize_runtime
  spatial_crop_resize_runtime
  gaussian_blur_runtime
  gaussian_noise_runtime
full_paper_required_runtime_attack_families:
  multi_strength_codec_and_platform_transcode_compression
  complete_temporal_disturbance_including_irregular_drop_insert_speed
  spatial_geometry
  visual_degradation_including_denoise_gamma_sharpen
  combined_transformations_including_compression_color_and_crop_rotation
full_paper_required_non_runtime_or_adaptive_attack_protocols:
  generative_recompression_or_regeneration_attack
  endpoint_preserving_path_perturbation_attack
  flow_time_grid_mismatch_attack
  wrong_sampler_replay_attack
  wrong_prompt_replay_attack
  wrong_key_attack
  detector_probing_with_public_negatives
  watermark_removal_optimization_attack
  watermark_spoofing_or_copy_attack
  collusion_multi_sample_attack
  adversarial_detector_evasion_attack
```

`probe_paper` 必须使用与 `full_paper` 一致的 46 个 runtime attack 和 11 个
non-runtime/adaptive 协议, 但样本量和目标 FPR 保持 probe_paper 配置口径。
`pilot_paper` 与 `full_paper` 必须通过 protocol config 的 `required_runtime_attack_names`
显式切换样本规模和 FPR 等级, 不允许在 Notebook 中手写或临时删减 attack。

### 12.5 通过标准

```text
trajectory_trace_capture_success_rate >= 0.95
flow_velocity_alignment_gain > 0
path_marginal_gain_at_fixed_fpr > 0
trajectory_payload_redundancy <= preset_limit
quality_guard_violation_rate <= preset_limit
negative_tail_inflation_not_detected = true
wrong_key_score_separation_passed = true
wrong_sampler_replay_control_not_equivalent = true
```

### 12.6 失败处理

若 small-scale 机制 pilot 检查失败, 不应进入 `probe_paper`, 更不能进入 paper 级结果运行。应根据失败原因回退:

```text
flow_velocity_alignment_gain <= 0 -> 回退 sampling_time_constraint_probe;
path_marginal_gain_at_fixed_fpr <= 0 -> 回退 trajectory_observation_core_probe;
negative tail inflation -> 回退 protocol_governance_foundation / state_space_inference_formalization / replay_and_authenticated_sketch_gate;
quality_guard_violation -> 回退 sampling_time_constraint_probe;
replay failure -> 回退 replay_and_authenticated_sketch_gate; 当前 paper profile 必须阻断。
```

---

## 13. generative_video_model_probe implementation package

本节描述真实生成式视频模型实验的实现 package, 不再定义独立主门禁。该 package 的输出由 `mechanism_validation`、`probe_paper`、`pilot_paper` 和 `full_paper_result_checker` 消费。

### 13.1 目标

在真实生成式视频模型上验证 SSTW 的轨迹观测、状态空间推断、检测协议、外部 baseline、内部消融、adaptive attack、replay/sketch、CI 和 artifact rebuild 是否形成完整 paper 机制闭环。该实现 package 应在 flow model adapter preflight、sampling-time constraint 和 small-scale 机制 pilot 检查之后使用, 但阶段放行以 `probe_paper`、`pilot_paper` 和 `full_paper` 主干门禁为准。

### 13.2 数据集构造

真实数据集构造应与测试运行流程分离。数据集准备步骤应产生:

```text
prompt_suite.json
seed_plan.json
content_manifest.json
generation_manifest.json
```

测试运行流程只能读取这些 manifest, 不应在正式实验中临时改变 prompt、seed 或 split。

### 13.3 必须比较

```text
sstw_full_method
endpoint_only_control
trajectory_only_score
without_velocity_constraint
without_endpoint_aware_control
without_replay_uncertainty_weighting
without_admissibility_or_flow_state_gate
explicit_dtw_temporal_alignment
frame_matching_temporal_registration
generic_ssm_baseline
key_agnostic_state_space_baseline
videoshield
vidsig
videoseal
```

### 13.4 外部 baseline

顶刊顶会版本不得只使用显式时间对齐作为外部 baseline。外部 baseline 必须覆盖至少三类方法:

```text
posthoc_neural_video_watermark: VideoSeal
explicit_synchronization_control: explicit_dtw_temporal_alignment, frame_matching_temporal_registration
```

推荐优先级如下:

| baseline | 推荐角色 | 纳入原因 | governed record 要求 |
|---|---|---|---|
| VideoShield | primary modern video diffusion watermark baseline | 训练-free in-generation 视频扩散水印, 与生成时水印最接近 | 记录模型、视频长度、攻击、检测阈值、失败原因 |
| VideoSeal | post-hoc robust video watermark baseline | 开源且工程成熟, 用于证明 SSTW 不是普通后处理水印 | 记录后处理开销、质量指标与攻击鲁棒性 |
| explicit_dtw_temporal_alignment | synchronization control | 显式恢复时间路径的反事实 baseline | 只能支撑 control 结论 |
| frame_matching_temporal_registration | synchronization control | 显式帧匹配配准 baseline | 只能支撑 control 结论 |

外部 baseline 选择原则:

1. 优先选择 2025-2026 年顶会、顶刊、OpenReview、CVF OpenAccess、arXiv 或官方代码中与视频生成水印直接相关的方法。
2. 至少一个 baseline 必须是 in-generation video watermark, 至少一个 baseline 必须是 post-hoc video watermark, 至少一个 baseline 必须是显式时间同步 control。
3. 不能把无法运行的 baseline 直接写成“弱于 SSTW”。无法运行只能形成 `external_baseline_not_run_reason` 与 protocol gap。
4. 所有正向比较必须来自 records 和 tables, 不得由人工阅读论文后手写。

### 13.5 必须证明

1. Wan2.1 主线记录必须标记为 Flow Matching / velocity-field sampler 相关模型。
2. velocity / flow trajectory proxy 必须参与水印同步证据。
3. 在 `probe_paper` 层, SSTW full method 必须在 `target_fpr=0.1` 下完成公平比较产物闭合和小样本论文闭合判定; 该层支持 FPR=10% 条件下的完整三层论文结论, 不支持外推为 `pilot_paper` 或 `full_paper` 结论。在 `pilot_paper` 层才可以报告 `TPR@FPR=0.01` 级别的 pilot 对比。
4. `probe_paper` gate 前必须已经生成同批小样本 test trace 的 external_baseline comparison records、内部消融 records、adaptive attack records、完整 replay/authenticated sketch gate records、CI report 和 artifact rebuild report。
5. 质量、运动和语义指标不能显示不可接受退化。
6. `probe_paper` 通过并生成 `probe_paper_to_pilot_paper_transition_decision` 后只能进入 `pilot_paper`, 不得直接进入 `full_paper`。full_paper claim 仍需 `pilot_paper_gate`、`pilot_paper_to_full_paper_transition_decision`、`full_paper_result_checker` 和轻量判定通过。若完整现代 baseline、内部消融或 replay/sketch 机制仍缺失, 只能报告阻断原因, 不能进入 paper 级结果运行。

### 13.6 probe_paper 作为 FPR=10% 小样本论文闭合层

`probe_paper` 的职责是证明 paper 级运行所需的全部机制和产物链路已经在小样本规模上闭合, 并在 probe_paper protocol config 中的 `target_fpr=0.1` 口径下判断 SSTW 是否具备完整三层论文结论。它既用于以最小成本提前发现 baseline、消融、attack、CI、artifact rebuild、claim audit 和 package 阻断, 也用于检查 SSTW 相对 5 个现代 external baseline 的优势证据是否在 FPR=10% 设定下成立。该结论不得外推为 `pilot_paper` 的 FPR=1% 结果或 `full_paper` 的 FPR=0.1% 主结果。

`probe_paper` 至少必须满足:

```text
validation_generation_records_ready
validation_detection_records_ready
validation_external_baseline_comparison_records_ready
external_baseline_measured_adapter_count >= 5
modern_external_baseline_formal_measured_adapter_count >= 5
validation_internal_ablation_records_ready
required_internal_ablation_variants covered
validation_adaptive_attack_records_ready
validation_replay_and_authenticated_sketch_gate_passed
validation_confidence_interval_report_ready
probe_paper_sstw_advantage_claim_ready
validation_tables_figures_reports_ready
validation_artifact_rebuild_dry_run_ready
validation_claim_audit_ready
```

该门禁失败时, 下一步是补齐缺失机制或修正 adapter, 不是进入 `pilot_paper`。显式 DTW 与 frame matching 只能作为同步 control, 不能替代现代视频水印 baseline 的 `measured_formal` records。

---

## 14. replay_and_authenticated_sketch_gate

### 14.1 目标

规范 owner-side trajectory audit、model-side replay verification 和 video-only proxy observation 的证据等级。

### 14.2 证据等级

```text
level_1_authenticated_cached_trajectory
level_2_model_side_replay_with_uncertainty
level_3_video_only_proxy_observation
```

### 14.3 必须实现

```text
authenticated_trajectory_sketch_status
trajectory_sketch_digest_random
trajectory_sketch_verification_status
replay_uncertainty_weight
replay_scheduler_id_placeholder
replay_time_grid_id_placeholder
wrong_sampler_replay_control
wrong_prompt_replay_control
```

### 14.4 判定规则

1. level 1 可以支撑高置信 owner-side trajectory audit。
2. level 2 可以支撑受限 model-side replay verification, 但必须记录 replay uncertainty。
3. level 3 只能作为补充证据, 除非在 fixed-FPR 下经过独立验证。
4. 未认证 trajectory logging 只能作为 control, 不能支撑主 claim。

### 14.5 replay/sketch gate 的正式阻断边界

`replay/sketch gate` 是 Claim-3 的正式支撑门禁，不能被未认证 trajectory logging、简单 replay score 或 owner-side audit 替代。三个 paper profile 都必须完整通过该门禁；失败时当前 profile 直接阻断，不存在 Claim-3 降级流程。

该 gate 必须至少检查：

```text
authenticated_trajectory_sketch_status == ready
trajectory_sketch_verification_status == pass
calibrated_probability_posterior_ready == true
wrong_key_replay_records_ready == true
wrong_sampler_replay_records_ready == true
wrong_prompt_replay_records_ready == true
wrong_time_grid_replay_records_ready == true
replay_negative_fpr_controlled == true
replay_and_sketch_gate_decision == PASS
```

允许的阶段策略为：

```text
probe_paper: replay/sketch gate_required
pilot_paper: replay/sketch gate_required
full_paper: replay/sketch gate_required
```

---

## 15. flow_specific_adaptive_attack_gate

### 15.1 目标

验证方法不是依赖容易被破坏的固定采样步或固定 endpoint pattern。

### 15.2 常规 Flow mismatch 攻击

```text
scheduler_change
step_count_change
time_grid_jitter
wrong_sampler_replay
latent_noise_perturbation
vae_reencode_attack
video_compression
frame_rate_resampling
temporal_crop
local_clip
```

### 15.3 Flow-specific 自适应攻击

```text
velocity_projection_suppression
path_response_cancellation
endpoint_path_decoupling
replay_signature_mismatch
trajectory_sketch_replacement_attempt
```

### 15.4 必须证明

1. 时间重参数化不变路径证据优于固定 step-index 路径证据。
2. replay uncertainty weighting 能降低错误 replay 带来的误报风险。
3. admissibility gate 能限制 adaptive attack 下的 negative score tail。


### 15.5 pilot_paper 前置子门禁

`pilot_paper` 不是只跑主方法的缩小版, 而是在 `probe_paper` 已经闭合全部 paper 机制之后执行的小规模 paper 级结果运行。因此在 `pilot_paper_gate` 允许报告 pilot 级 `TPR@FPR=0.01` 前, 必须复核以下条件没有在 paper 级运行中退化:

```text
paper_profile_gate_decision == PASS
external_baseline_comparison_decision == PASS
external_baseline_measured_adapter_count >= 5
modern_external_baseline_formal_measured_adapter_count >= 5
pilot_paper_external_baseline_trace_count_min >= 50
validation_internal_ablation_decision == PASS
pilot_paper_internal_ablation_trace_count_min >= 50
required_internal_ablation_variants covered
```

显式 DTW 与 frame matching 只能作为同步 control proxy。现代视频水印 baseline 如果尚未 runnable, 必须写出 governed non-run reason, 不能被人工省略或替换成 control proxy。

---

## 16. submission_package_freeze

### 16.1 目标

将 governed records 转换为论文可使用的 tables、figures、reports 和 manifests。

### 16.2 必须产物

```text
records/event_scores.jsonl
records/thresholds.jsonl
records/trajectory_traces.jsonl
tables/main_detection_table.csv
tables/baseline_comparison_table.csv
tables/ablation_table.csv
figures/roc_or_tpr_at_fpr_figure.json
figures/trajectory_evidence_figure.json
reports/claim_audit_report.json
reports/readiness_summary.json
manifests/submission_package_manifest.json
```

### 16.3 禁止事项

1. 不得手工改写正式表格数值。
2. 不得从 test split 反向更新 calibration threshold。
3. 不得用 placeholder 字段支撑 supported claims。
4. 不得把临时 Colab 输出直接当成论文 artifact。

---

## 17. 三层论文主张策略

### 17.1 Claim-1: Flow velocity watermarking

主张内容:

```text
密钥条件弱约束进入 Flow Matching velocity / flow trajectory, 并形成可检测的轨迹响应。
```

必须证据:

```text
flow_velocity_alignment_gain
velocity_projection_operator_id
sampling_constraint_variant
quality_guard_status
```

### 17.2 Claim-2: Path-posterior trajectory inference

主张内容:

```text
时间重参数化不变路径证据与密钥条件状态空间后验推断在 fixed-FPR 下提供独立增益。
```

必须证据:

```text
S_path_inv
S_state_posterior
S_final_conservative
threshold_value
target_fpr
path_marginal_gain_at_fixed_fpr
flow_state_admissibility_status
```

### 17.3 Claim-3: Robust replay verification

主张内容:

```text
在 owner-side trajectory audit 和受限 model-side replay verification 中, SSTW 能保持低误报并优于 baseline。
```

必须证据:

```text
authenticated_trajectory_sketch_status
replay_uncertainty_weight
wrong_sampler_replay_control
baseline_comparison_table
claim_audit_report
```

### 17.4 Claim 降级规则

如果 Claim-1 或 Claim-2 不成立, 不应宣称 Flow Matching trajectory watermark。

如果 Claim-3 只在部分攻击下成立, 论文应把 owner-side audit 作为高置信设定, 把 replay verification 作为受限但有效的半白盒验证, 不应宣称完全 video-only black-box detection。

---

## 18. 投稿 gate

### 18.1 Gate A: 机制成立

```text
flow_velocity_alignment_gain > 0
endpoint evidence 可检测
path evidence 与 endpoint evidence 不高度冗余
path_marginal_gain_at_fixed_fpr > 0
wrong_key_score_separation_passed = true
wrong_sampler_replay_control_not_equivalent = true
```

### 18.2 Gate B: 检测成立

```text
TPR@FPR=0.01 达到预设目标
TPR@FPR=0.001 达到 full_paper 预设目标
clean_negative_fpr_controlled = true
attacked_negative_fpr_controlled = true
replay_negative_fpr_controlled = true
sampler_mismatch_negative_fpr_controlled = true
threshold_freeze_passed = true
```

### 18.3 Gate C: 论文成立

```text
sstw_full_method_beats_internal_baselines = true
sstw_full_method_beats_temporal_alignment_baselines = true
quality_degradation_within_limit = true
motion_degradation_within_limit = true
semantic_degradation_within_limit = true
claim_audit_passed = true
artifact_rebuild_passed = true
submission_manifest_complete = true
```

### 18.4 投稿判断

若 Gate A 或 Gate B 失败, 不建议作为完整 SSTW 投稿。

若 Gate A 与 Gate B 成立, 但 Gate C 不完整, 可作为技术报告、workshop 或降级论文继续完善。

若 Gate A、Gate B 与 Gate C 全部通过, 可以按完整 SSTW 主论文准备投稿。

若 Claim-3 也在中等强度攻击下稳定成立, 则该版本具备更高水平会议投稿潜力。

---

## 19. Notebook 与 Colab 构建规则

### 19.1 Notebook 职责

Notebook 是远程 GPU 实验入口, 不是正式协议逻辑的唯一实现。

Notebook 必须调用:

```text
main/
experiments/
scripts/check_results/
scripts/package_results/
paper_workflow/notebook_utils/
```

Notebook 不得直接手写正式 records、thresholds、tables、figures 或 reports。

### 19.2 Google Drive 输出规则

Colab 输出应落盘到:

```text
/content/drive/MyDrive/SSTW
```

建议目录结构为:

```text
/content/drive/MyDrive/SSTW/datasets
/content/drive/MyDrive/SSTW/resources
/content/drive/MyDrive/SSTW/motion_threshold
/content/drive/MyDrive/SSTW/<workflow_profile>/<stage_package_id>
/content/drive/MyDrive/SSTW/<workflow_profile>/external_baseline_official_reference
/content/drive/MyDrive/SSTW/helper
```

当前 Colab 正式流程采用“本地热路径 + Drive 冷归档”的阶段 zip 交接规则:

1. Notebook 启动时只从新目录结构中选择最新的
   `<workflow_profile>_<stage_package_id>_<YYYYMMDD_HHMMSS>_<git_short_commit>.zip`
   复制少量上游阶段包到 Colab 本地 workspace。
2. 模型运行、attack、scoring、adapter 和 result checker 的小文件读写只能发生在
   Colab 本地 workspace, 不应循环读写 Google Drive 小文件。
3. Notebook 初始化阶段不应在 Drive 上预创建 `runs/`、`logs/` 或 `datasets/`
   热路径空目录; 这些路径在 `local_zip` 模式下会被映射到 `/content` 本地 workspace。
4. 阶段完成后由 `publish_colab_stage_package` 统一生成带时间戳的 zip 和 manifest
   并写回 Drive。默认保留时间戳包, 不再写固定 latest 小入口。
5. 旧版 `packages/` 目录不再作为 Notebook 间自动交接入口。
6. 独立运行 `scripts/package_results/*_drive_packager.py` 时, 默认输出目录也必须
   解析到当前阶段归档结构, 不得默认回退到旧版 `SSTW/packages/`。
7. failed 或未闭合的 `external_baseline_formal_reference_*` 阶段只能写阻断 manifest,
   不得保存可被后续门禁恢复的 zip。这样可以防止失败运行占用大量 Drive 空间, 也防止
   后续 Notebook 误用旧 external baseline 输出。

`resources/` 只保存可复用的大模型、checkpoint、官方资源包等冷资源。若资源以 zip 包保存,
或等价资源包保存, Notebook 解包后再由官方运行器使用; 它不应随每次实验阶段重复打包。

数据集构造、模型运行、结果检查和阶段打包下载应分离为不同步骤。

### 19.3 Notebook 必须包含的步骤

```text
mount_google_drive
install_repository_dependencies
prepare_or_read_dataset_manifests
run_preflight_or_generation
run_detection_and_baselines
run_result_checker
package_to_google_drive
print_next_step_and_evidence_gap
```

### 19.4 Notebook 禁止事项

1. 不得在 Notebook 中手工生成正式论文表格。
2. 不得在 Notebook 中重写 threshold。
3. 不得将 pilot split 结果直接写入主表。
4. 不得将未认证 trajectory logging 当作高置信证据。

---

## 20. 最终投稿判断

### 20.1 不建议作为完整 SSTW 投稿的情况

出现以下任一情况时, 不应将论文主张写成完整 Flow Matching 轨迹水印:

1. velocity / flow trajectory 没有参与水印同步证据。
2. path evidence 与 endpoint evidence 高度冗余且无独立边际增益。
3. fixed-FPR 条件下无法达到目标 TPR。
4. wrong key、wrong sampler 或 negative replay 产生高误报。
5. 质量、运动或语义退化超过可接受边界。
6. supported claims 无法映射到 governed artifacts。

### 20.2 可以作为主论文的最低情况

最低情况需要同时满足:

1. Wan2.1 主线模型运行链路可复现。
2. `TPR@FPR=0.01` 达到论文预设目标, 且 full_paper 结果包必须额外达到 `TPR@FPR=0.001` 的冻结阈值评估要求。
3. SSTW full method 优于内部机制 baseline 和外部 baseline。
4. trajectory evidence、state posterior 和 endpoint evidence 形成互补。
5. claim audit、artifact rebuild 和 harness 全部通过。

### 20.3 强接收版本建议

强接收版本应进一步满足:

1. 跨 prompt、seed、attack 和 negative family 的泛化结果稳定。
2. 至少一个额外 Flow Matching / velocity-field 视频模型作为补充验证。
3. Flow-specific adaptive attacks 下仍保持低 FPR。
4. authenticated trajectory sketch 与 replay uncertainty 形成完整证据链。
5. 所有主表、主图和 claim audit 可由 records 与 manifests 自动重建。

## 21. 数据集与 prompt suite 构建方法

本节规定数据集与 prompt suite 的构建方法。该规则用于降低弱 prompt、低运动样本、prompt 泄漏、test-time 阈值更新和 cherry-picking 导致的拒稿风险。

### 21.1 数据集分层

SSTW 的数据集必须按“样本事实来源”和“实验用途”同时分层。推荐数据层如下:

```text
engineering_calibration_dataset: 用于工程阈值和可观测性门控, 不进入论文主表
pilot_dataset: 用于 small-scale claim pilot, 只决定机制是否值得继续
probe_paper_dataset: 用于在小样本规模验证 full_paper 运行链路、attack runner、baseline、消融和 gate 产物是否能够打通; 正式 fpr=0.1 小样本论文闭合由 probe_paper_dataset 承担
probe_paper_dataset: 用于 FPR=10% 小样本论文闭合结果包, 使用 pilot_paper 级样本结构验证结论是否可写
pilot_paper_dataset: 用于小样本论文级结果包, 小规模跑代表性 paper 协议并产出 pilot 级论文结果
full_paper_dataset: 用于最终论文结果包, 只允许在全部前序 gate 通过后运行
stress_dataset: 用于强攻击、adaptive attack、跨模型和失败边界分析
```

每个 dataset 必须写出以下 manifest, 并且正式运行只能读取 manifest, 不能在运行中临时增删 prompt 或 seed:

```text
prompt_suite.json
seed_plan.json
content_manifest.json
generation_manifest.json
attack_manifest.json
baseline_manifest.json
split_manifest.json
```

### 21.2 prompt 构建原则

Prompt suite 必须覆盖静态、刚体运动、非刚体运动、相机运动、主体运动、复杂背景、低纹理、高纹理和遮挡等场景。每个 prompt 必须带有可审计元数据:

```text
prompt_id
prompt_text
prompt_family
motion_claim_role
motion_pattern_id
expected_motion_observability
foreground_scale_requirement
camera_motion_status
semantic_risk_tag
negative_contamination_risk
```

### 21.3 避免弱 prompt 的规则

弱 prompt 指生成模型可能稳定输出低运动、近静止或不可观测运动的视频, 从而让 motion / trajectory claim 被错误阻塞的 prompt。构建时必须执行以下规则:

1. 正向运动 prompt 必须显式包含 `large foreground object`、`visible displacement`、`across the frame`、`fixed camera` 或等价约束。
2. 对 rotation、bounce、slide、walk、drive、flow、splash 等运动类别, prompt 必须描述运动幅度、主体尺度和连续帧可见性。
3. 禁止使用 `gently rotates`、`subtle movement`、`slight motion` 等弱运动表述支撑 motion claim。此类 prompt 只能进入 ambiguous 或 stress split。
4. negative_static prompt 必须避免高频纹理、闪烁光源、时钟、棋盘、电视屏幕等容易造成伪运动的内容。
5. prompt 修复只能影响未来 run, 不能 retroactively 修改旧 records 的事实解释。

### 21.4 prompt 可观测性预审

在进入 probe_paper、pilot_paper 或 full_paper 前, 必须先运行 prompt observability audit。该 audit 只能使用视觉质量、运动可观测性和 prompt validity 字段, 禁止读取 `S_final`、`S_final_conservative` 或任何最终检测分数。

允许用于 prompt 过滤的字段包括:

```text
formal_visual_quality_ready
formal_motion_consistency_ready
formal_semantic_consistency_ready
motion_calibration_score
prompt_contamination_status
expected_motion_observability
```

禁止用于 prompt 过滤的字段包括:

```text
S_final
S_final_conservative
watermark_detection_score
claim_support_status
decision
```

### 21.5 full_paper 样本规模原则

full_paper 的目标是报告 `TPR@FPR=0.001`。因此 calibration negative 与 held-out test negative 必须远大于 1000 条。推荐最低事件规模为:

```text
calibration_negative_event_count >= 50000
heldout_test_negative_event_count >= 50000
heldout_attacked_positive_event_count >= 46000
negative_event_count_per_family >= 12500
calibration_negative_event_count_per_family >= 12500
heldout_negative_event_count_per_family >= 12500
attack_event_count_per_attack >= 1000
method_variant_event_count_per_primary_variant >= 500
```

这里的 event 是 `(video, attack, negative_family, method_variant, replay_setting)` 级记录。若 GPU 成本无法产生足够独立视频, 必须在 manifest 中区分 `unique_video_count` 与 `event_count`, 并在论文中报告置信区间、bootstrap 稳定性和 cluster-by-video robust interval。

### 21.6 split 与阈值冻结

`pilot_paper_dataset` 与 `full_paper_dataset` 必须使用同一协议顺序:

```text
calibration split -> frozen threshold artifact -> held-out test split -> tables / figures / claim audit
```

禁止事项:

1. 不得用 held-out test split 更新 threshold。
2. 不得用 small-scale pilot split 产生主论文表格。
3. 不得用 prompt 修复后的结果覆盖旧 run。
4. 不得在发现结果不好后增删 negative family 或 attack family。

## 22. full_paper 结果包门禁

`full_paper` 是论文结果包语义, 不是普通工程 package。只有所有前序验证通过后, 才允许运行 full_paper。

### 22.1 前置 gate

运行 full_paper 前必须全部满足:

```text
protocol_governance_foundation_passed = true
mechanism_validation_passed = true
paper_profile_gate_passed = true
pilot_paper_gate_passed = true
pilot_paper_to_full_paper_transition_decision_passed = true
data_split_and_leakage_guard_passed = true
external_baseline_self_containment_decision_passed = true
replay_and_authenticated_sketch_gate_passed = true
replay_and_authenticated_sketch_gate_passed_for_strong_claim = true
flow_specific_adaptive_attack_gate_passed = true
external_baseline_integration_ready = true
internal_ablation_matrix_ready = true
statistical_confidence_interval_decision_passed = true
artifact_rebuild_dry_run_passed = true
```

如果任一条件不满足, Codex 只能补齐前序流程, 不得产出 full_paper 论文结果包。

`replay_and_authenticated_sketch_gate_passed = true` 是所有正式 profile 的必需条件；任一 profile 都不能以 reduced-scope 或降级状态绕过 Claim-3。

### 22.2 full_paper 输出边界

full_paper 运行应输出到本地或远程运行目录, 不得写入 checked-in `outputs/`。正式归档必须由 packager 根据 records 与 manifests 生成。

必须产物包括:

```text
records/event_scores.jsonl
records/trajectory_traces.jsonl
records/thresholds.jsonl
records/baseline_scores.jsonl
records/ablation_scores.jsonl
tables/main_detection_table.csv
tables/baseline_comparison_table.csv
tables/ablation_table.csv
figures/tpr_at_fpr_figure.json
figures/trajectory_evidence_figure.json
reports/claim_audit_report.json
reports/readiness_summary.json
manifests/full_paper_package_manifest.json
```

### 22.3 full_paper 审计指标

full_paper 必须至少报告:

```text
TPR@FPR=0.01
TPR@FPR=0.001
clean_negative_fpr
attacked_negative_fpr
replay_negative_fpr
sampler_mismatch_negative_fpr
wrong_key_negative_fpr
path_marginal_gain_at_fixed_fpr
trajectory_payload_redundancy
quality_degradation
motion_degradation
semantic_degradation
generation_overhead
detection_overhead
```

## 23. 现代外部 baseline 来源登记

本节只规定 baseline 选择来源和工程接入要求, 不记录本项目是否已经运行。建议优先检查以下公开来源:

| baseline | 来源 | 工程接入备注 |
|---|---|---|
| VideoShield | https://github.com/hurunyi/VideoShield | 作为 2025 in-generation video diffusion watermark baseline。 |
| VidSig | https://github.com/hardenyu21/Video-Signature | 作为 latent video signature baseline 候选。 |
| VideoSeal | https://github.com/facebookresearch/videoseal | 作为开源 post-hoc neural video watermark baseline。 |

若 baseline 无法直接运行, 必须写出:

```text
external_baseline_name
external_baseline_source_url
external_baseline_runnable_status
external_baseline_not_run_reason
external_baseline_protocol_gap
external_baseline_result_used_for_claim = false
```

### 23.1 外部 baseline 接入层级

外部 baseline 不得只表现为一条命令或一个表格行。正式接入必须按以下层级闭合:

```text
source_registry
-> source_intake_manifest
-> source_inspection_manifest
-> clone_results_or_manual_command_gap
-> adapter
-> score_records
-> comparison_decision
-> execution_manifest
-> probe_paper / pilot_paper / full_paper gate
```

各层职责如下:

| 层级 | 产物 | 职责 | claim 边界 |
|---|---|---|---|
| source registry | `external_baseline/source_registry.json` | 登记 baseline 身份、来源、adapter 路径和 source 状态 | 只说明候选对象存在 |
| source intake | `external_baseline_intake_manifest.json` | 记录 source 目录、adapter 是否存在、官方命令环境变量是否配置 | 不支持论文 claim |
| source inspection | `external_baseline_source_inspection.json` | 记录第三方 source 中的入口、依赖和许可证候选文件 | 不支持论文 claim |
| clone results | `external_baseline_clone_results.json` | 记录 clone 计划、已执行 clone 或无法 clone 的原因 | 不支持论文 claim |
| table plan | `plans/external_baseline_table_plan.json` | 固定 baseline 在主表或 control 表中的角色 | 不支持论文 claim |
| repository-generated result cache preflight | `artifacts/external_baseline_official_result_bundle_preflight_decision.json` | 检查本项目 workflow 生成的官方结果缓存或可直接运行的官方资源是否覆盖当前 comparison unit | 不支持论文 claim, 但可作为严格门禁资源闭合证据 |
| adapter score records | `records/external_baseline_score_records.jsonl` | 在同一 run_root 上写出 measured_proxy 或 measured_formal records | 仅 measured_formal 可进入正式对比候选 |
| execution manifest | `artifacts/external_baseline_execution_manifest.json` | 记录 measured / formal 数量、evidence paths、source intake 路径和执行边界 | evidence paths 缺失时不能升级为正式主表 claim |


### 23.2 probe_paper 对 baseline 的硬阻断

`probe_paper` 是进入 `pilot_paper` 前的 FPR=10% 小样本论文闭合门禁, 因此 external baseline 条件不得只要求两个显式同步 control。该阶段必须检查:

```text
external_baseline_measured_adapter_count >= 5
modern_external_baseline_formal_measured_adapter_count >= 5
missing_modern_external_baseline_formal_adapter_names == []
external_baseline_execution_manifest_status == present
```

若现代视频水印 baseline 仍处于 `official_command_not_configured`、`manual_source_or_command_required` 或 `unsupported`, 则只能说明 baseline 接入尚未闭合, 不得判定 `probe_paper` 通过, 也不得进入 `pilot_paper` 结果运行, 也不得继续准备 `full_paper` claim。

对于无法在当前 Colab 会话中直接复跑的官方方法, 不允许要求外部补交结果。允许的闭环方式是:

```text
project_clone -> project_build -> project_run -> project_adapt -> project_record
```

若本项目 workflow 为避免重复计算而生成 repository-owned official result cache, 该缓存路径可以由 `SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT` 指向, 但它只能来自本项目 clone / build / run 官方代码后的输出, 不能来自人工补交的 JSON、NPZ 或论文表格数字。若官方资源、权重、checkpoint 或 maintained info 无法在项目流程内获得, checker 必须写出 `non_runnable_with_governed_reason`, 并阻断主表 claim。

## 24. Codex 构建执行手册

Codex 按本文档推进项目时, 应遵循以下顺序:

1. 读取 `.codex/project_contract.md` 与相关 `.codex/skills/`。
2. 检查 `docs/builds/sstw_phase_completion_status.md` 的当前阻塞项。
3. 只修改当前阻塞阶段所需的 repository modules、runner、checker、configs 或 docs。
4. 不跳过 `probe_paper`、`pilot_paper`、`full_paper_result_checker` 或三个轻量判定进入 full_paper。
5. 不把 proxy-only evidence 写成 supported claim。
6. 每次修改后运行 `pytest -q` 和 `python tools/harness/run_all_audits.py`。
7. 只有 full_paper 前置 gate 全部通过后, 才执行 full_paper 结果包流程。

## 25. 自动化门禁与 checker 实现要求

本文档中的每个关键 gate 最终都应有 repository checker 或 runner 支撑。若某个 gate 只停留在文档描述, Codex 不得把该 gate 标记为通过, 只能标记为:

```text
checker_not_implemented
records_not_available
manual_review_required
claim_not_supported
```

### 25.1 gate checker 层级

| checker 层级 | 作用 | 是否可支撑 full_paper |
|---|---|---|
| `structure_checker` | 检查目录、配置、字段和入口是否存在 | 否 |
| `mechanism_validation_checker` | 检查机制 records 与 small_scale_mechanism_pilot_check 记录是否足以进入 probe_paper | 否 |
| `paper_profile_checker` | 检查 probe_paper 小样本全流程、baseline、攻击、消融、CI、artifact rebuild 和 claim audit 是否闭合 | 否, 只允许在生成 probe_paper_to_pilot_paper_transition_decision 后进入 pilot_paper, 不能直接进入 full_paper |
| `probe_paper_gate` | 检查 target_fpr=0.1 小样本论文闭合结果是否具备可写论文的效果与优势证据 | 否, 只允许在生成 probe_paper_to_pilot_paper_transition_decision 后进入 pilot_paper |
| `pilot_paper_gate` | 小规模跑代表性 paper 协议, 检查 calibration、frozen threshold、held-out test、表格、图和 claim audit 是否闭合 | 是, 但评价等级仅为 pilot_paper |
| `full_paper_result_checker` | 检查 full_paper records、阈值、表格、图、报告和 claim audit | 是 |

### 25.2 三个轻量判定

三个轻量判定不新增重型实验阶段, 只在阶段跳转或 claim 升级前提供 fail-closed 判断:

```text
stage_transition_decision: 检查 source_stage、target_stage、上游 gate、profile 和 protocol config 是否允许跳转。
external_baseline_self_containment_decision: 检查 modern external baseline 是否全部由项目内 clone / build / run / adapt / record 产出 measured_formal records。
data_split_and_leakage_guard: 检查 calibration / held-out test / stress / ablation split 隔离、video identity 泄漏和 threshold 来源。
```

其中 `stage_transition_decision` 必须按跳转方向落盘为阶段明确的 decision artifact, 不能用一个泛化 PASS 同时代表多个跳转:

```text
probe_paper_to_pilot_paper_transition_decision
probe_paper_to_pilot_paper_transition_decision
pilot_paper_to_full_paper_transition_decision
full_paper_to_submission_freeze_transition_decision
```

这些跳转判定只能在 source gate PASS 后生成, 并由 target gate 消费。例如 `probe_paper_to_pilot_paper_transition_decision` 只能在 `probe_paper_gate` PASS 后生成, 再由 `pilot_paper_gate` 消费; 它不得作为 probe source gate 自身的 PASS 条件。

governed non-run record 只能作为 external baseline 阻断记录或 limitation 说明, 不能替代正式 `measured_formal` baseline。若三个轻量判定任一失败, 后续 `pilot_paper`、`full_paper` 或 `submission_package_freeze` 必须 fail closed。

### 25.3 pilot_paper gate 必须检查

```text
all_required_phase_decisions_exist
all_required_phase_decisions_passed_or_downgraded
method_mechanism_validation_passed
paper_profile_gate_passed

probe_paper_gate_passed
probe_paper_to_pilot_paper_transition_decision_passed
data_split_and_leakage_guard_passed
prompt_suite_manifest_frozen
seed_plan_manifest_frozen
generation_manifest_frozen
attack_manifest_frozen
baseline_manifest_frozen
ablation_manifest_frozen
threshold_manifest_frozen
calibration_negative_event_count_plan_sufficient
heldout_test_negative_event_count_plan_sufficient
attacked_positive_event_count_plan_sufficient
modern_external_baseline_plan_complete
adaptive_attack_plan_complete
replay_and_authenticated_sketch_plan_complete
artifact_rebuild_plan_complete
checked_in_outputs_blocked
calibration_split_to_frozen_threshold_to_heldout_test_split
pilot_paper_protocol_difference_from_full_paper == sample_scale_and_target_fpr_only
```

### 25.4 full_paper result checker 必须检查

```text
threshold_source_split == calibration
test_time_threshold_update_blocked == true
target_fpr includes 0.001
heldout_clean_negative_fpr <= target_fpr
heldout_attacked_negative_fpr <= target_fpr
heldout_replay_negative_fpr <= target_fpr
heldout_sampler_mismatch_negative_fpr <= target_fpr
wrong_key_negative_fpr <= target_fpr
full_method_beats_modern_external_baseline
full_method_beats_internal_ablation_baseline
path_marginal_gain_at_fixed_fpr > 0
trajectory_payload_redundancy <= preset_limit
quality_degradation_within_limit == true
motion_degradation_within_limit == true
semantic_degradation_within_limit == true
claim_audit_passed == true
artifact_rebuild_passed == true
```

### 25.5 checker 失败处理

任何 checker 失败时, 应输出:

```text
blocking_stage
blocking_requirement
observed_value
required_value
recommended_next_action
claim_support_status
full_paper_allowed = false
```

禁止把 checker 失败解释为“可以人工放行”。若必须降级, 应显式写入 claim 降级记录, 并从 full_paper supported claims 中移除对应主张。

## 26. 统计报告与置信区间要求

顶会论文不能只报告单点 TPR / FPR。所有 full_paper 主结果必须报告统计稳定性。

### 26.1 必须报告的区间

```text
binomial_confidence_interval_for_fpr
binomial_confidence_interval_for_tpr
bootstrap_confidence_interval_for_tpr_at_fpr
cluster_by_video_confidence_interval
per_attack_family_confidence_interval
per_negative_family_confidence_interval
per_prompt_family_confidence_interval
```

其中, `cluster_by_video_confidence_interval` 用于避免同一视频被多个 attack / method variant 扩展后造成事件数虚高。

### 26.2 论文主表最低统计单元

主表不得只按 event 统计, 必须同时报告:

```text
unique_video_count
event_count
unique_prompt_count
unique_seed_count
attack_family_count
negative_family_count
baseline_count
method_variant_count
```

### 26.3 FPR=0.001 的样本量解释

若 held-out negative event 小于 50000, 可以继续运行 validation, 但不得把 `TPR@FPR=0.001` 写成 full_paper 主 claim。此时只能写为:

```text
low_fpr_validation_incomplete
sample_size_insufficient_for_fpr_0_001_claim
```

## 27. 审稿风险对照矩阵

为了减少“实验不足被拒稿”, full_paper 前必须逐项回答以下审稿风险。

| 审稿风险 | 必须证据 | 未满足时的处理 |
|---|---|---|
| baseline 不足 | modern external baseline records 和 baseline comparison table | 不进入 full_paper |
| 消融不足 | internal ablation table 覆盖 injection、path、state、admissibility、replay | 不进入 full_paper |
| 低 FPR 样本不足 | FPR=0.001 大规模 held-out negative 和 CI | 降级为 validation 结果 |
| prompt cherry-picking | frozen prompt manifest、prompt observability audit、旧 run 保留 | 不允许手工替换主表样本 |
| attack 不足 | spatial、temporal、generative、Flow-specific adaptive attacks | adaptive claim 降级 |
| replay 伪证据 | replay negative、wrong sampler、wrong prompt、uncertainty records | 当前 paper profile 失败 |
| 服务端日志质疑 | authenticated trajectory sketch verification | 当前 paper profile 失败 |
| 质量退化 | FVD / LPIPS / SSIM / CLIP / motion metrics | 降低强度或降级方法 |
| 结果不可复现 | records、manifests、rebuild commands、code version | 不进入 submission freeze |

## 28. full_paper 执行顺序

full_paper 必须按以下顺序执行, 不得并行跳过 gate。`probe_paper` 承担 FPR=10% 小样本论文闭合职责; `pilot_paper` 承担 FPR=1% 中等规模论文协议结果职责。二者均不能被 full_paper 运行后补造。

```text
1. verify_probe_paper_gate_passed
2. verify_pilot_paper_gate_passed
3. freeze_full_paper_manifests
4. run_generation
5. run_attacks
6. run_replay_or_sketch_verification
7. run_detection
8. run_external_baselines
10. run_internal_ablations
11. run_adaptive_attacks
12. freeze_thresholds_from_calibration
13. evaluate_heldout_test
14. build_tables_figures_reports
15. run_claim_audit
16. run_artifact_rebuild_audit
17. package_full_paper
```

若任一步失败, 后续步骤只能生成 diagnostic package, 不得生成 full_paper result package。

## 29. 大规模 full_paper 可运行性预演与分片执行

full_paper 目标包含 `TPR@FPR=0.001` 级别评估, 因此不能把一次超大规模运行作为首次端到端验证。Codex 必须在 full_paper 前至少完成 `probe_paper`, 用 probe_paper protocol config 指定的 target_fpr 小样本论文闭合验证证明数据、模型、攻击、检测、baseline、消融、统计和打包链路不会阻断, 并产出 FPR=10% 小样本论文闭合结果; 再通过 `pilot_paper` 产出由 pilot_paper protocol config 指定 target_fpr 的中等规模论文协议结果。

### 29.1 分级预演规模

推荐采用以下分级:

```text
method_mechanism_validation: 验证配置、schema、路径、模型加载、单 shard 输出、方法机制和 packager
probe_paper: FPR=10% 小样本论文闭合验证, 覆盖 baseline、ablation、adaptive attack、replay、CI 和 artifact rebuild, 并判断结论是否可写
pilot_paper: FPR=1% 小规模代表性 paper 协议运行并产出 pilot 级论文结果
full_paper: 只在全部前置 gate 通过后执行
```

`method_mechanism_validation` 只用于排查链路和判断是否进入 paper profile。`probe_paper` 可以产出 FPR=10% 条件下的完整三层论文结论, `pilot_paper` 可以产出 FPR=1% 条件下的完整三层论文结论; 二者不能替代 `full_paper` 的 FPR=0.1% 主表, 也不能向更低 FPR 外推。

`pilot_paper_rehearsal` 在当前项目中升级为 `pilot_paper`。`pilot_paper` 是小规模跑代表性 paper 协议并产出 pilot 级论文结果的正式阶段。它可以报告 pilot_paper 级 `TPR@FPR=0.01` 论文主张, 前提是它采用与 full_paper 一致的数据切分、阈值冻结和 artifact rebuild 方式:

```text
calibration split
-> frozen threshold artifact
-> held-out test split
-> tables / figures / claim audit
```

当前工程采用十倍递增的独立视频规模: `probe_paper` 为 10 个 prompt × 6 个 seed = 60 个视频, calibration/test 各30个; `pilot_paper` 为 50 个 prompt × 12 个 seed = 600 个视频, calibration/test 各300个; `full_paper` 为 200 个 prompt × 30 个 seed = 6000 个视频, calibration/test 各3000个。三个 paper profile 均显式拆分 calibration / test seed, 且 attack manifest、baseline、消融、图表和打包协议必须同构; 差异只允许是样本规模、统计置信度和 target FPR。`probe_paper` 和 `pilot_paper` 结论仍不能外推为 `TPR@FPR=0.001` 或 full_paper 规模主表结果。

### 29.2 分片执行协议

full_paper 必须支持 shard 化执行, 避免单次长任务失败导致全量重跑。每个 shard 必须有独立 manifest:

```text
shard_id
shard_role
split
prompt_id_range
seed_id_range
attack_family_range
baseline_subset
method_variant_subset
expected_record_count
actual_record_count
shard_status
shard_checksum
resume_policy
artifact_write_scope
```

分片写入必须满足:

1. 同一个 `sample_id` 和 `method_variant` 的正式 record 只能由一个 shard 负责。
2. shard 重跑必须覆盖同一 shard 输出, 不能追加重复 records。
3. 所有 shard 合并前必须执行 schema audit、duplicate audit、split audit 和 threshold-source audit。
4. 合并后的 records 才能进入 tables、figures 和 reports rebuild。

### 29.3 pilot_paper 与 full_paper 前资源与阻断预检

pilot_paper gate 与 full_paper 启动前除了实验协议外, 还必须检查:

```text
model_cache_ready
gpu_memory_budget_ready
disk_free_space_ready
google_drive_or_remote_storage_ready
dependency_lock_ready
dataset_manifest_readable
output_write_scope_not_checked_in
records_schema_version_compatible
packager_manifest_schema_compatible
resume_from_interrupted_shard_supported
```

若资源预检失败, 只能生成 blocking report, 不允许启动生成任务。

### 29.4 失败恢复规则

允许恢复的失败:

```text
single_shard_runtime_timeout
single_shard_upload_interruption
single_baseline_adapter_timeout
single_attack_runner_timeout
non_claim_supporting_metric_missing
```

不允许自动恢复并继续 full_paper 的失败:

```text
threshold_source_split_violation
test_time_threshold_update_detected
schema_incompatible_records
duplicate_claim_records
prompt_manifest_changed_after_freeze
baseline_manifest_changed_after_freeze
attack_manifest_changed_after_freeze
claim_audit_failed
artifact_rebuild_failed
```

第二类失败必须回到对应前置阶段修复, 不能通过删除失败 records 或手工改表解决。

## 30. 外部 baseline 选择与公平比较协议

顶刊顶会版本不能只比较弱 baseline。外部 baseline 必须覆盖“机制接近性”和“审稿熟悉度”两个维度。

### 30.1 baseline 层级

推荐 baseline 层级如下:

| baseline 层级 | 作用 | 是否可作为主外部对比 |
|---|---|---|
| in_generation_video_watermark | 与生成过程内嵌水印最接近, 用于主对比 | 是 |
| diffusion_or_flow_video_watermark | 与视频扩散或 Flow sampler 机制相关, 用于主对比或强补充 | 是 |
| post_hoc_neural_video_watermark | 代表公开视频水印强基线, 用于鲁棒性和质量对比 | 是, 但不能单独作为唯一 baseline |
| image_or_frame_watermark_extended_to_video | 代表传统弱 baseline, 用于说明视频时序挑战 | 否 |
| explicit_temporal_alignment_control | 证明 SSTW 不是普通显式时间同步 | 否 |
| endpoint_only_or_latent_only_control | 证明 SSTW 不是 endpoint-only 方法 | 否 |

### 30.2 最低 baseline 组合

full_paper 至少应包含:

```text
one_in_generation_or_diffusion_video_watermark_baseline
one_post_hoc_neural_video_watermark_baseline
one_explicit_temporal_alignment_control
one_endpoint_only_control
one_generic_state_space_or_temporal_aggregator_control
```

如果无法运行某个现代 baseline, 必须提供 governed non-run record, 并说明协议不兼容原因。不能把无法运行的现代方法从论文对比中静默删除, 也不能用 non-run record 替代正式 `measured_formal` 主表结果。

### 30.3 公平比较要求

所有 baseline 必须尽量共享以下输入条件:

```text
same_video_resolution_or_declared_resize_policy
same_video_duration_or_declared_clip_policy
same_attack_manifest
same_clean_negative_split
same_attacked_negative_split
same_quality_metric_suite
same_threshold_source_split_policy
same_target_fpr_reporting
```

若 baseline 原论文只支持不同输入格式, 需要写入:

```text
external_baseline_input_compatibility_status
external_baseline_protocol_gap
external_baseline_adapter_status
external_baseline_result_used_for_claim
```

### 30.4 baseline 结果使用边界

只有同时满足以下条件的 baseline 结果, 才能进入主表:

```text
external_baseline_runnable_status == runnable
external_baseline_output_record_status == governed_records_written
external_baseline_project_clone_status == completed
external_baseline_project_build_status == completed
external_baseline_project_run_status == completed
external_baseline_project_adapt_status == completed
external_baseline_project_record_status == completed
metric_status == measured_formal
external_baseline_threshold_policy_compatible == true
external_baseline_attack_manifest_compatible == true
external_baseline_result_used_for_claim == true
```

若 baseline 只能以论文报告数值、外部补交结果或非同协议结果引用, 只能放入 related work 或 supplementary discussion, 不能作为主表胜负证据。

## 31. 内部消融矩阵

内部消融必须证明每个算法原语都有必要性, 不能只证明 full method 分数较高。

### 31.1 必须包含的消融变体

```text
sstw_full_method
without_velocity_field_weak_constraint
endpoint_only_control
trajectory_only_control
without_endpoint_aware_minimum_energy_control
without_time_reparameterization_invariant_observation
without_replay_uncertainty
without_flow_state_admissibility
key_agnostic_state_space_control
generic_state_space_control
mean_temporal_aggregator_control
explicit_temporal_alignment_baseline
trajectory_time_shuffled_control
trajectory_key_shuffled_control
wrong_sampler_replay_control
wrong_prompt_replay_control
without_quality_guard
```

### 31.2 每个消融必须报告的指标

```text
TPR@FPR=0.01
TPR@FPR=0.001
clean_negative_fpr
attacked_negative_fpr
replay_negative_fpr
path_marginal_gain_at_fixed_fpr
trajectory_payload_redundancy
quality_degradation
motion_degradation
semantic_degradation
detection_overhead
generation_overhead
```

### 31.3 消融失败解释

若某个消融与 full method 差异不显著, 不能简单删除该消融。必须生成:

```text
ablation_failure_analysis_report
effect_size_with_confidence_interval
power_analysis_or_sample_size_note
claim_downgrade_recommendation
```

这用于避免审稿人认为该算法原语只是工程堆叠。

## 32. 审稿证据索引与 rebuttal-ready package

submission package 必须包含一个面向审稿问题的证据索引。该索引的作用不是重写论文结论, 而是把每个潜在质疑直接映射到 governed artifacts。

### 32.1 evidence index 字段

```text
reviewer_question_id
reviewer_question_category
paper_claim_id
required_evidence_artifact
supporting_record_path
supporting_table_path
supporting_figure_path
supporting_report_path
supporting_manifest_path
evidence_status
claim_downgrade_if_missing
```

### 32.2 必须覆盖的审稿问题

```text
why_not_endpoint_only
why_not_post_hoc_video_watermark
why_not_explicit_temporal_alignment
why_flow_matching_specific
why_low_fpr_result_is_reliable
why_negative_tail_is_controlled
why_prompt_suite_is_not_cherry_picked
why_external_baselines_are_sufficient
why_ablation_is_sufficient
why_quality_degradation_is_acceptable
why_replay_or_sketch_evidence_is_trustworthy
why_results_are_reproducible
```

### 32.3 不允许进入 rebuttal-ready package 的内容

```text
manual_table_edits
unsupported_claims
placeholder_supported_claims
test_split_threshold_tuning
unregistered_external_numbers
screenshots_without_records
```

## 33. Codex 任务执行模板

当 Codex 按本文档推进项目时, 每次任务应输出或更新以下信息:

```text
current_stage
blocking_gate
files_changed
records_or_manifests_created
tests_run
harness_audits_run
full_paper_allowed
next_allowed_action
next_forbidden_action
claim_support_status
```

若本轮只修改文档, `records_or_manifests_created` 应写为 `none`, 并明确说明没有产出 full_paper 或 submission 结果包。

## 34. full_paper 工程门禁实现规范

本文档定义的是项目总体流程。具体 checker、runner、reporter 和 builder 的实现接口, 必须以以下文档为工程规范:

```text
docs/builds/sstw_full_paper_engineering_gate_spec.md
```

该规范把以下仍需工程化的门禁、runner、reporter 和轻量判定拆解为可实现组件:

```text
paper_profile_gate
pilot_paper_gate
modern_external_baseline_runner
flow_specific_adaptive_attack_runner
statistical_confidence_interval_reporter
full_paper_result_checker
reviewer_evidence_index_builder
stage_transition_decision
external_baseline_self_containment_decision
data_split_and_leakage_guard
```

这些组件已经具有真实工程入口, 但没有真实 GPU records 时仍必须保持失败关闭。若实际实现与规范不一致, 必须同步更新规范文档、阶段文档和 tests, 不能只修改代码。

在完整真实实验结果尚未生成前, full_paper 仍必须保持:

```text
full_paper_allowed = false
submission_freeze_allowed = false
```

### 34.1 操作手册执行闭环缺口

当前操作手册已经定义 probe_paper、pilot_paper 和 full_paper 的语义顺序, 但仍必须显式保留以下工程缺口, 防止把文档完整误判为执行闭环已经完成:

```text
paper_profile_gate_figure_builder: implemented_in experiments/generative_video_model_probe/paper_profile_artifact_package.py
probe_paper_package_manifest_builder: implemented_in experiments/generative_video_model_probe/paper_profile_artifact_package.py
stage_transition_decision_common_checker: implemented_in scripts/check_results/stage_transition_decision.py
probe_paper_to_pilot_paper_transition_decision: implemented_in scripts/check_results/stage_transition_decision.py
pilot_paper_to_full_paper_transition_decision: implemented_in scripts/check_results/stage_transition_decision.py
full_paper_to_submission_freeze_transition_decision: implemented_in scripts/check_results/stage_transition_decision.py
external_baseline_self_containment_decision: implemented_in scripts/check_results/external_baseline_self_containment_decision.py
data_split_and_leakage_guard: implemented_in scripts/check_results/data_split_and_leakage_guard.py
full_paper_result_checker: implemented_in scripts/check_results/full_paper_result_checker.py
reviewer_evidence_index_builder: implemented_in experiments/generative_video_model_probe/reviewer_evidence_index.py
external_baseline_project_clone_manifest: checked_by external_baseline_self_containment_decision, real Colab evidence still pending until measured_formal run
external_baseline_project_build_manifest: checked_by external_baseline_self_containment_decision, real Colab evidence still pending until measured_formal run
external_baseline_project_run_manifest: checked_by external_baseline_self_containment_decision, real Colab evidence still pending until measured_formal run
external_baseline_project_adapt_manifest: checked_by external_baseline_self_containment_decision, real Colab evidence still pending until measured_formal run
external_baseline_project_record_manifest: checked_by external_baseline_self_containment_decision, real Colab evidence still pending until measured_formal run
```

这些缺口的处理规则是: 可以继续完善 repository runner、checker、reporter 和 builder, 但不得用手工文件、外部补交结果或 Notebook 临时变量填补正式 evidence path。


## 35. external_baseline 接入流程

外部 baseline 必须通过 `external_baseline/` 适配边界进入本项目, 不允许把外部论文数字、临时脚本输出、外部补交结果或 Notebook 手工结果直接写入主表。正式 baseline 证据必须由本项目完成:

```text
project_clone
project_build
project_run
project_adapt
project_record
```

### 35.1 接入层级

```text
source_registry_layer:
  path: external_baseline/source_registry.json
  role: 登记 baseline 身份、源码状态、adapter 路径和 claim 边界

adapter_layer:
  path: external_baseline/primary/<baseline_id>/adapter/run_sstw_eval.py
  required_functions: adapter_status, build_score_records
  role: 把外部方法输出或显式 control 输出映射为统一 score records

self_contained_execution_layer:
  required_steps: project_clone, project_build, project_run, project_adapt, project_record
  role: 记录第三方官方代码、依赖、权重、命令、输出和 adapter 转换状态

experiment_scheduler_layer:
  path: experiments/generative_video_model_probe/external_baseline_runner.py
  role: 调度 adapter, 写出 records、tables、artifacts 和 reports

gate_and_package_layer:
  paths:
    - experiments/generative_video_model_probe/paper_profile_gate.py
    - experiments/generative_video_model_probe/validation_artifact_rebuild.py
    - scripts/package_results/generative_video_drive_packager.py
  role: 只读取已落盘 artifacts, 不重新解释 baseline 分数
```

### 35.2 adapter 输入约束

adapter 只能读取 `run_root` 中已受治理的输入:

```text
records/runtime_detection_records.jsonl
records/trajectory_trace.jsonl
records/external_baseline_records.jsonl
artifacts/runtime_detection_decision.json
```

adapter 不得读取 Notebook cell 临时变量, 不得使用人工表格, 不得用 `S_final` 或最终判定分数进行污染过滤。

### 35.3 adapter 输出约束

adapter 必须输出统一 comparison score records:

```text
records/external_baseline_score_records.jsonl
tables/external_baseline_comparison_table.csv
artifacts/external_baseline_comparison_decision.json
reports/external_baseline_comparison_report.md
```

当前显式 DTW 与 frame matching 只能作为同步 control proxy。现代视频水印 baseline 在 adapter 未接入时必须写为 `unsupported`, 不能被解释为 SSTW 胜出。

### 35.4 full_paper 前置条件

进入 full_paper baseline claim 前必须满足:

```text
modern_external_baseline_adapter_integrated
modern_external_baseline_measured_records_ready
external_baseline_project_clone_status == completed
external_baseline_project_build_status == completed
external_baseline_project_run_status == completed
external_baseline_project_adapt_status == completed
external_baseline_project_record_status == completed
common_prompt_protocol_ready
common_attack_protocol_ready
common_threshold_policy_ready
external_baseline_result_used_for_claim_governed
```

在上述条件未满足前, `external_baseline_comparison_decision = PASS` 只表示工程产出链路闭合, 不表示论文主 claim 成立。

### 35.5 Colab-only 真实 baseline 执行约束

现代视频水印 baseline 的真实执行统一放在 Colab 或等价 GPU 运行环境中完成。本地仓库只允许执行以下轻量工程任务:

```text
adapter 接口检查
source registry / source intake manifest 生成
schema 与字段审计
validation gate 逻辑测试
Notebook 入口静态检查
packager 轻量回归测试
```

本地不得把第三方 baseline 的重型推理结果伪造成正式记录。Colab 冷启动时必须完成:

```text
clone / pull SSTW 仓库
安装 SSTW 依赖
通过本项目 clone / build / run 流程准备现代 baseline 官方实现
配置 5 个主实验现代 baseline 自包含 adapter 命令
可选执行 source clone
绑定 project-generated external baseline provenance paths
运行 runtime detection 后执行 external_baseline_runner
打包 records / artifacts / reports / manifests 到 Google Drive
```

当 `PROFILE` 为以下任一阶段时:

```text
probe_paper
pilot_paper
full_paper
```

Notebook 必须在真实 GPU 生成前检查现代 baseline command 是否全部配置。若缺少任何一个 command, 应提前阻断, 不允许先生成缺 baseline 的混淆结果包。该规则属于项目特定写法, 目的是保证 `probe_paper` 作为 FPR=10% 小样本论文闭合层时已经能够产出完整 baseline comparison 结果。

该检查不得散落在 Notebook cell 中手工维护 baseline 清单。Notebook 应调用:

```text
workflows/generative_video_paper.py::build_modern_baseline_command_env
workflows/generative_video_paper.py::write_external_baseline_colab_preflight_decision
workflows/generative_video_paper.py::validate_modern_baseline_commands_for_profile
```

并在阻断前写出:

```text
artifacts/external_baseline_colab_preflight_decision.json
artifacts/external_baseline_command_template_summary.json
artifacts/external_baseline_official_bridge_preflight_decision.json
artifacts/external_baseline_official_resource_bootstrap_decision.json
artifacts/external_baseline_official_bundle_generation_decision.json
artifacts/external_baseline_official_result_bundle_preflight_decision.json
```

其中 `external_baseline_colab_preflight_decision.json` 只记录 Colab 冷启动 preflight 状态, 不运行第三方 baseline, 不支持论文 claim。它的作用是让用户在 Google Drive 中直接看到缺少哪些现代 baseline command, 避免 Colab 断开后无法定位失败原因。

`external_baseline_command_template_summary.json` 由以下配置生成:

```text
configs/external_baselines/modern_baseline_colab_commands.json
```

该配置记录联网核验后的现代 baseline 官方仓库 URL、branch HEAD commit、Colab clone 目录、官方入口候选脚本、SSTW bridge command 模板和 repository official adapter command 模板。它属于配置辅助层, 不属于正式 baseline 结果层。Notebook 可以用 repository official adapter 自动补齐 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`, 但这只表示“命令入口已配置”, 不表示“官方 baseline 已测量”。只有该命令真实调用官方源码/API, 或读取由本项目 workflow 生成且带 `official_execution_manifest_path` 的 official bundle cache, 并写出 score JSON 后, 才能生成 `measured_formal` records。

现代 baseline 从“源码已核验”进入“正式 measured_formal 对比”至少需要满足:

```text
official_source_or_weights_available
sstw_eval_wrapper_exists
SSTW_<BASELINE>_EVAL_COMMAND points_to_real_wrapper
SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND points_to_official_source_or_repository_fail_closed_adapter
wrapper_writes_output_json_with_score
external_baseline_command_manifest_written
external_baseline_score_records.metric_status == measured_formal
```

这一区分是项目特定约束, 用于防止把“已联网找到 baseline 仓库”误解释为“baseline comparison 已经完成”。

### 35.6 repository bridge command 规则

为了减少 Colab 中为 5 个主实验现代 baseline 重复编写 SSTW 外层 I/O wrapper 的成本, 项目提供统一 bridge:

```text
external_baseline/official_command_bridge.py
```

bridge 的职责是:

```text
读取 source_video_path / attacked_video_path / attack_name
调用用户配置的官方 baseline 命令
读取官方输出 JSON
归一化为 SSTW command adapter 接受的 output_json_path
```

bridge 不是第三方 baseline 算法本体。它不能自行计算视频相似度、不能读取 SSTW `S_final`,
也不能在缺少官方命令时输出替代分数。正式运行时必须为每个 baseline 提供内部官方命令:

```text
SSTW_VIDEOSHIELD_OFFICIAL_EVAL_COMMAND
SSTW_VIDSIG_OFFICIAL_EVAL_COMMAND
SSTW_VIDEOSEAL_OFFICIAL_EVAL_COMMAND
```

内部官方命令必须把官方输出写入:

```text
{official_output_json_path}
```

当前项目提供 5 个主实验 repository official adapter:

```text
external_baseline/official_eval_adapters/videoshield.py
external_baseline/official_eval_adapters/vidsig.py
external_baseline/official_eval_adapters/videoseal.py
```

这些 adapter 的职责是把官方仓库源码、官方 API、官方 checkpoint 或项目内 official bundle cache 转换为 SSTW 统一 JSON。它们不是替代 baseline。缺少官方源码、权重、key、message、maintained info 或项目内 official bundle cache 时必须 fail closed, 不能产生 proxy 分数。若官方仓库提供更合适的原生命令, 用户可以通过 `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND` 覆盖单个 adapter 的内部执行逻辑。

bridge 再把该输出归一化写入:

```text
{output_json_path}
```

Notebook 默认可通过 `SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS=true` 使用 repository bridge
外层命令。若使用 bridge, `external_baseline_official_bridge_preflight_decision.json` 必须在
真实生成前检查 5 个主实验 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND` 是否齐全。若用户希望完全自定义
外层命令, 可以设置 `SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS=0`, 但此时仍必须保证
`SSTW_<BASELINE>_EVAL_COMMAND` 直接写出合规 score JSON。

命令优先级约束为:

```text
显式 SSTW_<BASELINE>_EVAL_COMMAND
  > baseline_id 对应的 repository bridge 模板
```

因此, 用户也可以在保持 `SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS=true` 的情况下, 只对某些
baseline 提供直接外层命令。preflight 必须按 baseline 逐项判断: 使用 bridge 的 baseline
要求 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`, 使用直接外层命令的 baseline 不再要求内部官方命令。

`external_baseline_execution_manifest.json` 必须记录:

```text
source_intake_manifest_path
external_baseline_measured_adapter_count
modern_external_baseline_formal_measured_adapter_count
modern_external_baseline_formal_measured_adapter_names
formal_evidence_status
evidence_paths
```

其中 evidence paths 应指向 Colab 或 Google Drive 中实际存在的官方 baseline 运行日志、配置、输出 JSON 或依赖快照。没有 evidence paths 的 formal rows 只能作为工程接入证据, 不得直接升级为论文主表 claim。

现代 baseline command adapter 还必须把每条官方命令的输出自动持久化到当前 `run_root`:

```text
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_output.json
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_stdout.txt
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_stderr.txt
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_command_manifest.json
```

这些文件属于 Colab 真实运行证据, 会被 `external_baseline_execution_manifest.json` 自动收集到 `evidence_paths`。这样即使 Colab 运行环境断开, Google Drive package 仍保留每条 `measured_formal` external baseline score 的官方输出来源。该机制属于项目特定写法, 目的是防止现代 baseline 对比退化为只保留聚合分数的不可审计表格。

### 35.7 官方资源 bootstrap 与 official bundle 自动生成规则

probe_paper、pilot_paper 与 full_paper 的外部 baseline 流程不得停留在“只检查缺什么”。在
runtime detection records 已经落盘后, Notebook 必须按以下顺序推进:

```text
external_baseline_source_intake
-> external_baseline_official_resource_bootstrap
-> external_baseline_official_bundle_generation
-> external_baseline_official_result_bundle_preflight
-> external_baseline_comparison
-> formal_comparison_scoring_colab 重新聚合 5 个主实验 official reference 阶段包
-> paper_evidence_postprocess_colab 生成最终门禁前辅助证据
-> paper_gate_and_package_colab 执行最终门禁
```

其中:

1. `external_baseline_official_resource_bootstrap` 尝试自动安装或下载公开可获得的官方资源,
   例如 VideoSeal 官方 API 依赖、VidSig 公开 checkpoint 等。
2. `external_baseline_official_bundle_generation` 只对当前仓库可以真实调用官方 API 或
   项目内官方流程运行器的 baseline 生成 repository-owned official bundle cache。当前可自动尝试
   官方生成流程得到 baseline 自己的 watermarked videos, 再施加项目 runtime attack 并调用官方
   反演 / 检测逻辑, 不允许直接检测 SSTW / Wan 视频后伪造成 baseline 结果。
3. 对需要高显存生成模型、训练得到的 extractor、PRC key、maintained info 或官方中间产物的
   baseline, 自动流程必须写出 `non_runnable_with_governed_reason` 或 `manual_official_resource_required`, 不得用 SSTW 的
   `S_final`、最终判定分数、视频相似度或随机数生成替代分数。
4. `external_baseline_official_result_bundle_preflight` 只能在官方资源或本项目 workflow 生成的官方结果缓存覆盖当前
   comparison unit 时通过。失败时必须保留缺口 artifact, 作为 probe_paper 阻断依据。
5. `external_baseline_formal_reference_*` 只有在对应
   `<baseline_id>_formal_reference_decision.json` 的 `formal_reference_decision` 为 `PASS`
   时, 才能发布完整时间戳 zip。若结果为 `FAIL` 或决策文件缺失, 只能发布
   manifest-only 阻断记录, 且不得发布可被后续门禁恢复的 zip。
6. 单 baseline Notebook 中的 comparison records 只用于即时自检。最终论文 gate 必须在
   `formal_comparison_scoring_colab.ipynb` 恢复 5 个主实验 official reference 阶段包后重新运行
   `external_baseline_comparison`, 生成全量 `measured_formal` records、self-containment 判定、公平校准和差值区间统计。

该规则的目标是让 Colab 冷启动具备“能自动补齐的就自动补齐”的工程能力, 同时保持论文
baseline 对比的 fail-closed 边界。严格门禁通过的前提仍然是 5 个主实验现代 baseline 最终都由本项目产出
`measured_formal` records, 且 evidence path 可以追溯到官方源码、官方 API、官方 checkpoint
或本项目生成的官方结果缓存。

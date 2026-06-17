# sampling_time_constraint_probe 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段验证 sampling-time weak constraint 是否真正进入 Flow Matching 采样动力学, 并与 endpoint-aware minimum-energy control、quality guard、semantic projection 和 flow velocity proxy 形成可审计证据链。该阶段应在 `flow_model_adapter_preflight` 确认模型 callback、time grid 和 velocity / displacement proxy 可用之后进行。

### 1.2 输入

```text
configs/protocol/sampling_time_constraint_preflight.json
configs/generation/sampling_constraint.json
configs/generation/lambda_schedules.json
configs/generation/generation_models.json
main/generation/
experiments/sampling_time_constraint/
paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb
```

### 1.3 构建任务

1. 在采样 callback 或 scheduler adapter 中接入 velocity-field weak watermark constraint。
2. 记录 callback latent displacement 或 velocity proxy。
3. 执行 endpoint-aware minimum-energy flow control。
4. 执行 quality guard 和 semantic projection。
5. 比较 no constraint、endpoint-only、constant lambda、wrong key 和 without semantic projection 等 control。
6. 将结果打包到 Google Drive 目录, 供 checker 与 submission freeze 使用。

### 1.4 必须比较

```text
no_constraint_control
endpoint_only_constraint
constant_lambda_constraint
no_endpoint_aware_control
no_semantic_projection
wrong_key_constraint
```

### 1.5 必须记录字段

```text
flow_velocity_proxy_available
flow_velocity_proxy_source
flow_velocity_alignment_before_constraint
flow_velocity_alignment_after_constraint
flow_velocity_alignment_gain
endpoint_consistency_score
quality_guard_status
semantic_projection_status
lambda_schedule_id
sampling_constraint_variant
```

### 1.6 通过标准

1. keyed flow velocity alignment gain 大于 baseline。
2. endpoint payload 或 endpoint evidence 与路径证据一致。
3. quality guard 没有被关闭或绕过。
4. semantic projection 有可审计字段或明确 placeholder 边界。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有 sampling-time constraint 相关模块:

```text
main/generation/sampling_constraint_adapter.py
main/generation/velocity_projection_constraint.py
main/generation/lambda_schedule.py
main/generation/scheduler_adapter.py
main/generation/trajectory_capture.py
experiments/sampling_time_constraint/colab_runtime.py
experiments/sampling_time_constraint/runner.py
experiments/sampling_time_constraint/postprocess_runner.py
scripts/check_results/sampling_time_constraint_colab_result_checker.py
scripts/package_results/sampling_time_constraint_drive_packager.py
paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb
```

### 2.2 当前阶段补充要求

按照新的整体流程, 本阶段需要显式覆盖:

```text
endpoint_aware_minimum_energy_flow_control
flow_velocity_proxy_record
callback_latent_displacement_record
quality_guard_status
semantic_projection_status
```

如果 velocity constraint 不能提供有效增益, 不能宣称 SSTW 是 Flow Matching trajectory watermark。

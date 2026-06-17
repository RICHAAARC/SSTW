# flow_model_adapter_preflight 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段在进入真实 GPU 生成实验前, 确认目标 Flow Matching / Rectified Flow 视频生成模型是否能够暴露、记录和复现 SSTW 所需的 trajectory proxy。该阶段的主要风险不是检测分数是否足够高, 而是模型接口是否能提供后续方法机制所需的可审计轨迹证据。

### 1.2 输入

```text
configs/generation/generation_models.json
configs/generation/prompts.json
configs/generation/seeds.json
configs/generation/scheduler.json
main/generation/model_registry.py
main/generation/scheduler_adapter.py
main/generation/latent_capture.py
main/generation/trajectory_capture.py
experiments/sampling_time_constraint/colab_runtime.py
paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb
```

### 1.3 主线模型

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

轻量模型只能作为接口预验证或 fallback probe, 不能单独支撑完整 Flow Matching trajectory watermark 主 claim。

### 1.4 必须实现

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

### 1.5 必须检查

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

### 1.6 必须记录字段

```text
generation_model_id_placeholder
sampler_id_placeholder
sampler_signature_placeholder
trajectory_source_level
trajectory_source_status
trajectory_source_unavailable_reason
flow_velocity_proxy_available
flow_velocity_proxy_source
replay_scheduler_id_placeholder
replay_time_grid_id_placeholder
```

### 1.7 通过标准

1. `trajectory_trace_capture_success_rate >= 0.95`。
2. 每个样本均能记录 sampler signature 与 time grid。
3. 至少一种 velocity proxy 或 latent displacement proxy 可用。
4. 生成结果可由 prompt、seed、model id、sampler id 与 config 复现。
5. 若 velocity field 原始值不可访问, 必须记录 `flow_velocity_proxy_available=false` 与 proxy 类型, 不得伪称拥有真实 velocity field。

### 1.8 失败处理

若本阶段失败, 不得进入 `sampling_time_constraint_probe` 与 `generative_video_model_probe` 的主实验。应优先修复 backend adapter、callback、scheduler hook 或选择替代 Flow Matching / velocity-field 模型。

## 2. 当前阶段具体完成情况

### 2.1 已有工程基础

当前仓库已经存在 generation model registry、scheduler adapter、latent capture、trajectory capture、sampling constraint Colab runtime 和 notebook 入口。它们可作为本阶段 preflight 的实现基础。

### 2.2 当前阶段缺口

当前仍需要将 preflight 结果整理为独立 checker 或明确的 Colab 步骤, 并将 sampler signature、time grid、trajectory capture success rate 与 GPU memory budget 形成可审计记录。

### 2.3 当前阶段使用边界

该阶段只证明真实 Flow 模型接口可用, 不证明 SSTW 检测已经达到 `TPR@FPR=0.01`。若本阶段只获得 latent displacement proxy 而非真实 velocity field, 论文叙事必须使用 proxy evidence 的受限表述。

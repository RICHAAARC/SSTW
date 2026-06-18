# flow_model_adapter_preflight 分阶段构建流程

本文档记录 `flow_model_adapter_preflight` 阶段的构建流程与当前完成情况。本文档只描述工程、协议、records 和 artifact 状态, 不直接支撑论文最终 claim。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段在进入真实 GPU 生成实验前, 确认目标 Flow Matching / Rectified Flow 视频生成模型是否能够暴露、记录和复现 SSTW 所需的 trajectory proxy。该阶段的主要风险不是检测分数是否足够高, 而是模型接口是否能提供后续方法机制所需的可审计轨迹证据。

### 1.2 主线模型

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

轻量模型只能作为接口预验证或 fallback probe, 不能单独支撑完整 Flow Matching trajectory watermark claim。

### 1.3 必须检查

```text
model_load_success
single_prompt_generation_success
callback_latent_capture_success
velocity_proxy_available
latent_displacement_available
sampler_signature_available
time_grid_available
gpu_memory_budget_status
colab_runtime_budget_status
```

## 2. 当前阶段完成情况

### 2.1 当前阶段判定

`flow_model_adapter_preflight` 当前判定为:

```text
structure_ready / mechanism_ready / protocol_ready / artifact_ready
```

该判定基于真实 Colab / L4 / Wan2.1 运行链路已经证明以下接口能力可用:

```text
Wan2.1 pipeline 可以加载
callback_on_step_end 可以捕获 latent
trajectory time grid 可以记录
sampler signature 可以记录
latent displacement / flow velocity proxy 可以记录
Google Drive package 可以落盘并复核
```

### 2.2 证据边界

该阶段只证明真实 Flow Matching 视频生成接口可承接 SSTW trajectory records。它不直接证明最终 watermark detection 指标, 也不直接支持 `TPR@FPR=0.01` 或 full experiment claim。

后续阶段应继续使用 governed records 中的:

```text
sampler_signature_id
sampler_signature_sha256
trajectory_source_level
flow_velocity_proxy_available
flow_velocity_proxy_source
trajectory_time_grid_id
```

作为 replay、sampler mismatch 与 trajectory evidence 的输入证据。


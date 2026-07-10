# real_video_latent_transfer_check 分阶段构建流程

> 历史说明: 本文记录已删除的代理阶段, 不代表当前可执行架构。正式实现以 `main/methods/state_space_watermark/`、`evaluation/`、`experiments/generative_video_model_probe/` 和 `workflows/` 为准。


本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段验证 synthetic 阶段成立的密钥条件状态空间推断, 在真实视频 VAE encode-decode-reencode 链路中是否仍然有效并保持低误报。

### 1.2 输入

```text
configs/protocol/real_video_latent_transfer.json
configs/backends/video_vae_backend.json
configs/methods/method_variants_real_video_transfer.json
main/vae/
main/video/
experiments/real_video_latent_transfer/
```

### 1.3 构建任务

1. 从视频帧进入 VAE encode。
2. 在 latent 或 latent trace 层构造水印相关状态证据。
3. 经 VAE decode 回到视频帧。
4. 施加空间、时间和压缩攻击。
5. 再次 VAE re-encode 并执行状态空间检测。
6. 使用 protocol 阶段固定的 calibration threshold, 不允许 test-time threshold update。

### 1.4 必须 baseline

```text
endpoint_only_latent_score
frame_prc_baseline
explicit_temporal_alignment_baseline
generic_ssm_baseline
key_conditioned_state_space_method
```

### 1.5 必须审计

```text
reconstruction_quality_audit
motion_consistency_audit
semantic_consistency_audit
fixed_low_fpr_threshold_reuse
negative_tail_audit
```

### 1.6 通过标准

1. VAE 重建误差不破坏低 FPR 校准。
2. attacked positive 中 state-space evidence 仍有有效增益。
3. endpoint-only 方法不足以解释 SSTW 的整体提升。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有真实视频 latent transfer 相关模块:

```text
experiments/real_video_latent_transfer/runner.py
experiments/real_video_latent_transfer/mechanism_audit.py
experiments/real_video_latent_transfer/artifact_builder.py
experiments/real_video_latent_transfer/table_builder.py
experiments/real_video_latent_transfer/package_outputs.py
main/vae/vae_backend.py
main/vae/vae_io.py
main/vae/vae_reconstruction_audit.py
main/video/video_io.py
```

### 2.2 当前阶段使用边界

该阶段证明 VAE 链路可迁移性, 但仍不是 Flow Matching 采样动力学本身。若该阶段成立, 它为真实视频链路低误报提供证据; 若该阶段失败, 后续 Flow Matching 主线必须重新评估 endpoint consistency 与 VAE robustness。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 部分完成 |
| 主要差距项 | 真实 VAE 大规模 transfer 与低误报验证仍不足。 |
| 下一步构建方向 | pilot PASS 后执行 real-video transfer validation, 检查 VAE 重建对 endpoint/path evidence 的影响。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 部分完成
gap_item: 真实 VAE 大规模 transfer 与低误报验证仍不足。
next_action: pilot PASS 后执行 real-video transfer validation, 检查 VAE 重建对 endpoint/path evidence 的影响。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

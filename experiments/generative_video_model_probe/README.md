# Generative Video Model Probe

> 完整论文机制的实现顺序、三层 Claim 证据链和 fail-closed 规则见
> `docs/builds/complete_paper_mechanism_implementation.md`。三个 paper profile 使用同一机制,
> `probe_paper` 也必须在 FPR=0.1 下闭合 Claim-1、Claim-2 和不降级的 Claim-3。

本目录保存 generative_video_model_probe 生成式视频模型探测的可审计运行入口。当前无 GPU 时只生成 blocked decision, 不生成正向机制结论。

## Minimal trajectory replay smoke

只执行已有4-source包的标准攻击与真实 VAE/replay 诊断:

```bash
python -m experiments.generative_video_model_probe.trajectory_replay_smoke \
  --package-path outputs/<method_mechanism_validation_package>.zip \
  --run-root outputs/trajectory_replay_smoke \
  --config-path configs/protocol/sstw_minimal_trajectory_paper.json
```

该入口不生成新视频、不运行 external baseline、不使用 test split，也不连接其他项目。
缺少锁定 GPU 运行时或 owner key 时会生成环境阻断型 `NO_GO` 报告，禁止代理 replay。

## 生成模型分工

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 是三层主张与固定 FPR 主表模型。
- `Lightricks/LTX-Video` 是参数规模较小的 Flow Matching 跨模型泛化模型。
- LTX 使用三维 packed token latent, 但在进入 SSTW 算法原语前必须通过可逆
  layout 转换为 `[B, C, T, H, W]`; endpoint、path 和 replay 均不得使用代理分数。
- 跨模型子集按 calibration/test 分层抽样, 只支撑模型泛化结论, 不替代当前
  profile 在 Wan 主模型上的固定 FPR 闭合结论。

正式运行会生成:

```text
artifacts/cross_model_generalization_decision.json
tables/cross_model_generalization_table.csv
```

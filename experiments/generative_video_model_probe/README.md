# Generative Video Model Probe

> 完整论文机制的实现顺序、三层 Claim 证据链和 fail-closed 规则见
> `docs/builds/complete_paper_mechanism_implementation.md`。三个 paper profile 使用同一机制,
> `probe_paper` 也必须在 FPR=0.1 下闭合 Claim-1、Claim-2 和不降级的 Claim-3。

本目录保存 generative_video_model_probe 生成式视频模型探测的可审计运行入口。当前无 GPU 时只生成 blocked decision, 不生成正向机制结论。

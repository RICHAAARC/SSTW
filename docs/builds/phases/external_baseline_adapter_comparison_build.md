# external_baseline adapter comparison build

## 构建目标

本阶段把外部对比 baseline 从“状态记录”推进到“adapter 产出比较结果”的工程闭环。设计参考 `D:\Code\SLM-WM` 的外部 baseline 适配方式: 官方方法、项目 adapter、命令调度、observation records 和 comparison table 分层管理。

## 已完成内容

```text
external_baseline/README.md
external_baseline/source_registry.json
external_baseline/registry.py
external_baseline/runtime_trace_io.py
external_baseline/primary/explicit_dtw_temporal_alignment/adapter/run_sstw_eval.py
external_baseline/primary/explicit_frame_matching_temporal_registration/adapter/run_sstw_eval.py
experiments/generative_video_model_probe/external_baseline_runner.py
```

## 当前可产出的 governed artifacts

```text
records/external_baseline_records.jsonl
tables/external_baseline_status_table.csv
artifacts/external_baseline_status_decision.json
reports/external_baseline_status_report.md

records/external_baseline_score_records.jsonl
tables/external_baseline_comparison_table.csv
artifacts/external_baseline_comparison_decision.json
reports/external_baseline_comparison_report.md
```

## 关键边界

1. 显式 DTW 与 frame matching adapter 当前属于同步 control proxy, 不是现代视频水印 baseline。
2. adapter score 只使用 callback trajectory records 与 runtime video metadata, 不使用 `S_final` 或最终判定分数做污染过滤或 baseline 打分。
3. 现代视频水印 baseline 在官方实现和协议映射未接入前, 只能形成 governed unsupported comparison row。
4. 所有 comparison 表必须从 `external_baseline_score_records.jsonl` 重建, 不能手工写结论。

## 当前阶段状态

```text
external_baseline_adapter_boundary: implemented
explicit_dtw_temporal_alignment_adapter: implemented_proxy_control
explicit_frame_matching_temporal_registration_adapter: implemented_proxy_control
modern_video_watermark_baseline_adapter: not_integrated
baseline_comparison_output_chain: implemented
claim_support_status: external_baseline_proxy_comparison_not_claim_supporting
```

## 后续工作

```text
1. 为 VideoShield / SigMark / SPDMark / VideoSeal 等现代 baseline 接入官方或方法忠实 adapter。
2. 将官方 baseline 输出映射到同一 prompt、attack、threshold、quality guard 协议。
3. 在 validation-scale 和 full-paper 阶段替换当前 proxy comparison 为正式 baseline results。
4. 仅当现代 baseline 产生 governed measured records 后, 才允许进入论文主表 claim 审计。
```

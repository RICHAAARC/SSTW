# external_baseline adapter comparison build

## 构建目标

本阶段把外部对比 baseline 从“状态记录”推进到“adapter 产出比较结果”的工程闭环。本项目采用独立的 adapter-source-observation-comparison 分层方式: 官方方法或外部方法本体与 SSTW 的受治理输出分离, adapter 负责把外部方法输出映射到统一 records, 下游表格和报告只能从 records 重建。

## external_baseline 接入方式

### 1. baseline source registry

`external_baseline/source_registry.json` 是外部 baseline 的身份登记表。它记录:

```text
baseline_id
baseline_name
baseline_family
comparison_group
source_dir
source_status
official_repository_url
official_repository_commit
adapter_path
adapter_status
result_status
paper_claim_support
```

该文件只说明来源、接入状态和 claim 边界。它不能单独证明 baseline 已经完成实验对比。

### 2. adapter implementation

每个可运行 baseline 的适配代码放在:

```text
external_baseline/primary/<baseline_id>/adapter/run_sstw_eval.py
```

adapter 必须提供:

```text
adapter_status()
build_score_records(run_root, baseline_record)
```

`adapter_status()` 返回可运行状态、输入兼容状态、输出记录状态和 claim 边界。`build_score_records()` 只读取 `run_root` 中已落盘的 governed records, 并返回统一字段的 comparison score records。

### 3. experiment scheduler

`experiments/generative_video_model_probe/external_baseline_runner.py` 是本项目调度层。它负责:

```text
读取 configs/external_baselines/external_baselines.json
写出 external_baseline_records.jsonl
调用 external_baseline/ 下已注册 adapter
写出 external_baseline_score_records.jsonl
从 score records 重建 external_baseline_comparison_table.csv
写出 decision artifact 和 report
```

### 4. gate and package integration

validation-scale gate、artifact rebuild dry-run 和 Google Drive packager 都只读取已落盘文件:

```text
records/external_baseline_records.jsonl
records/external_baseline_score_records.jsonl
tables/external_baseline_status_table.csv
tables/external_baseline_comparison_table.csv
artifacts/external_baseline_status_decision.json
artifacts/external_baseline_comparison_decision.json
reports/external_baseline_status_report.md
reports/external_baseline_comparison_report.md
```

这保证了 Notebook 只是入口, baseline comparison 不是 Notebook cell 中的临时逻辑。

## 已完成内容

```text
external_baseline/README.md
external_baseline/source_registry.json
external_baseline/registry.py
external_baseline/runtime_trace_io.py
external_baseline/modern_command_adapter.py
external_baseline/primary/explicit_dtw_temporal_alignment/adapter/run_sstw_eval.py
external_baseline/primary/explicit_frame_matching_temporal_registration/adapter/run_sstw_eval.py
external_baseline/primary/videoshield/adapter/run_sstw_eval.py
external_baseline/primary/sigmark/adapter/run_sstw_eval.py
external_baseline/primary/spdmark/adapter/run_sstw_eval.py
external_baseline/primary/videomark_or_vidsig/adapter/run_sstw_eval.py
external_baseline/primary/videoseal/adapter/run_sstw_eval.py
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
3. 现代视频水印 baseline 必须通过正式 command adapter 产出 `metric_status = measured_formal`; 官方命令未配置时只能形成 governed unsupported comparison row。
4. 所有 comparison 表必须从 `external_baseline_score_records.jsonl` 重建, 不能手工写结论。
5. 显式同步 control 的 `external_baseline_result_used_for_claim` 必须保持 `false`; 现代 baseline 只有 measured_formal records 才能进入 pilot_paper / full_paper 比较审计。

## 当前阶段状态

```text
external_baseline_adapter_boundary: implemented
explicit_dtw_temporal_alignment_adapter: implemented_proxy_control
explicit_frame_matching_temporal_registration_adapter: implemented_proxy_control
modern_video_watermark_baseline_adapter: formal_command_adapter_integrated_requires_official_command
required_modern_video_watermark_baselines: videoshield, sigmark, spdmark, videomark_or_vidsig, videoseal
baseline_comparison_output_chain: implemented
claim_support_status: formal_results_required_for_pilot_paper
```

## 后续工作

```text
1. 在 Colab 或本地环境中安装每个现代 baseline 的官方源码、权重和命令入口。
2. 配置 `SSTW_VIDEOSHIELD_EVAL_COMMAND`、`SSTW_SIGMARK_EVAL_COMMAND`、`SSTW_SPDMARK_EVAL_COMMAND`、`SSTW_VIDEOMARK_OR_VIDSIG_EVAL_COMMAND` 和 `SSTW_VIDEOSEAL_EVAL_COMMAND`。
3. 运行 external_baseline comparison, 使 5 个现代 baseline 均产出 `measured_formal` records。
4. 仅当全部现代 baseline 都产生 governed measured_formal records 后, 才允许进入 pilot_paper gate。
```


### 5. 现代 baseline 正式命令契约

现代视频水印 baseline 的 adapter 不在本仓库中重写第三方论文方法本体。每个 adapter 调用用户在 Colab / 本地配置的官方命令, 并要求命令输出 JSON:

```text
external_baseline_score 或 score
external_baseline_detected 或 detected
external_baseline_bit_accuracy 或 bit_accuracy
external_baseline_threshold 或 threshold
```

命令模板可以使用以下占位字段:

```text
{source_video_path}
{attacked_video_path}
{attack_name}
{output_json_path}
{prompt_id}
{seed_id}
{trajectory_trace_id}
{run_root}
```

若命令未配置或输出缺失, adapter 必须写出 unsupported record, 并使 `pilot_paper` gate 失败。该设计保证 `pilot_paper` 与 `full_paper` 的差异只保留为样本规模和 FPR 评价级别, 而不是 baseline 协议缺口。

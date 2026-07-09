# external_baseline adapter comparison build

## 构建目标

本阶段把外部对比 baseline 从“状态记录”推进到“项目内自包含 comparison records”的工程闭环。本项目采用独立的 source-intake、clone / build / run、adapter、record 和 comparison 分层方式: 官方方法或外部方法本体与 SSTW 的受治理输出分离, adapter 只负责把项目内运行得到的外部方法输出映射到统一 records, 下游表格和报告只能从 records 重建。

本阶段不再接受“外部补交结果”。正式 modern external baseline 必须由项目内流程完成:

```text
project_clone -> project_build -> project_run -> project_adapt -> project_record
```

无法完成上述链路的 baseline 只能写出 `non_runnable_with_governed_reason` 或 protocol gap, 该记录只能解释阻断原因, 不能替代 `measured_formal` 主表结果。

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
project_clone_status
project_build_status
project_run_status
project_adapt_status
project_record_status
paper_claim_support
```

该文件只说明来源、接入状态、项目内自包含执行状态和 claim 边界。它不能单独证明 baseline 已经完成实验对比。

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

`adapter_status()` 返回可运行状态、输入兼容状态、输出记录状态、项目内 clone / build / run / adapt / record 状态和 claim 边界。`build_score_records()` 只读取 `run_root` 中已落盘的 governed records 与项目内 baseline 输出, 并返回统一字段的 comparison score records。adapter 不得读取外部补交 JSON、论文表格数字、Notebook 临时变量或 SSTW proxy 分数作为 modern baseline 正式结果。

### 3. experiment scheduler

`experiments/generative_video_model_probe/external_baseline_runner.py` 是本项目调度层。它负责:

```text
读取 configs/external_baselines/external_baselines.json
写出 external_baseline_records.jsonl
调用 external_baseline/ 下已注册 adapter
写出 external_baseline_score_records.jsonl
从 score records 重建 external_baseline_comparison_table.csv
写出 comparison decision、self-containment decision 和 report
```

### 4. gate and package integration

validation_scale gate、artifact rebuild dry-run 和 Google Drive packager 都只读取已落盘文件:

```text
records/external_baseline_records.jsonl
records/external_baseline_score_records.jsonl
tables/external_baseline_status_table.csv
tables/external_baseline_comparison_table.csv
artifacts/external_baseline_status_decision.json
artifacts/external_baseline_comparison_decision.json
artifacts/external_baseline_self_containment_decision.json
reports/external_baseline_status_report.md
reports/external_baseline_comparison_report.md
reports/external_baseline_self_containment_report.md
```

这保证了 Notebook 只是入口, baseline comparison 不是 Notebook cell 中的临时逻辑。

## 已完成内容

```text
external_baseline/README.md
external_baseline/source_registry.json
external_baseline/registry.py
external_baseline/runtime_trace_io.py
external_baseline/modern_command_adapter.py
external_baseline/official_runtime_closure.py
configs/external_baselines/official_runtime_closure_requirements.json
configs/external_baselines/requirements/<baseline_id>.txt
external_baseline/primary/explicit_dtw_temporal_alignment/adapter/run_sstw_eval.py
external_baseline/primary/explicit_frame_matching_temporal_registration/adapter/run_sstw_eval.py
external_baseline/primary/videoshield/adapter/run_sstw_eval.py
external_baseline/primary/vidsig/adapter/run_sstw_eval.py
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
artifacts/external_baseline_official_runtime_closure_requirements.json
artifacts/external_baseline_comparison_decision.json
artifacts/external_baseline_self_containment_decision.json
reports/external_baseline_comparison_report.md
reports/external_baseline_self_containment_report.md
```

## 关键边界

1. 显式 DTW 与 frame matching adapter 当前属于同步 control proxy, 不是现代视频水印 baseline。
2. adapter score 只使用 callback trajectory records 与 runtime video metadata, 不使用 `S_final` 或最终判定分数做污染过滤或 baseline 打分。
3. 现代视频水印 baseline 必须通过项目内 clone / build / run / adapt / record 和正式 adapter 产出 `metric_status = measured_formal`; 官方资源或命令未配置时只能形成 governed unsupported / non-run comparison row。
4. 所有 comparison 表必须从 `external_baseline_score_records.jsonl` 重建, 不能手工写结论。
5. 显式同步 control 的 `external_baseline_result_used_for_claim` 必须保持 `false`; 现代 baseline 只有项目内自包含的 `measured_formal` records 才能进入 validation_scale / probe_paper / pilot_paper / full_paper 比较审计。

## 当前阶段状态

```text
external_baseline_adapter_boundary: implemented
explicit_dtw_temporal_alignment_adapter: implemented_proxy_control
explicit_frame_matching_temporal_registration_adapter: implemented_proxy_control
modern_video_watermark_baseline_adapter: self_contained_project_execution_required
baseline_comparison_output_chain: implemented
claim_support_status: measured_formal_self_contained_results_required_for_validation_scale_probe_paper_pilot_paper_and_full_paper
```

## 后续工作

```text
1. 在 Colab 或等价受治理运行环境中, 由本项目 clone / build 每个现代 baseline 的官方源码、权重和依赖。
2. 通过 `configs/external_baselines/official_runtime_closure_requirements.json` 和 `configs/external_baselines/requirements/<baseline_id>.txt` 明确每个 baseline 的真实运行要求; 5 个主实验 formal reference Notebook 默认会安装各自的 requirements 文件。
3. 运行 `external_baseline.official_runtime_closure`, 写出 `artifacts/external_baseline_official_runtime_closure_requirements.json`; 若 Google Drive 默认资源路径已存在, Notebook 应自动应用其中的 `environment_updates`。
4. 默认使用 repository bridge + official inner command; 不要求用户手写 5 个主实验 `SSTW_<BASELINE>_EVAL_COMMAND`。只有在官方仓库新增更合适 CLI 时, 才通过 `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND` 覆盖单个 baseline。
6. 在 5 个主实验 official bundle 全部完成后运行 `paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb`, 由该阶段恢复全部 official reference 阶段包后执行全量统一转写、self-containment 判定、公平校准和差值区间统计。随后 `paper_evidence_postprocess_colab.ipynb` 生成辅助证据, `paper_gate_and_package_colab.ipynb` 恢复 runtime、motion threshold、formal comparison scoring 和 paper evidence postprocess 阶段包并执行最终门禁与打包。
7. 写出 `artifacts/external_baseline_self_containment_decision.json`, 逐 baseline 记录 clone / build / run / adapt / record 状态。
8. 仅当全部现代 baseline 都产生项目内自包含 governed `measured_formal` records 后, 才允许 validation_scale gate 通过; validation_scale 通过后还必须生成 validation_scale_to_probe_paper_transition_decision 才能进入 probe_paper; probe_paper 通过后再生成 probe_paper_to_pilot_paper_transition_decision 才能进入 pilot_paper, full_paper 仍需 pilot_paper gate、pilot_paper_to_full_paper_transition_decision 与 full_paper_result_checker。
```


### 5. 现代 baseline 自包含执行契约

现代视频水印 baseline 的 adapter 不在本仓库中重写第三方论文方法本体。每个 adapter 只能调用由本项目 clone / build 后配置的官方命令或 repository bridge 命令, 并要求命令输出 JSON:

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

若命令未配置、项目内 clone / build / run / adapt / record 任一缺失, 或输出缺失, adapter 必须写出 unsupported / non-run record, 并使 `validation_scale`、`probe_paper`、`pilot_paper` 和 `full_paper` 相关检查失败。该设计保证 `pilot_paper` 与后续更大规模 paper 运行的差异由 protocol config 显式记录; baseline 自包含产出规则不得存在协议缺口。

### 6. Notebook role 边界

外部 baseline 不应再作为主方法 runtime Notebook 的长尾附属步骤来维护。当前推荐边界为:

```text
5 个 SSTW runtime 拆分 Notebook: 执行真实 GPU 生成、formal metrics、机制后处理、runtime attack 和 detection, 不执行 baseline command preflight 或 baseline comparison。
*_formal_reference_colab.ipynb: 读取同一 workflow_profile 的 run_root, 逐 baseline 执行 source intake、project clone / build / run / adapt 和 official bundle 生成; 不执行全量 measured_formal 转写。
formal_comparison_scoring_colab.ipynb: 读取同一 workflow_profile 的 run_root, 恢复 5 个主实验 official reference 阶段包后执行全量统一转写、self-containment 判定、公平校准和差值区间统计。paper_evidence_postprocess_colab.ipynb 先恢复 formal comparison scoring 阶段包并生成辅助证据; paper_gate_and_package_colab.ipynb 再恢复 runtime、motion threshold、formal comparison scoring 和 paper evidence postprocess 阶段包并执行最终 gate 打包。旧的通用 external baseline scoring Notebook 已删除, 防止与 paper gate 聚合职责重复。
```

profile、Drive 目录和 stage plan 由下述配置统一控制:

```text
configs/paper_workflow/generative_video_notebook_workflows.json
```

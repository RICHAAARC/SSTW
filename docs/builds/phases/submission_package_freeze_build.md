# submission_package_freeze 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段将 governed records 转换为论文可使用的 tables、figures、reports 和 manifests, 并执行 claim audit、readiness summary 和 release extraction 检查。

### 1.2 输入

```text
records/event_scores.jsonl
records/thresholds.jsonl
records/trajectory_traces.jsonl
experiments/submission_freeze_preparation/
scripts/package_results/submission_freeze_preparation_packager.py
docs/builds/sstw_project_construction_flow.md
docs/builds/sstw_method_mechanism_design.md
```

### 1.3 构建任务

1. 从 records 重建主表、baseline 表和 ablation 表。
2. 从 records 和 manifests 重建主图所需数据。
3. 生成 claim audit report。
4. 生成 readiness summary。
5. 生成 submission package manifest。
6. 执行 release extraction contract 检查。

### 1.4 必须产物

```text
tables/main_detection_table.csv
tables/baseline_comparison_table.csv
tables/ablation_table.csv
figures/roc_or_tpr_at_fpr_figure.json
figures/trajectory_evidence_figure.json
reports/claim_audit_report.json
reports/readiness_summary.json
manifests/submission_package_manifest.json
```

### 1.5 禁止事项

1. 不得手工改写正式表格数值。
2. 不得从 test split 反向更新 calibration threshold。
3. 不得用 placeholder 字段支撑 supported claims。
4. 不得把临时 Colab 输出直接当成论文 artifact。

### 1.6 通过标准

1. 主表、主图、报告和 claim audit 可由 records 与 manifests 自动重建。
2. `pytest -q` 和 harness 审计通过。
3. supported claims 全部绑定 governed artifacts。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有 submission freeze preparation 相关模块:

```text
experiments/submission_freeze_preparation/runner.py
experiments/submission_freeze_preparation/main_tables.py
experiments/submission_freeze_preparation/readiness_summary.py
scripts/package_results/submission_freeze_preparation_packager.py
```

### 2.2 当前阶段使用边界

该阶段只能组织和重建 governed artifacts, 不能手工创造论文结果。若上游阶段缺少真实 GPU、真实模型 records、pilot gate 记录或 negative family 记录, 本阶段只能报告 evidence gap, 不能补写 supported claims。

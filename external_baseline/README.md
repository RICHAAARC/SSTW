# 外部 baseline 适配边界

本目录用于保存 SSTW 项目的外部 baseline 适配器。该目录的核心职责是把外部方法、显式同步 control 或后续接入的现代视频水印方法, 转换为本项目统一的 governed records、tables、artifacts 和 reports。

## 接入方式

external baseline 采用“六层接入”方式:

1. `source_registry.json` 登记 baseline 身份、方法家族、源码状态、adapter 路径、结果状态和 claim 边界。
2. `external_baseline_intake_manifest.json`、`external_baseline_source_inspection.json`、`external_baseline_clone_results.json` 和 `plans/external_baseline_table_plan.json` 记录 source intake、源码检查、clone 计划和主表角色。
3. `primary/<baseline_id>/source/` 保存第三方官方源码或用户放置的等价入口, 该目录由 `.gitignore` 排除, 不进入主方法层。
4. `primary/<baseline_id>/adapter/` 保存本仓库维护的 adapter。adapter 必须暴露 `adapter_status()` 和 `build_score_records(run_root, baseline_record)` 两类接口。
5. `experiments/generative_video_model_probe/external_baseline_runner.py` 负责在某次 `run_root` 中调度 adapter, 并把状态记录、比较结果和 `external_baseline_execution_manifest.json` 落盘。
6. validation gate、artifact rebuild dry-run 和 packager 只读取已落盘 artifacts, 不直接调用第三方源码或 Notebook 临时变量。

## 接入规则

1. 若后续下载第三方官方源码, 应放在 `primary/<baseline_id>/source/` 或 `supplemental/<baseline_id>/source/`, 并由 `.gitignore` 排除。
2. adapter 只能读取 `run_root` 中已经受治理的输入, 例如 `runtime_detection_records.jsonl` 和 `trajectory_trace.jsonl`。
3. adapter 只能写出受治理 observation 或 score records, 不能手工拼接论文结论。
4. adapter 不得使用 `S_final` 或最终判定分数进行污染过滤、样本剔除或 baseline 打分。
5. 当前显式 DTW 与 frame matching adapter 属于工程级同步 control, 用于验证 baseline 对比链路闭合, 不能支持正向论文 claim。
6. 现代视频水印 baseline 必须通过正式 command adapter 调用官方实现或用户配置的等价命令。未配置官方命令时, 只能写出 governed non-run record 和 comparison unsupported row。
7. source intake 清单只证明 baseline 来源、adapter 和命令边界已被登记, 不等价于 baseline 已完成正式运行。
8. `external_baseline_execution_manifest.json` 必须记录本次 run_root 的 measured / formal 记录数量、evidence path 状态和 source intake manifest 路径。没有 evidence path 的 formal rows 只能作为工程接入证据, 不能自动升级为论文主表 claim。

## 与 `main/` 的职责区别

- `main/` 保存 SSTW 方法、协议字段和可复用算法。
- `external_baseline/` 保存外部方法到 SSTW 统一协议的适配入口。
- `experiments/` 负责在某次 `run_root` 中调度 adapter 并落盘比较结果。


## 现代 baseline command adapter

`pilot_paper` 与 `full_paper` 的差异只允许是样本规模和 FPR 评价级别, 因此现代 baseline 不能只接入一个, 也不能由显式同步 control 替代。当前必须覆盖:

```text
videoshield
sigmark
spdmark
videomark_or_vidsig
videoseal
```

每个现代 baseline 的 adapter 会优先读取以下环境变量命令:

```text
SSTW_VIDEOSHIELD_EVAL_COMMAND
SSTW_SIGMARK_EVAL_COMMAND
SSTW_SPDMARK_EVAL_COMMAND
SSTW_VIDEOMARK_OR_VIDSIG_EVAL_COMMAND
SSTW_VIDEOSEAL_EVAL_COMMAND
```

命令模板可使用 `{source_video_path}`、`{attacked_video_path}`、`{attack_name}`、`{output_json_path}`、`{prompt_id}`、`{seed_id}`、`{trajectory_trace_id}` 和 `{run_root}`。命令必须写出 JSON score。adapter 只负责把该 JSON 映射为 `external_baseline_score_records.jsonl`, 不在本仓库中重写第三方论文方法本体。

## Source intake 命令

可使用下述命令生成或刷新 source intake 治理文件:

```bash
python scripts/build_external_baseline_source_intake.py --output-root external_baseline --repo-root .
```

默认模式只写出计划和缺口, 不访问网络。若已经确认第三方 URL 可 clone 且需要在 Colab 冷启动中拉取源码, 才使用:

```bash
python scripts/build_external_baseline_source_intake.py --output-root external_baseline --repo-root . --execute-clone
```

该命令属于工程准备层, 不能替代真实 baseline 运行。真实比较仍必须由 adapter 生成 `external_baseline_score_records.jsonl`、`external_baseline_comparison_decision.json` 和 `external_baseline_execution_manifest.json`。

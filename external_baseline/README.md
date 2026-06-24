# 外部 baseline 适配边界

本目录用于保存 SSTW 项目的外部 baseline 适配器。该目录的核心职责是把外部方法、显式同步 control 或后续接入的现代视频水印方法, 转换为本项目统一的 governed records、tables、artifacts 和 reports。

## 接入方式

external baseline 采用“四层接入”方式:

1. `source_registry.json` 登记 baseline 身份、方法家族、源码状态、adapter 路径、结果状态和 claim 边界。
2. `primary/<baseline_id>/adapter/` 保存本仓库维护的 adapter。adapter 必须暴露 `adapter_status()` 和 `build_score_records(run_root, baseline_record)` 两类接口。
3. `experiments/generative_video_model_probe/external_baseline_runner.py` 负责在某次 `run_root` 中调度 adapter, 并把状态记录与比较结果落盘。
4. validation gate、artifact rebuild dry-run 和 packager 只读取已落盘 artifacts, 不直接调用第三方源码或 Notebook 临时变量。

## 接入规则

1. 若后续下载第三方官方源码, 应放在 `primary/<baseline_id>/source/` 或 `supplemental/<baseline_id>/source/`, 并由 `.gitignore` 排除。
2. adapter 只能读取 `run_root` 中已经受治理的输入, 例如 `runtime_detection_records.jsonl` 和 `trajectory_trace.jsonl`。
3. adapter 只能写出受治理 observation 或 score records, 不能手工拼接论文结论。
4. adapter 不得使用 `S_final` 或最终判定分数进行污染过滤、样本剔除或 baseline 打分。
5. 当前显式 DTW 与 frame matching adapter 属于工程级同步 control, 用于验证 baseline 对比链路闭合, 不能支持正向论文 claim。
6. 现代视频水印 baseline 必须通过正式 command adapter 调用官方实现或用户配置的等价命令。未配置官方命令时, 只能写出 governed non-run record 和 comparison unsupported row。

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

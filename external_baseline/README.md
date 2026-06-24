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
videomark
vidsig
videoseal
```

每个现代 baseline 的 adapter 会优先读取以下环境变量命令:

```text
SSTW_VIDEOSHIELD_EVAL_COMMAND
SSTW_SIGMARK_EVAL_COMMAND
SSTW_SPDMARK_EVAL_COMMAND
SSTW_VIDEOMARK_EVAL_COMMAND
SSTW_VIDSIG_EVAL_COMMAND
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

## Colab 冷启动运行要求

真实现代 baseline 运行统一放在 Colab 环境中完成。本地仓库只负责 adapter、schema、manifest、gate 和轻量测试, 不负责执行第三方视频水印模型的重型推理。

Colab 中必须显式配置以下三类内容:

1. 现代 baseline command:

   ```text
   SSTW_VIDEOSHIELD_EVAL_COMMAND
   SSTW_SIGMARK_EVAL_COMMAND
   SSTW_SPDMARK_EVAL_COMMAND
   SSTW_VIDEOMARK_EVAL_COMMAND
   SSTW_VIDSIG_EVAL_COMMAND
   SSTW_VIDEOSEAL_EVAL_COMMAND
   ```

2. 可选的官方 source clone:

   ```text
   RUN_EXTERNAL_BASELINE_SOURCE_CLONE = True 或 False
   ```

   该开关只会对可 git clone 的 source URL 生效。不能 git clone 的论文页面、PDF 或 HTML source 仍需要用户在 Colab 中提供官方命令或手工放置 source snapshot。

3. 额外外部运行证据路径:

   ```text
   EXTERNAL_BASELINE_EVIDENCE_PATHS = [
       "/content/drive/MyDrive/SSTW/...",
   ]
   ```

这些 evidence path 应指向官方 baseline 运行日志、配置、输出 JSON、依赖快照或校验文件。adapter 会把路径写入 `artifacts/external_baseline_execution_manifest.json`。

同时, 现代 baseline command adapter 会自动把每条官方命令的输出持久化到:

```text
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/
```

该目录包含:

```text
official_output.json
official_stdout.txt
official_stderr.txt
official_command_manifest.json
```

这些文件会被自动纳入 `external_baseline_execution_manifest.json` 的 `evidence_paths`。因此 Colab 断开后, 用户仍可以从 Google Drive package 中审计每条 `measured_formal` baseline score 的官方输出来源。若没有自动持久化证据且没有额外 evidence path, `measured_formal` rows 只能证明 command adapter 写出了受治理 records, 不能单独支撑论文主表 claim。

当前 `validation_scale` 已经是进入 paper 级运行前的最后一道完整门禁。因此在 `SSTW_WORKFLOW_PROFILE=validation_scale` 或 `SSTW_WORKFLOW_PROFILE=pilot_paper` 时, Colab notebook 会在 6 个现代 baseline command 缺失时提前阻断, 避免先消耗 GPU 生成视频后才发现 baseline 主表无法闭合。旧的 `fpr01_pilot` 仅作为兼容别名映射到 `pilot_paper`。

Notebook 的现代 baseline command preflight 逻辑集中在:

```text
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py
```

该 helper 会从 protocol config 读取 `required_modern_external_baseline_adapter_names`, 生成需要配置的环境变量列表, 并在阻断前写出:

```text
artifacts/external_baseline_colab_preflight_decision.json
```

该 artifact 只用于冷启动执行审计, 不表示 baseline 已经运行, 也不能单独支撑外部 baseline 对比 claim。

## 独立 baseline Notebook

现代 baseline 正式运行应优先使用:

```text
paper_workflow/colab_utils/external_baseline_formal_scoring_colab.ipynb
```

该 Notebook 的职责是读取同一 `workflow_profile` 的 `drive_run_root`, 执行 source intake、
command adapter 和 comparison records 生成。它不重新生成 Wan2.1 视频, 也不执行最终
paper gate。这样可以把三类失败原因分离:

1. `generative_video_runtime_colab.ipynb`: 主方法生成、attack 和 detection 是否成功。
2. `external_baseline_formal_scoring_colab.ipynb`: 现代视频水印 baseline 官方命令和 evidence 是否成功。
3. `paper_gate_and_package_colab.ipynb`: validation-scale 或 pilot-paper gate 是否满足论文协议。

Notebook 的 profile、Drive 目录和 stage plan 均由以下配置控制:

```text
configs/paper_workflow/generative_video_notebook_workflows.json
```

因此, 切换 `validation_scale` 与 `pilot_paper` 时, 不应复制或改写 Notebook 中的 run_root /
package_dir 字符串, 而应设置:

```text
SSTW_WORKFLOW_PROFILE=validation_scale
或
SSTW_WORKFLOW_PROFILE=pilot_paper
```

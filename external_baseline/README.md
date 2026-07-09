# 外部 baseline 适配边界

本目录用于保存 SSTW 项目的外部 baseline 适配器。该目录的核心职责是把外部方法、显式同步 control 或后续接入的现代视频水印方法, 转换为本项目统一的 governed records、tables、artifacts 和 reports。

## 接入方式

external baseline 采用分层接入方式:

1. `source_registry.json` 登记 baseline 身份、方法家族、源码状态、adapter 路径、结果状态和 claim 边界。
2. `external_baseline_intake_manifest.json`、`external_baseline_source_inspection.json`、`external_baseline_clone_results.json` 和 `plans/external_baseline_table_plan.json` 记录 source intake、源码检查、clone 计划和主表角色。
3. `primary/<baseline_id>/source/` 保存第三方官方源码或用户放置的等价入口, 该目录由 `.gitignore` 排除, 不进入主方法层。
4. `primary/<baseline_id>/adapter/` 保存本仓库维护的 adapter。adapter 必须暴露 `adapter_status()` 和 `build_score_records(run_root, baseline_record)` 两类接口。
5. `official_eval_adapters/<baseline_id>.py` 保存 repository 维护的 fail-closed 官方命令 wrapper, 用于把官方源码、官方 API、官方 checkpoint 或本项目生成的官方结果缓存转换成 SSTW 统一 JSON。
6. `official_resource_bootstrap.py` 在 Colab 冷启动中自动补齐可公开获得的官方资源, 并对无法自动补齐的 baseline 写出 `manual_official_resource_required` 或 `non_runnable_with_governed_reason`。
7. `official_bundle_generator.py` 为可由官方 API 自动支持的 baseline 生成 repository-owned official bundle cache, 同时为高显存或缺训练权重 baseline 写出不能自动生成的原因。
8. `official_runtime_closure.py` 根据 `configs/external_baselines/official_runtime_closure_requirements.json` 检查真实运行所需的 source、requirements、runtime videos、官方资源和 official bundle cache, 并写出可由 Notebook 读取的环境变量修复建议。
9. `official_result_bundle.py` 检查本项目 workflow 生成的官方结果缓存或当前会话可运行资源是否覆盖当前 runtime comparison unit。
10. `experiments/generative_video_model_probe/external_baseline_runner.py` 负责在某次 `run_root` 中调度 adapter, 并把状态记录、比较结果和 `external_baseline_execution_manifest.json` 落盘。
11. validation gate、artifact rebuild dry-run 和 packager 只读取已落盘 artifacts, 不直接调用第三方源码或 Notebook 临时变量。

## 接入规则

1. 若后续下载第三方官方源码, 应放在 `primary/<baseline_id>/source/` 或 `supplemental/<baseline_id>/source/`, 并由 `.gitignore` 排除。
2. adapter 只能读取 `run_root` 中已经受治理的输入, 例如 `runtime_detection_records.jsonl` 和 `trajectory_trace.jsonl`。
3. adapter 只能写出受治理 observation 或 score records, 不能手工拼接论文结论。
4. adapter 不得使用 `S_final` 或最终判定分数进行污染过滤、样本剔除或 baseline 打分。
5. 当前显式 DTW 与 frame matching adapter 属于工程级同步 control, 用于验证 baseline 对比链路闭合, 不能支持正向论文 claim。
6. 现代视频水印 baseline 必须通过正式 command adapter 调用官方实现、repository official adapter 或用户配置的等价命令, 并由本项目完成 clone / build / run / adapt / record。未配置官方命令或缺少官方权重、key、message、maintained info 时, 只能写出 governed non-run record 和 comparison unsupported row。
7. 对无法在当前 Colab 会话即时复跑的高显存、训练型或 maintained-info baseline, 不接受外部补交结果。若使用 official result bundle 路径, 它只能作为本项目 workflow 生成的内部缓存, 且 bundle JSON 必须可追溯到本项目调用第三方官方代码或官方原生命令的记录, 不能由 SSTW `S_final`、最终判定分数或视频相似度 proxy 派生。
8. source intake 清单只证明 baseline 来源、adapter 和命令边界已被登记, 不等价于 baseline 已完成正式运行。
9. `external_baseline_execution_manifest.json` 必须记录本次 run_root 的 measured / formal 记录数量、evidence path 状态和 source intake manifest 路径。没有 evidence path 的 formal rows 只能作为工程接入证据, 不能自动升级为论文主表 claim。

## 与 `main/` 的职责区别

- `main/` 保存 SSTW 方法、协议字段和可复用算法。
- `external_baseline/` 保存外部方法到 SSTW 统一协议的适配入口。
- `experiments/` 负责在某次 `run_root` 中调度 adapter 并落盘比较结果。


## 现代 baseline command adapter

`pilot_paper` 与 `full_paper` 的差异只允许是样本规模和 FPR 评价级别, 因此现代 baseline 不能只接入一个, 也不能由显式同步 control 替代。当前保留并必须覆盖:

```text
videoshield
vidsig
videoseal
revmark
wam_frame
```

每个现代 baseline 的 adapter 会优先读取以下环境变量命令:

```text
SSTW_VIDEOSHIELD_EVAL_COMMAND
SSTW_VIDSIG_EVAL_COMMAND
SSTW_VIDEOSEAL_EVAL_COMMAND
```

命令模板可使用 `{source_video_path}`、`{attacked_video_path}`、`{attack_name}`、`{output_json_path}`、`{prompt_id}`、`{seed_id}`、`{trajectory_trace_id}` 和 `{run_root}`。命令必须写出 JSON score。adapter 只负责把该 JSON 映射为 `external_baseline_score_records.jsonl`, 不在本仓库中重写第三方论文方法本体。

若使用默认 bridge 模式, Notebook 可以通过 `SSTW_USE_REPOSITORY_OFFICIAL_BASELINE_ADAPTERS=true`
把内部命令自动指向:

```text
external_baseline/official_eval_adapters/videoshield.py
external_baseline/official_eval_adapters/vidsig.py
external_baseline/official_eval_adapters/videoseal.py
external_baseline/official_eval_adapters/revmark.py
external_baseline/official_eval_adapters/wam_frame.py
```

这些 wrapper 是 fail-closed 入口。它们只允许调用官方源码/API或读取由本项目 workflow
生成的 official bundle cache。若缺少第三方 checkpoint、extractor、message、
maintained info 或项目内 official bundle cache, wrapper 会失败, 不会输出
视频相似度、SSTW 分数或其他 proxy 分数。用户也可以通过
`SSTW_<BASELINE>_NATIVE_EVAL_COMMAND` 覆盖单个 baseline 的官方原生命令。

VidSig 是特殊边界: 它属于生成过程中嵌入签名的方法, 因此默认 adapter 不允许把
SSTW / Wan 生成视频直接送入 VidSig detector 后输出正式 baseline 分数。VidSig 正式路径
必须由 `external_baseline/vidsig_official_runtime.py` 先运行官方 `generate_ms.py`
生成 VidSig 自己的 clean / watermarked videos, 再施加项目 runtime attack, 最后调用
官方 `attack.py` 写出 project-owned official bundle。该 bundle 随后由统一 runner
转写为 `metric_status: measured_formal` records。

## repository-owned 官方结果缓存

部分现代视频水印 baseline 绑定特定生成模型、训练得到的 extractor、message decoder、
PRC key 或 maintained info。为了避免同一 run_root 在 Colab 冷启动后重复执行重型官方流程, 正式流程支持
读取 Google Drive 中由本项目 workflow 生成的官方结果缓存:

```text
SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT
SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS
```

单条结果只能由 formal reference Notebook、repository official adapter 或 official
bundle generator 写入, 命名位置可采用以下任一形式:

```text
<bundle_root>/<baseline_id>/records/<prompt_id>__<seed_id>__<attack_name>.json
<bundle_root>/<baseline_id>/records/<trajectory_trace_id>__<attack_name>.json
<bundle_root>/<baseline_id>/<prompt_id>/<seed_id>/<attack_name>.json
<bundle_root>/<baseline_id>/<trajectory_trace_id>/<attack_name>.json
```

每个 JSON 必须包含可审计 score 字段, 例如 `external_baseline_score`、`score`、
`bit_accuracy`、`confidence` 或 `detected`。同时, `official_result_provenance`
必须为 `repository_generated_from_third_party_official_code`, 且
`official_execution_manifest_path` 必须指向已落盘 manifest。旧的
`third_party_official_code`、手写 JSON、NPZ 分数文件或 SSTW proxy 分数都会被视为
不满足正式 `measured_formal` 输入要求。

新生成的 baseline official reference bundle 整包 manifest 必须使用统一状态:

```text
execution_status = "official_reference_bundle_complete"
```

旧历史 manifest 中的 `executed`、`completed`、`generated`、`ready` 仅作为
self-containment checker 的兼容输入保留, 不应继续作为新整包完成状态输出。

```text
official_result_provenance = "repository_generated_from_third_party_official_code"
external_baseline_source_video_path
external_baseline_attacked_video_path
external_baseline_generation_model_id
official_execution_manifest_path
```

可用以下命令在当前 `run_root` 上检查项目内官方结果缓存是否足以进入严格门禁:

```bash
python -m external_baseline.official_resource_bootstrap \
  --run-root /content/SSTW_stage_workspace/runs/generative_video_model_probe/probe_paper \
  --resource-root /content/drive/MyDrive/SSTW/resources/external_baseline

python -m external_baseline.official_runtime_closure \
  --run-root /content/SSTW_stage_workspace/runs/generative_video_model_probe/probe_paper \
  --resource-root /content/drive/MyDrive/SSTW/resources/external_baseline \
  --official-result-bundle-root /content/SSTW_stage_workspace/external_baseline_official_result_bundles/probe_paper

python -m external_baseline.official_bundle_generator \
  --run-root /content/SSTW_stage_workspace/runs/generative_video_model_probe/probe_paper \
  --bundle-root /content/SSTW_stage_workspace/external_baseline_official_result_bundles/probe_paper \
  --generate-auto-supported

python -m external_baseline.official_result_bundle \
  --run-root /content/SSTW_stage_workspace/runs/generative_video_model_probe/probe_paper
```

在 Colab `local_zip` 模式下, 上述 `run_root` 和 `bundle_root` 是本地热路径。不要把
重型 official bundle 小文件直接写到 Google Drive 热路径; formal reference Notebook
会在阶段结束后把完整 bundle 打包为 zip 并保存到
`SSTW/<workflow_profile>/external_baseline_official_reference/`。

上述顺序体现严格门禁的修复策略: 不是只检查失败, 而是先自动准备可公开资源、再落盘真实运行闭合要求、
再自动生成可由官方 API 或项目内官方流程运行器支持的 repository-owned official bundle
cache、最后执行 fail-closed preflight。当前 VideoShield 通过
`external_baseline.videoshield_official_runtime` 运行官方 watermark generation、
ModelScope 生成、latent inversion 与 temporal matching; 失败时仍记录为运行缺口,
不会降级为直接检测 SSTW / Wan 视频。若某个 baseline 的官方仓库未公开训练权重、
需要大显存生成模型、需要 PRC key / maintained info 或需要项目内官方中间产物,
bootstrap 与 bundle generator 会写出明确阻断原因, 不会用 SSTW 检测分数或视频相似度
伪造结果, 也不会要求用户外部补交分数文件。

真实运行闭合要求配置位于:

```text
configs/external_baselines/official_runtime_closure_requirements.json
configs/external_baselines/requirements/<baseline_id>.txt
```

`official_runtime_closure.py` 会在 `run_root/artifacts/` 下写出:

```text
external_baseline_official_runtime_closure_requirements.json
```

该 artifact 的性质是“可运行性与缺口清单”, 不是论文结果。它会说明:

1. 当前 `runtime_detection_records.jsonl`、`generation_records.jsonl`、`videos/` 和 `attacked_videos/` 是否存在。
2. 每个 baseline 的官方源码目录和关键入口文件是否存在。
3. 每个 baseline 的 requirements 文件是否存在。
4. 默认 Google Drive 资源目录中是否存在可自动绑定到 `SSTW_*` 环境变量的官方资源。
5. official bundle cache 是否已经覆盖全部 runtime comparison units。

若 artifact 中出现 `environment_updates`, Notebook 会自动应用这些更新。该机制只减少手动填路径,
不会把资源文件或配置文件升级为 `metric_status: measured_formal`。正式结果仍必须由
external baseline runner 读取 official adapter 输出后写入 `external_baseline_score_records.jsonl`。

5 个主实验独立 formal reference Notebook 默认会安装对应的 requirements 文件。若在已安装环境中调试,
可以设置 `SSTW_INSTALL_BASELINE_REQUIREMENTS=false` 跳过该步骤; 正式 Colab 冷启动运行不应跳过。

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

Colab 中必须满足以下三类内容。默认 formal reference Notebook 会自动配置 repository official adapter command; 只有禁用仓库 adapter 或改用自定义命令时, 才需要手动覆盖现代 baseline command:

1. 现代 baseline command:

   ```text
   SSTW_VIDEOSHIELD_EVAL_COMMAND
   SSTW_VIDSIG_EVAL_COMMAND
   SSTW_VIDEOSEAL_EVAL_COMMAND
   SSTW_REVMARK_EVAL_COMMAND
   SSTW_WAM_FRAME_EVAL_COMMAND
   ```

2. 可选的官方 source clone:

   ```text
   RUN_EXTERNAL_BASELINE_SOURCE_CLONE = True 或 False
   ```

   该开关只会对可 git clone 的 source URL 生效。不能 git clone 的论文页面、PDF 或 HTML source 仍需要用户在 Colab 中提供官方命令或手工放置 source snapshot。

3. 额外项目内运行证据路径:

   ```text
   EXTERNAL_BASELINE_EVIDENCE_PATHS = [
       "/content/drive/MyDrive/SSTW/...",
   ]
   ```

这些 evidence path 只能指向本项目 workflow 产生或收集的官方 baseline 运行日志、配置、
输出 JSON、依赖快照或校验文件。adapter 会把路径写入
`artifacts/external_baseline_execution_manifest.json`, 但这些路径不能替代
`metric_status: measured_formal` records。

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

当前 `probe_paper` 已重新定义为 paper 级前的 FPR=10% 小样本全流程打通层。因此在 `SSTW_WORKFLOW_PROFILE=probe_paper`、`SSTW_WORKFLOW_PROFILE=probe_paper`、`SSTW_WORKFLOW_PROFILE=pilot_paper` 或 `SSTW_WORKFLOW_PROFILE=full_paper` 时, Colab notebook 会在 5 个主实验现代 baseline command 缺失时提前阻断, 避免先消耗 GPU 生成视频后才发现 baseline 主表无法闭合。`probe_paper` 是 FPR=10% 小样本论文闭合入口, `pilot_paper` 是 FPR=1% 小规模论文结果入口。

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

现代 baseline 正式运行应先逐个运行 5 个主实验 formal reference Notebook:

```text
paper_workflow/colab_notebooks/videoseal_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/vidsig_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/videoshield_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/revmark_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/wam_frame_formal_reference_colab.ipynb
```


每个 Notebook 都读取同一 `workflow_profile` 的 `drive_run_root`, 以同一
`prompt_id / seed_id / attack_name / trajectory_trace_id` runtime comparison unit
为锚点, 在自己的 baseline 内完成 source intake、project clone / build / run /
adapt、official bundle 生成, 然后默认调用统一 external baseline runner 将当前可用
bundle 转写为 `metric_status: measured_formal` records。若其它 baseline bundle 尚未
完成, 统一 runner 会写出 governed unsupported rows; 这些 rows 只能作为阻断记录,
不能替代正式 baseline 结果。

5 个主实验 official bundle 全部完成后, 直接运行:

```text
paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb
```

该 Notebook 会恢复 5 个 official reference 阶段包, 重新执行全量统一转写、
self-containment 判定、validation / pilot gate 和打包。旧的通用 external baseline
scoring Notebook 已删除, 防止它与 paper gate 的聚合职责重复并造成运行顺序误读。
这样可以把三类失败原因分离:

1. `generative_video_generation_colab.ipynb`、`generative_video_quality_scoring_colab.ipynb`、`sstw_mechanism_postprocess_colab.ipynb`、`runtime_attack_colab.ipynb`、`runtime_detection_colab.ipynb`: 主方法生成、formal metrics、机制后处理、attack 和 detection 是否成功。
2. 5 个主实验 `*_formal_reference_colab.ipynb`: 现代视频水印 baseline official bundle 是否成功。
3. `paper_gate_and_package_colab.ipynb`: 全量统一转写、self-containment、paper profile、pilot-paper 或 full-paper gate 是否满足论文协议。

Notebook 的 profile、Drive 目录和 stage plan 均由以下配置控制:

```text
configs/paper_workflow/generative_video_notebook_workflows.json
```

因此, 切换 `probe_paper` 与 `pilot_paper` 时, 不应复制或改写 Notebook 中的 run_root /
package_dir 字符串, 而应设置:

```text
SSTW_WORKFLOW_PROFILE=probe_paper
或
SSTW_WORKFLOW_PROFILE=pilot_paper
```

`full_paper` profile 当前只登记协议入口, 不允许在未通过 probe_paper、pilot_paper 和后续 full_paper checker 前作为可运行 claim profile 使用。

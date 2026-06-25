# Colab Notebook 运行 workflow

本文档说明如何在 Colab 冷启动环境中执行当前项目的 Notebook workflow。Colab 运行环境不会保留代码、模型、权重或中间结果, 因此每次运行都必须挂载 Google Drive、拉取仓库代码、安装依赖, 并把结果打包保存到 Google Drive。

## 1. 基本原则

1. Notebook 只是远程执行入口, 不承载正式协议逻辑。
2. 正式 records、tables、figures、reports 和 package manifest 必须由仓库模块生成。
3. 不要手写 records, 也不要把 Colab 中临时整理的表格当作论文证据。
4. 当前 Colab 主流程只使用新的 profile-driven workflow。旧的诊断 Notebook 可以用于排查, 但不作为 paper workflow 的正式执行顺序。
5. 默认 Drive 根目录为 `/content/drive/MyDrive/SSTW`, Windows 本地映射通常对应 `G:\我的云端硬盘\SSTW`。

## 2. Colab 冷启动准备

每个 Notebook 都应从空环境开始执行以下准备逻辑:

```python
from google.colab import drive
drive.mount('/content/drive')
```

随后拉取仓库代码并进入仓库目录。Notebook 中已有对应 cell, 冷启动时必须实际执行, 不能假设 `/content/SSTW` 已存在。

```bash
cd /content
git clone <仓库地址> SSTW
cd /content/SSTW
```

如需加载 Hugging Face 模型, 需要在 Colab secret 或环境变量中提供 `HF_TOKEN`。Notebook 只记录 `provided` 或 `not_provided` 状态, 不应记录 token 明文。

如需更换 Drive 项目目录, 可以在 Notebook 前置 cell 中显式设置:

```python
import os
os.environ["SSTW_DRIVE_PROJECT_ROOT"] = "/content/drive/MyDrive/SSTW"
```

除非需要隔离实验, 否则建议保持默认目录, 便于在本地通过 `G:\我的云端硬盘\SSTW` 检查落盘结果。

## 3. 当前推荐执行顺序

### 3.0 validation-scale 单 Notebook 正式门禁测试

如果目标是一次性检查 `validation_scale` 正式门禁是否具备完整可跑通路径, 可以直接使用:

```text
paper_workflow/colab_utils/validation_scale_formal_gate_colab.ipynb
```

该 Notebook 会在同一个 Colab session 中串联:

```text
prompt suite
-> external baseline preflight
-> Wan2.1 runtime generation
-> formal metric scoring
-> motion threshold reuse
-> runtime attack / detection
-> external baseline source intake
-> official result bundle preflight
-> external baseline comparison
-> internal ablation
-> adaptive attack proxy
-> replay/sketch gate
-> Claim-3 downgrade gate
-> statistical confidence interval
-> artifact rebuild dry-run
-> validation_scale_gate
-> drive packaging
```

它的 profile 固定为:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'
```

该单 Notebook 不是旧的多 profile 综合 Notebook。它只用于 validation-scale 正式门禁测试,
不用于 `pilot_paper` 或未来 `full_paper`。如果 Google Drive 中尚未存在已通过的
`motion_calibration` artifact, 默认会提前失败; 只有显式设置
`SSTW_RUN_MOTION_CALIBRATION_IF_MISSING=true` 时, 才会先运行长耗时的 motion calibration
前置流程。

当前 validation-scale 单 Notebook 默认执行严格正式门禁, 因此默认设置为:

```python
SSTW_VALIDATION_SCALE_RUN_THROUGH_TEST = "false"
```

若只想测试工程链路并落盘阻断原因, 可以显式打开 run-through test:

```python
os.environ["SSTW_VALIDATION_SCALE_RUN_THROUGH_TEST"] = "true"
```

该模式只跳过 external baseline preflight 的前置中断, 不会伪造第三方 baseline 分数,
也不会把 `validation_scale_gate_decision` 改成 PASS。严格正式门禁必须保持该变量为
`false`, 并保证 6 个现代 baseline 的 official command 能真实写出 score JSON。

严格门禁还会默认检查 Google Drive 官方结果包目录:

```text
/content/drive/MyDrive/SSTW/external_baseline_official_result_bundles/<workflow_profile>
```

该目录由 `SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT` 控制。对于无法在当前
Colab 会话中即时训练或生成的第三方 baseline, 可以先用其官方代码在独立会话中生成
结果包, 再由本 workflow 读取。结果包仍必须是官方结果, 不能由 SSTW 最终检测分数或
proxy 分数派生。

### 3.1 运动阈值校准

Notebook:

```text
paper_workflow/colab_utils/motion_threshold_calibration_colab.ipynb
```

用途:

- 生成 `motion_calibration` 数据。
- 冻结 motion threshold artifact。
- 只为后续 profile 提供阈值复用依据, 不直接支撑论文 detection claim。

profile 设置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = ''
```

该 Notebook 的默认 role profile 是 `motion_calibration`, 因此通常保持空字符串即可。不要把该 Notebook 切换到 `validation_scale` 或 `pilot_paper`。

主要落盘位置:

```text
/content/drive/MyDrive/SSTW/runs/generative_video_model_probe/motion_calibration
/content/drive/MyDrive/SSTW/packages/generative_video_model_probe/motion_calibration
```

只有缺少 motion threshold artifact、阈值设计发生变化或需要重新校准时, 才需要重新运行该 Notebook。

### 3.2 真实生成与 runtime 级 attack / detection

Notebook:

```text
paper_workflow/colab_utils/generative_video_runtime_colab.ipynb
```

用途:

- 构造 prompt suite。
- 检查 external baseline 的 Colab preflight 配置。
- 加载 Wan2.1 并生成视频。
- 记录 latent / time grid / sampler signature / velocity proxy 或 latent displacement proxy。
- 复用 motion threshold artifact。
- 执行 formal metric、attack runner、detection runner 和 small-scale claim pilot gate。
- 打包当前 run 到 Google Drive。

当前 validation-scale 配置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'
```

该 Notebook 是 validation-scale 与 pilot-paper 两个阶段共用的真实生成入口。切换到 pilot-paper 时只改 profile, 不改运行逻辑。

现代 baseline 使用 bridge 模式时, workflow 默认会把 `SSTW_RUN_EXTERNAL_BASELINE_SOURCE_CLONE`
视为 `true`, 以适配 Colab 冷启动环境。若已经手动挂载或克隆官方源码, 可以显式设置为 `"false"`。

### 3.3 外部 baseline 正式评分

Notebook:

```text
paper_workflow/colab_utils/external_baseline_formal_scoring_colab.ipynb
```

用途:

- 读取统一 workflow profile。
- 执行 external baseline source intake。
- 执行 official result bundle preflight。
- 通过正式 command adapter 调用现代视频水印 baseline。
- 生成 external baseline comparison records。
- 打包 baseline 对比结果到 Google Drive。

当前 validation-scale 配置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'
```

进入 paper gate 前必须配置 6 个现代 baseline command:

```text
SSTW_VIDEOSHIELD_EVAL_COMMAND
SSTW_SIGMARK_EVAL_COMMAND
SSTW_SPDMARK_EVAL_COMMAND
SSTW_VIDEOMARK_EVAL_COMMAND
SSTW_VIDSIG_EVAL_COMMAND
SSTW_VIDEOSEAL_EVAL_COMMAND
```

这些 command 应由 Notebook 传入 adapter, 并写出 governed comparison records。不要把 baseline 的临时日志手动整理成正式对比表。

Notebook 会额外写出以下配置辅助 artifact:

```text
artifacts/external_baseline_command_template_summary.json
```

该 artifact 来自:

```text
configs/external_baselines/modern_baseline_colab_commands.json
```

其作用是列出联网核验后的官方仓库 URL、当前已核验 branch HEAD commit、Colab clone 目录、官方入口候选脚本、外层 bridge 命令模板和 repository official adapter 命令模板。它只帮助配置, 不会把 baseline 视为 `measured_formal`; 只有实际执行官方命令并写出合法 score JSON 后, `external_baseline_runner` 才会写出 `measured_formal` records。

正式 command 有两种配置方式。

### 3.3.1 推荐方式: repository bridge command

默认 Notebook 会使用 repository bridge command 统一 SSTW I/O。此时用户不需要手写
`SSTW_<BASELINE>_EVAL_COMMAND`, 但必须为每个 baseline 配置真正调用官方实现的内部命令:

```text
SSTW_VIDEOSHIELD_OFFICIAL_EVAL_COMMAND
SSTW_SIGMARK_OFFICIAL_EVAL_COMMAND
SSTW_SPDMARK_OFFICIAL_EVAL_COMMAND
SSTW_VIDEOMARK_OFFICIAL_EVAL_COMMAND
SSTW_VIDSIG_OFFICIAL_EVAL_COMMAND
SSTW_VIDEOSEAL_OFFICIAL_EVAL_COMMAND
```

当前仓库已经提供 6 个 fail-closed 的 repository official adapter。validation-scale 单
Notebook 默认设置:

```python
os.environ["SSTW_USE_REPOSITORY_OFFICIAL_BASELINE_ADAPTERS"] = "true"
```

此时 Notebook 会自动把 6 个 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND` 指向:

```text
external_baseline/official_eval_adapters/videoshield.py
external_baseline/official_eval_adapters/sigmark.py
external_baseline/official_eval_adapters/spdmark.py
external_baseline/official_eval_adapters/videomark.py
external_baseline/official_eval_adapters/vidsig.py
external_baseline/official_eval_adapters/videoseal.py
```

这些 adapter 不是替代 baseline, 只负责把官方仓库源码、官方 API、官方 checkpoint
或官方结果产物转换成 SSTW 统一 JSON。若缺少第三方官方权重、key、message、
maintained info 或官方输出文件, adapter 会直接失败, 不会输出 proxy 分数。

内部官方命令必须把官方 detector / extractor 的输出写入:

```text
{official_output_json_path}
```

示例:

```python
import os
os.environ["SSTW_SPDMARK_OFFICIAL_EVAL_COMMAND"] = (
    "python /content/SSTW/external_baseline/primary/spdmark/source/<official_eval_script>.py "
    "--source-video {source_video_path} "
    "--attacked-video {attacked_video_path} "
    "--attack-name {attack_name} "
    "--output-json {official_output_json_path}"
)
```

bridge 会读取 `{official_output_json_path}`, 提取 score 字段, 再写出 SSTW 统一的
`{output_json_path}`。如果缺少内部官方命令, Notebook 会在真实生成前写出:

```text
artifacts/external_baseline_official_bridge_preflight_decision.json
```

并提前失败, 防止只配置 wrapper 壳层却没有真实 baseline 分数。

若某个官方仓库有更适合当前 Colab 环境的原生命令, 可以只覆盖该 baseline 的内部
native 命令, 变量模式为 `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND`。例如:

```python
os.environ["SSTW_SPDMARK_NATIVE_EVAL_COMMAND"] = (
    "python /content/SSTW/external_baseline/primary/spdmark/source/<official_eval_script>.py "
    "--source-video {source_video_path} "
    "--attacked-video {attacked_video_path} "
    "--attack-name {attack_name} "
    "--output-json {official_output_json_path}"
)
```

常见需要额外提供的官方产物包括:

```text
SSTW_VIDEOSHIELD_RESULT_JSON
SSTW_SIGMARK_BIT_ACCURACY_NPZ
SSTW_SPDMARK_EXTRACTOR_PATH
SSTW_SPDMARK_GT_BITS_PATH
SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON
SSTW_VIDSIG_MSG_DECODER_PATH
```

`SSTW_VIDEOSEAL_OFFICIAL_EVAL_COMMAND` 默认可直接调用 VideoSeal 官方 Python API,
但仍需要 Colab 能成功安装 VideoSeal 依赖并下载其官方 checkpoint。

### 3.3.2 官方结果包方式: 解决高显存或训练 checkpoint 阻断

部分现代 baseline 不是“任意输入视频 -> detector score”的后处理水印, 而是绑定
特定生成模型、训练出的 extractor、maintained key / message 或 latent inversion
流程。对于这类方法, 在同一个 Wan2.1 Colab 会话中强行即时复跑并不一定可行。
正式 workflow 支持读取由第三方官方代码预先生成的结果包:

```python
import os
os.environ["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = (
    "/content/drive/MyDrive/SSTW/external_baseline_official_result_bundles/validation_scale"
)
```

每个 baseline 的结果 JSON 可放在以下任一命名位置:

```text
<bundle_root>/<baseline_id>/records/<prompt_id>__<seed_id>__<attack_name>.json
<bundle_root>/<baseline_id>/records/<trajectory_trace_id>__<attack_name>.json
<bundle_root>/<baseline_id>/<prompt_id>/<seed_id>/<attack_name>.json
<bundle_root>/<baseline_id>/<trajectory_trace_id>/<attack_name>.json
```

每个 JSON 至少需要包含以下任一 score 字段:

```text
external_baseline_score
watermark_score
detection_score
score
bit_accuracy
confidence
detected
```

并建议包含:

```text
official_result_provenance = "third_party_official_code"
external_baseline_source_video_path
external_baseline_attacked_video_path
external_baseline_generation_model_id
official_execution_manifest_path
```

workflow 会写出:

```text
artifacts/external_baseline_official_result_bundle_preflight_decision.json
```

若某个 baseline 既没有可直接运行的官方资源, 也没有覆盖全部 runtime comparison unit
的结果包, 该 preflight 会失败。该失败是正式门禁的一部分, 目的是防止把缺权重、
缺 checkpoint 或缺官方中间产物的问题延后到 comparison 阶段才暴露。

### 3.3.3 直接方式: 完全自定义 SSTW 外层命令

如果不使用 bridge, 可以设置:

```python
import os
os.environ["SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS"] = "0"
```

然后直接配置 `SSTW_<BASELINE>_EVAL_COMMAND`, 例如:

```python
import os
os.environ["SSTW_SPDMARK_EVAL_COMMAND"] = (
    "python /content/SSTW/external_baseline/primary/spdmark/source/run_sstw_eval.py "
    "--source-video {source_video_path} "
    "--attacked-video {attacked_video_path} "
    "--attack-name {attack_name} "
    "--output-json {output_json_path}"
)
```

注意: 即使 `SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS` 保持默认启用, 用户显式设置的
`SSTW_<BASELINE>_EVAL_COMMAND` 也会优先于 repository bridge 模板。此时
`external_baseline_official_bridge_preflight_decision.json` 会把对应 baseline 记录为
`official_bridge_direct_eval_baseline_ids`, 不再额外要求该 baseline 的
`SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`。该设计用于支持两种等价的正式接入路径:
一种是 repository bridge + 官方内部命令, 另一种是用户自定义外层命令直接写出 SSTW 合规 JSON。

其中外层命令必须是真实 wrapper: 它可以调用官方源码、官方权重或官方 detector, 但最终必须把结果写入 `{output_json_path}`。输出 JSON 至少需要包含以下任一字段:

```text
external_baseline_score
watermark_score
detection_score
score
bit_accuracy
confidence
detected
```

如果只是 clone 了官方仓库, 但没有能输出上述 JSON 的 wrapper command, preflight 仍应失败。这是刻意保留的 fail-closed 设计, 用于防止把“已找到源码”误判为“已经完成正式 baseline 对比”。

### 3.4 paper gate 与打包

Notebook:

```text
paper_workflow/colab_utils/paper_gate_and_package_colab.ipynb
```

用途:

- 执行 validation internal ablation。
- 执行 adaptive attack proxy。
- 执行 replay/sketch gate。
- 必要时执行 Claim-3 downgrade gate。
- 执行 statistical confidence interval。
- 执行 validation artifact rebuild dry run。
- 对 validation-scale 或 pilot-paper 进行最终 gate 判断。
- 打包完整结果到 Google Drive。

当前 validation-scale 配置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'
```

该 Notebook 必须在 runtime 与 external baseline notebook 完成后运行, 因为 gate 需要读取前序 artifacts。
执行 gate 前, Notebook 会再次校验并复制 `motion_calibration` run root 中已冻结的
`motion_threshold_calibration_decision.json` 到当前 `validation_scale` 或 `pilot_paper`
run root。该步骤不重新估计阈值, 只把独立 calibration split 的阈值 artifact 固化到当前
gate 所需的 governed artifacts 中。

## 4. validation-scale 到 pilot-paper 的切换

validation-scale 是进入 paper 级运行前的最后完整工程门禁。通过后, 可以把以下 3 个 Notebook 中的 profile 从 validation-scale 切到 pilot-paper:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'pilot_paper'
```

需要切换的 Notebook:

```text
paper_workflow/colab_utils/generative_video_runtime_colab.ipynb
paper_workflow/colab_utils/external_baseline_formal_scoring_colab.ipynb
paper_workflow/colab_utils/paper_gate_and_package_colab.ipynb
```

切换后仍按相同顺序执行:

```text
generative_video_runtime_colab.ipynb
external_baseline_formal_scoring_colab.ipynb
paper_gate_and_package_colab.ipynb
```

`pilot_paper` 是小规模跑完整 full paper 协议并产出 pilot 级论文结果。它与后续 full-paper 运行的核心区别只应是样本规模和评价等级, 不应更换判定逻辑、baseline 接口或 artifact 结构。

## 5. 每轮运行后必须检查的 artifacts

每轮 Colab 运行结束后, 应在对应 profile 的 run root 下检查以下文件。validation-scale 的典型目录是:

```text
/content/drive/MyDrive/SSTW/runs/generative_video_model_probe/validation_scale/artifacts
```

关键 decision artifacts:

```text
generative_video_colab_runtime_decision.json
external_baseline_colab_preflight_decision.json
external_baseline_official_result_bundle_preflight_decision.json
external_baseline_comparison_decision.json
validation_internal_ablation_decision.json
adaptive_attack_decision.json
replay_and_sketch_gate_decision.json
claim3_downgrade_decision.json
statistical_confidence_interval_decision.json
validation_artifact_rebuild_dry_run_decision.json
validation_scale_gate_decision.json
pilot_paper_gate_decision.json
```

在 validation-scale 中, `pilot_paper_gate_decision.json` 可能不会生成或不会作为当前 gate 的核心判定。切换到 `pilot_paper` 后, 需要重点检查 `pilot_paper_gate_decision.json`。

package 输出应位于:

```text
/content/drive/MyDrive/SSTW/packages/generative_video_model_probe/<workflow_profile>
```

package 文件名采用 `<utc_time>_<short_commit>` 批次标识, 便于定位同一轮 Colab 产物。

## 6. 真实工作量进度显示

长耗时 repository runner 会按实际 plan 或 records 数量输出工作量进度, 而不是按 Notebook cell 数量输出进度。典型显示形式为:

```text
SSTW 工作量进度 | wan21_runtime_generation | 7/24 (29.2%) | elapsed=42.0 min | eta=102.0 min | profile=validation_scale prompt=... seed=...
```

进度总数由运行时实际数据结构自动计算:

```text
Wan2.1 生成: len(plan)
formal metric 视频扫描: len(generation_records)
runtime attack 视频变换: len(eligible_generation_records) * len(attack_names)
runtime detection 视频扫描: len(runtime_attack_records)
external baseline adapter 矩阵: len(baseline_records)
单个 baseline 读取 runtime 视频: len(comparable_detection_records)
```

因此 `validation_scale`、`pilot_paper` 和未来 `full_paper` 的样本数量不同, 进度总数会自动变化, 不需要在 Notebook 中硬编码 24、168 或 224。

该进度显示只写 stdout, 不写入正式 records、tables、figures、reports、manifests 或 claim artifacts。若自动化环境需要静默运行, 可以设置:

```bash
export SSTW_PROGRESS=0
```

Colab Notebook 调用仓库命令时必须使用 `paper_workflow.notebook_utils.streaming_command.run_streaming_command`。该 helper 会逐行转发子进程输出, 避免 `subprocess.run(..., capture_output=True)` 把 `SSTW 工作量进度` 缓存到任务结束后才显示。若长时间没有看到进度, 优先确认 Colab 中的仓库代码已经拉取到包含该 helper 的最新提交。

为避免 Colab 输出被第三方库刷屏, workflow 默认压制 Hugging Face / Diffusers / tqdm 的内部下载、加载和单次采样进度条。默认应主要看到如下 SSTW 外层进度:

```text
SSTW 工作量进度 | video_generation_model_load | start | model=...
SSTW 工作量进度 | video_generation_model_load | finish | model=... | pipeline_progress_bar=disabled
SSTW 工作量进度 | wan21_runtime_generation | 1/24 (4.2%) | elapsed=...
```

该压制只影响屏幕日志, 不影响 callback latent、time grid、sampler signature、records 或 package 落盘。若需要调试第三方库内部下载或采样细节, 可以在 Notebook 前置 cell 中显式打开:

```python
import os
os.environ["SSTW_SUPPRESS_THIRD_PARTY_PROGRESS"] = "0"
os.environ["SSTW_ENABLE_PIPELINE_PROGRESS_BAR"] = "1"
```

除非正在排查模型下载、权重加载或单次 pipeline 调用失败, 否则不建议开启上述调试开关, 因为这会重新显示 `Fetching files`、`Loading checkpoint shards` 和单视频采样 step 等大量内部进度。

## 7. 常见失败原因

1. 未先运行或未保留 `motion_calibration` artifact, 导致 motion threshold reuse 失败。
2. 没有配置 6 个现代 baseline command, 或没有提供完整 official result bundle, 导致 external baseline preflight / bundle preflight / paper gate 阻断。
3. Colab 冷启动后没有重新 clone 仓库或安装依赖, 导致模块导入失败。
4. 直接运行 `paper_gate_and_package_colab.ipynb`, 但 runtime 与 baseline artifacts 不存在。
5. Colab 断开前没有打包到 Google Drive, 导致本地临时结果丢失。
6. 手动修改最终 detection score 或正式 records, 导致 harness 审计失败。

## 8. 诊断 Notebook 的使用边界

以下 Notebook 只作为诊断或历史机制调试入口, 不属于当前 paper workflow 的正式顺序:

```text
paper_workflow/colab_utils/wan21_flow_adapter_preflight_colab.ipynb
paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb
```

当 Wan2.1 无法加载、callback 捕获不到 latent、time grid 或 sampler signature 记录异常时, 可以先运行 `wan21_flow_adapter_preflight_colab.ipynb` 排查 adapter。排查通过后, 仍应回到第 3 节的主流程。

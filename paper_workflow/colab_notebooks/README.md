# Colab Notebook 运行 workflow

本文档说明如何在 Colab 冷启动环境中执行当前项目的 Notebook workflow。Colab 运行环境不会保留代码、模型、权重或中间结果, 因此每次运行都必须挂载 Google Drive、拉取仓库代码、安装依赖, 并把结果打包保存到 Google Drive。

## 1. 基本原则

1. Notebook 只是远程执行入口, 不承载正式协议逻辑。
2. 正式 records、tables、figures、reports 和 package manifest 必须由仓库模块生成。
3. 不要手写 records, 也不要把 Colab 中临时整理的表格当作论文证据。
4. 当前 Colab 主流程只使用新的 profile-driven workflow。旧的诊断 Notebook 可以用于排查, 但不作为 paper workflow 的正式执行顺序。
5. 默认 Drive 根目录为 `/content/drive/MyDrive/SSTW`, Windows 本地映射通常对应 `G:\我的云端硬盘\SSTW`。
6. Notebook 的热路径必须使用阶段 zip 交接: 先把前置阶段 zip 从 Drive 复制到 `/content` 本地解压, 后续 runner 只读写本地 workspace, 阶段结束后再把单个 zip 和 manifest 写回 Drive。

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

## 2.1 阶段 zip 交接与本地 workspace

为避免 Colab 对 Google Drive 大量小文件循环读取触发限制, 所有正式 Notebook 默认启用:

```python
os.environ["SSTW_COLAB_STAGE_IO_MODE"] = "local_zip"
```

启用后路径语义如下:

```text
Google Drive: 只保存阶段 zip、stage package manifest 和可复用资源包 / checkpoint。
/content/SSTW_stage_workspace: 当前 Notebook 的本地热路径, runner 在这里读写 records、artifacts、videos 和 official bundle。
/content/SSTW_stage_packages: 当前 Colab 会话复制过来的 zip 缓存。
```

启用 `local_zip` 后, Notebook 初始化阶段不应在 Drive 上预创建 `runs/`、`logs/`
或 `datasets/` 热路径空目录。只有阶段发布成功时, `publish_colab_stage_package`
才会创建对应的冷归档目录并写入 zip / manifest。

`SSTW/resources/external_baseline/` 用于保存可复用的 official checkpoint、模型和官方资源。
若该目录下存在资源 zip, Notebook 会先复制 zip 到 `/content` 本地缓存并解压到本地资源根目录,
VidSig 的 message decoder / VAE checkpoint 等可复用资源应优先以资源包形式保存,
避免运行时循环读取 Google Drive 小文件。

每个阶段完成后会额外写出阶段交接包:

```text
/content/drive/MyDrive/SSTW/<workflow_profile>/<stage_package_id>/<workflow_profile>_<stage_package_id>_<YYYYMMDD_HHMMSS>_<git_short_commit>.zip
```

其中 `motion_threshold_calibration_colab.ipynb` 写入 `/content/drive/MyDrive/SSTW/motion_threshold/`;
5 个主实验 baseline 专用 Notebook 写入
`/content/drive/MyDrive/SSTW/<workflow_profile>/external_baseline_official_reference/`;
历史或辅助 Notebook 写入 `/content/drive/MyDrive/SSTW/helper/`。阶段包默认保留时间戳,
后续 Notebook 会按文件名时间戳选择最新的完整 zip。

后续 Notebook 不应直接从:

```text
/content/drive/MyDrive/SSTW/runs/...
/content/drive/MyDrive/SSTW/external_baseline_official_result_bundles/...
```

循环读取大量小文件。正确流程是复制对应阶段最新时间戳 zip 到本地、解压、在本地路径继续运行。
旧版 `packages/.../*.zip` 不再作为 Notebook 间自动交接入口。
若用户绕过 Notebook 直接运行 `scripts/package_results/*_drive_packager.py`, CLI 默认
输出也会解析到上述阶段归档目录, 不会重新创建旧版 `SSTW/packages/`。

## 3. 当前推荐执行顺序

### 3.0 validation-scale 拆分式正式门禁流程

当前不保留单 Notebook 全流程入口。`validation_scale` 必须按阶段拆分执行, 以便每个阶段的输入、输出、阻断原因和重跑边界都能独立审计。

推荐顺序为:

```text
motion_threshold_calibration_colab.ipynb  # 仅当 motion calibration artifact 缺失时运行
-> generative_video_generation_colab.ipynb
-> generative_video_quality_scoring_colab.ipynb
-> sstw_mechanism_postprocess_colab.ipynb
-> runtime_attack_colab.ipynb
-> runtime_detection_colab.ipynb
-> 5 个主实验 modern external baseline formal reference Notebook
-> formal_comparison_scoring_colab.ipynb
-> paper_evidence_postprocess_colab.ipynb
-> paper_gate_and_package_colab.ipynb
```

该拆分设计的主要考虑在于:

1. runtime 生成、external baseline 官方运行、正式 baseline scoring 和 paper gate 的失败原因不同, 不应混在一个 Notebook 中。
2. 5 个主实验现代 external baseline 的官方源码、权重、key、显存和运行入口差异较大, 需要允许逐个 Notebook 独立重跑。
3. `validation_scale` 是 paper 级前的小样本全流程打通门禁, 不支持 full_paper 规模最终效果证明; 通过后只能进入 `probe_paper`。`probe_paper` 是 `target_fpr=0.1` 的小样本论文闭合验证层, 通过后才允许进入 `pilot_paper`。
4. `non-run record` 只能作为阻断记录, 不能替代 `metric_status: measured_formal` baseline 结果。

### 3.1 运动阈值校准

Notebook:

```text
paper_workflow/colab_notebooks/motion_threshold_calibration_colab.ipynb
```

用途:

- 生成 `motion_calibration` 数据。
- 冻结 motion threshold artifact。
- 只为后续 profile 提供阈值复用依据, 不直接支撑论文 detection claim。

profile 设置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = ''
```

该变量位于 Notebook 第一个代码 cell 的第一行, 便于打开 Notebook 后直接切换。该 Notebook 的默认 role profile 是 `motion_calibration`, 因此通常保持空字符串即可。不要把该 Notebook 切换到 `validation_scale` 或 `pilot_paper`。

主要落盘位置:

```text
/content/SSTW_stage_workspace/runs/generative_video_model_probe/motion_calibration
/content/drive/MyDrive/SSTW/motion_threshold
```

只有缺少 motion threshold artifact、阈值设计发生变化或需要重新校准时, 才需要重新运行该 Notebook。

### 3.2 SSTW 主方法拆分式 runtime 阶段

Notebook:

```text
paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb
paper_workflow/colab_notebooks/generative_video_quality_scoring_colab.ipynb
paper_workflow/colab_notebooks/sstw_mechanism_postprocess_colab.ipynb
paper_workflow/colab_notebooks/runtime_attack_colab.ipynb
paper_workflow/colab_notebooks/runtime_detection_colab.ipynb
```

用途:

- `generative_video_generation_colab.ipynb`: 构造 prompt suite, 加载 Wan2.1 并生成 clean / SSTW 视频, 记录 latent / time grid / sampler signature / velocity proxy 或 latent displacement proxy。
- `generative_video_quality_scoring_colab.ipynb`: 恢复 generation 阶段包, 执行正式视频质量、运动与语义 metric, 并复用已冻结 motion threshold artifact。
- `sstw_mechanism_postprocess_colab.ipynb`: 恢复 generation 与 quality scoring 阶段包, 执行机制后处理和 protocol evaluation matrix 后处理。
- `runtime_attack_colab.ipynb`: 恢复 generation 与 quality scoring 阶段包, 执行 46 个 runtime attack 并产出 attacked videos。
- `runtime_detection_colab.ipynb`: 恢复 generation 与 runtime attack 阶段包, 执行 SSTW runtime detection, 产出后续公平比较所需本方法 detection records。

当前 probe-paper 配置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'probe_paper'
```

这些 Notebook 是 validation-scale、probe-paper、pilot-paper 与 full-paper 共用的同构主方法运行入口。`SSTW_WORKFLOW_PROFILE_VALUE` 位于每个 Notebook 第一个代码 cell 的第一行。切换 profile 时只改这一行, 不改阶段顺序、命令映射或产物清单。

现代 baseline 使用 bridge 模式时, workflow 默认会把 `SSTW_RUN_EXTERNAL_BASELINE_SOURCE_CLONE`
视为 `true`, 以适配 Colab 冷启动环境。若已经手动挂载或克隆官方源码, 可以显式设置为 `"false"`。

### 3.3 外部 baseline official reference

推荐 Notebook:

```text
paper_workflow/colab_notebooks/videoseal_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/vidsig_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/videoshield_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/revmark_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/wam_frame_formal_reference_colab.ipynb
```


用途:

- 读取统一 workflow profile。
- 在各自 Notebook 内执行当前 baseline 的 source intake、clone / build / run / adapt / record。
- 以同一 prompt / seed / attack 协议为锚点生成当前 baseline 的 official bundle。
- 只打包当前 baseline 的 official bundle 与对应 governed artifacts。
- 将阶段包保存到 `/content/drive/MyDrive/SSTW/<workflow_profile>/external_baseline_official_reference/`。
- `formal_comparison_scoring_colab.ipynb` 恢复 5 个主实验 official bundle 后, 再统一重建最终 `measured_formal` records。

当前 probe-paper 配置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'probe_paper'
```

该变量位于每个 baseline Notebook 第一个代码 cell 的第一行。切换到 `probe_paper`、`pilot_paper` 或 `full_paper` 时只改这一行, 不改 baseline helper、runner 或打包逻辑。

进入 formal comparison scoring 前必须完成 5 个主实验现代 baseline 的 official reference 阶段包。若使用通用 command adapter,
仍必须配置 5 个主实验现代 baseline command:

```text
SSTW_VIDEOSHIELD_EVAL_COMMAND
SSTW_VIDSIG_EVAL_COMMAND
SSTW_VIDEOSEAL_EVAL_COMMAND
SSTW_REVMARK_EVAL_COMMAND
SSTW_WAM_FRAME_EVAL_COMMAND
```

这些 command 应由 Notebook 传入 adapter, 并写出 governed comparison records。不要把 baseline 的临时日志手动整理成正式对比表。

Notebook 会额外写出以下配置辅助 artifact:

```text
artifacts/external_baseline_command_template_summary.json
artifacts/external_baseline_official_resource_bootstrap_decision.json
artifacts/external_baseline_official_runtime_closure_requirements.json
artifacts/external_baseline_official_bundle_generation_decision.json
```

该 artifact 来自:

```text
configs/external_baselines/modern_baseline_colab_commands.json
```

其作用是列出联网核验后的官方仓库 URL、当前已核验 branch HEAD commit、Colab clone 目录、官方入口候选脚本、外层 bridge 命令模板和 repository official adapter 命令模板。它只帮助配置, 不会把 baseline 视为 `measured_formal`; 只有实际执行官方命令并写出合法 score JSON 后, `external_baseline_runner` 才会写出 `measured_formal` records。

正式 command 有两种配置方式。

### 3.3.0 现代 external baseline 独立官方参考 Notebook

若目标是逐个闭合 5 个主实验现代 external baseline 的官方参考结果包, 推荐先按以下顺序运行独立 Notebook:

```text
paper_workflow/colab_notebooks/videoseal_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/vidsig_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/videoshield_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/revmark_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/wam_frame_formal_reference_colab.ipynb
```


这些 Notebook 的职责是:

```text
clone / build / run / adapt / bundle
```

单 baseline Notebook 默认不执行全量 unified measured_formal 转写, 避免在 WAM-frame
等独立入口日志中继续出现 VideoSeal / VideoShield 等其它 baseline 的 scoring 进度。
统一转写、self-containment 判定和最终公平比较由 `formal_comparison_scoring_colab.ipynb`
在恢复 5 个 official reference 阶段包后集中执行。

每个独立 Notebook 会先安装对应的 baseline requirements 文件:

```text
configs/external_baselines/requirements/<baseline_id>.txt
```

若需要临时跳过该安装步骤, 可设置 `SSTW_INSTALL_BASELINE_REQUIREMENTS=false`;
但这只适合调试已安装环境, 正式 Colab 冷启动运行应保持默认启用。

也就是在项目内克隆官方源码、运行 source intake、调用仓库 official adapter、官方 API
或项目内官方流程运行器,
并以同一 `prompt_id / seed_id / attack_name / trajectory_trace_id` runtime comparison unit
为锚点, 把每个 baseline 的官方结果写入:

```text
/content/SSTW_stage_workspace/external_baseline_official_result_bundles/validation_scale
```

阶段完成后, 该本地 official bundle 会随对应 baseline 的阶段 zip 写入:

```text
/content/drive/MyDrive/SSTW/validation_scale/external_baseline_official_reference
```

其产物性质必须明确区分:

- `*_formal_reference_decision.json` 说明该 baseline 的 official bundle 是否生成完整。
- `official_reference_execution_manifest.json` 记录 clone / build / run / adapt 过程证据。
- Notebook cell 不直接手写 `metric_status: measured_formal` records。
- 每个 baseline Notebook 默认只生成并打包当前 baseline 的 official bundle。
- 若需要调试旧的后续转写路径, 可以显式设置
  `SSTW_RUN_EXTERNAL_BASELINE_COMPARISON_AFTER_REFERENCE=true` 和
  `SSTW_RUN_SELF_CONTAINMENT_AFTER_REFERENCE=true`, 但正式流程不推荐这样做。

因此, 每个独立 Notebook 的执行闭环是:

```text
official_reference_notebook
-> current_baseline_official_bundle
-> current_baseline_stage_zip
```

`formal_comparison_scoring_colab.ipynb` 会在恢复 5 个主实验 baseline official reference 阶段包后,
执行全量统一转写、self-containment 判定、clean negative 公平校准和差值区间统计。
`paper_evidence_postprocess_colab.ipynb` 再消费 formal comparison scoring、runtime 与 motion threshold 阶段包,
生成内部消融、adaptive attack、CI 和低 FPR 等最终门禁前辅助证据。
`paper_gate_and_package_colab.ipynb` 恢复 runtime、motion threshold、formal comparison scoring 和 paper evidence postprocess 阶段包并执行最终门禁,
但不再直接恢复 5 个 baseline 大包或重复运行 external baseline scoring。

最终必须检查:

```text
records/external_baseline_score_records.jsonl
artifacts/external_baseline_comparison_decision.json
artifacts/external_baseline_self_containment_decision.json
```

其中 5 个主实验现代 baseline 的正式记录必须全部为:

```text
metric_status: measured_formal
```

`non-run record` 只能作为阻断原因记录, 不能替代正式 measured baseline, 也不能支持论文效果主张。

### 3.3.1 推荐方式: repository bridge command

默认 Notebook 会使用 repository bridge command 统一 SSTW I/O。此时用户不需要手写
`SSTW_<BASELINE>_EVAL_COMMAND`, 但必须为每个 baseline 配置真正调用官方实现的内部命令:

```text
SSTW_VIDEOSHIELD_OFFICIAL_EVAL_COMMAND
SSTW_VIDSIG_OFFICIAL_EVAL_COMMAND
SSTW_VIDEOSEAL_OFFICIAL_EVAL_COMMAND
SSTW_REVMARK_OFFICIAL_EVAL_COMMAND
SSTW_WAM_FRAME_OFFICIAL_EVAL_COMMAND
```

当前仓库已经提供 5 个主实验 fail-closed repository official adapter。formal comparison scoring Notebook
和 5 个主实验独立 formal reference Notebook 默认设置:

```python
os.environ["SSTW_USE_REPOSITORY_OFFICIAL_BASELINE_ADAPTERS"] = "true"
```

此时 Notebook 会自动把 5 个主实验 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND` 指向:

```text
external_baseline/official_eval_adapters/videoshield.py
external_baseline/official_eval_adapters/vidsig.py
external_baseline/official_eval_adapters/videoseal.py
external_baseline/official_eval_adapters/revmark.py
external_baseline/official_eval_adapters/wam_frame.py
```

这些 adapter 不是替代 baseline, 只负责把官方仓库源码、官方 API、官方 checkpoint
或项目内 official bundle cache 转换成 SSTW 统一 JSON。若缺少第三方官方权重、key、message、
maintained info 或官方输出文件, adapter 会直接失败, 不会输出 proxy 分数。

5 个 baseline 的官方流程边界如下:

- VideoSeal: 使用官方 API 对同一 source video 后处理嵌入并检测, 再按项目 runtime attack 产出同锚点 score。
- VideoShield: 使用项目内 official runtime 调用官方水印、反演和 temporal matching 流程, 生成 project-owned official bundle。
- VidSig: 使用官方 `generate_ms.py -> attack.py` 路径, 在 VidSig 自身生成模型内完成水印视频生成和检测。
- REVMark: 使用官方 Encoder / Decoder 对同一 source video 后处理嵌入, 再以 bit accuracy 作为检测分数。
- WAM-frame: 作为图像水印逐帧适配视频 baseline, 对同一 source video 逐帧嵌入和检测, 再聚合为视频级 bit accuracy。

上述流程都会写入 project-owned official bundle。它们仍不直接手写 `metric_status: measured_formal`; 正式记录仍由统一 external baseline runner 转写。

VidSig 的 `vidsig_formal_reference_colab.ipynb` 使用项目内官方流程运行器:

```text
external_baseline/vidsig_official_runtime.py
```

该运行器默认执行 VidSig 官方 `generate_ms.py -> attack.py` 路径: 先用同一批
SSTW runtime prompt / seed 生成 VidSig 自己的 clean / watermarked videos, 再对
VidSig watermarked videos 应用同名项目 runtime attack, 最后调用官方 `attack.py`
生成 project-owned official bundle。该设计的主要考虑在于 VidSig 是生成过程中嵌入水印的方法,
正式比较不能把 SSTW / Wan 生成的视频直接送入 VidSig detector 后当作 baseline 结果。
直接检测给定视频的 adapter 路径仅保留为显式诊断入口, 默认 fail-closed。

内部官方命令必须把官方 detector / extractor 的输出写入:

```text
{official_output_json_path}
```


bridge 会读取 `{official_output_json_path}`, 提取 score 字段, 再写出 SSTW 统一的
`{output_json_path}`。如果缺少内部官方命令, Notebook 会在真实生成前写出:

```text
artifacts/external_baseline_official_bridge_preflight_decision.json
```

并提前失败, 防止只配置 wrapper 壳层却没有真实 baseline 分数。

若某个官方仓库有更适合当前 Colab 环境的原生命令, 可以只覆盖该 baseline 的内部
native 命令, 变量模式为 `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND`。例如可通过对应 baseline 的 `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND` 覆盖。

常见需要额外提供的官方产物包括:

```text
SSTW_VIDEOSHIELD_RESULT_JSON
SSTW_VIDSIG_MSG_DECODER_PATH
SSTW_VIDSIG_VAE_CHECKPOINT_PATH
SSTW_REVMARK_ENCODER_CHECKPOINT_PATH
SSTW_REVMARK_DECODER_CHECKPOINT_PATH
SSTW_WAM_FRAME_CHECKPOINT_PATH
```

REVMark 默认使用官方仓库自带的 Encoder / Decoder checkpoint。WAM-frame 默认从公开 URL 下载 WAM MIT checkpoint, 并作为图像水印逐帧适配视频 baseline 参与同一 prompt / seed / attack 公平比较。

`SSTW_VIDEOSEAL_OFFICIAL_EVAL_COMMAND` 默认可直接调用 VideoSeal 官方 Python API,
但仍需要 Colab 能成功安装 VideoSeal 依赖并下载其官方 checkpoint。

### 3.3.2 官方结果包方式: 解决高显存或训练 checkpoint 阻断

部分现代 baseline 不是“任意输入视频 -> detector score”的后处理水印, 而是绑定
特定生成模型、训练出的 extractor、maintained key / message 或 latent inversion
流程。对于这类方法, 在同一个 Wan2.1 Colab 会话中强行即时复跑并不一定可行。
正式 workflow 支持读取由本项目 workflow 调用官方代码生成的 official bundle cache:

```python
import os
os.environ["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = (
    "/content/SSTW_stage_workspace/external_baseline_official_result_bundles/validation_scale"
)
```

在 `local_zip` 模式下, 该路径通常由 Notebook layout 自动指向本地 workspace。不要把
热路径直接设置到 Google Drive 上循环读写大量小文件; 阶段结束时 Notebook 会把完整
official bundle 打包成单个 zip 保存到
`/content/drive/MyDrive/SSTW/<workflow_profile>/external_baseline_official_reference/`。

每个 baseline 的结果 JSON 只能由 formal reference Notebook、repository official adapter
或 official bundle generator 写入, 命名位置可采用以下任一形式:

```text
<bundle_root>/<baseline_id>/records/<prompt_id>__<seed_id>__<attack_name>.json
<bundle_root>/<baseline_id>/records/<trajectory_trace_id>__<attack_name>.json
<bundle_root>/<baseline_id>/<prompt_id>/<seed_id>/<attack_name>.json
<bundle_root>/<baseline_id>/<trajectory_trace_id>/<attack_name>.json
```

每个 JSON 必须由项目内 official bundle cache 生成流程写出, 且至少包含以下任一 score 字段:

```text
external_baseline_score
watermark_score
detection_score
score
bit_accuracy
confidence
detected
```

同时必须包含:

```text
official_result_provenance = "repository_generated_from_third_party_official_code"
external_baseline_source_video_path
external_baseline_attacked_video_path
external_baseline_generation_model_id
official_execution_manifest_path
```

workflow 会写出:

```text
artifacts/external_baseline_official_resource_bootstrap_decision.json
artifacts/external_baseline_official_runtime_closure_requirements.json
artifacts/external_baseline_official_bundle_generation_decision.json
artifacts/external_baseline_official_result_bundle_preflight_decision.json
```

`external_baseline_official_resource_bootstrap_decision.json` 会记录哪些 baseline 已由
Colab 自动补齐资源, 哪些 baseline 仍需要官方 bundle、官方 checkpoint、训练权重或更高显存
环境。`external_baseline_official_bundle_generation_decision.json` 会记录自动生成了哪些
official bundle, 以及哪些 baseline 因官方资源边界无法自动生成。

`external_baseline_official_runtime_closure_requirements.json` 是真实运行闭合要求清单,
由以下配置生成:

```text
configs/external_baselines/official_runtime_closure_requirements.json
configs/external_baselines/requirements/<baseline_id>.txt
```

该 artifact 会逐个 baseline 检查官方源码关键文件、requirements 文件、默认 Drive
资源路径、runtime videos、official bundle cache 和命令环境变量。若 Google Drive
中已经存在配置声明的默认资源文件, Notebook 会自动应用 artifact 中的
`environment_updates`, 因而不需要在 cell 中重复手动填写路径。该 artifact 仍然只是
preflight 产物, 不能替代 `metric_status: measured_formal`。

若某个 baseline 既没有可直接运行的官方资源, 也没有覆盖全部 runtime comparison unit
的结果包, 该 preflight 会失败。该失败是正式门禁的一部分, 目的是防止把缺权重、
缺 checkpoint 或缺官方中间产物的问题延后到 comparison 阶段才暴露。

### 3.3.3 直接方式: 完全自定义 SSTW 外层命令

如果不使用 bridge, 可以设置:

```python
import os
os.environ["SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS"] = "0"
```

然后直接配置对应 baseline 的 `SSTW_<BASELINE>_EVAL_COMMAND`。

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

### 3.4 formal comparison scoring

Notebook:

```text
paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb
```

用途:

- 恢复 `sstw_mechanism_postprocess_colab` 与 `runtime_detection_colab` 阶段包。
- 恢复 5 个主实验 baseline official reference 阶段包。
- 统一执行 `sstw_measured_formal_result`、`external_baseline_comparison`、`external_baseline_self_containment_decision`、`fair_detection_calibration`、`formal_method_baseline_comparison` 和 `formal_baseline_difference_interval`。
- 只打包公平比较 records、tables、reports 和 decision artifacts, 不重复打包上游视频或 official bundle。

该 Notebook 是 SSTW 与 external baseline 证据层级对齐的位置。所有正式 external baseline 主表结果必须在这里形成 `metric_status: measured_formal`。

### 3.5 paper evidence postprocess

Notebook:

```text
paper_workflow/colab_notebooks/paper_evidence_postprocess_colab.ipynb
```

用途:

- 恢复 5 个 SSTW runtime 拆分阶段包、`motion_threshold_calibration_colab` 和 `formal_comparison_scoring_colab` 阶段包。
- 执行 motion consistency exclusion 说明。
- 执行 validation internal ablation。
- 执行 adaptive attack proxy。
- 执行 replay/sketch gate。
- 必要时执行 Claim-3 downgrade gate。
- 执行 statistical confidence interval 和 low-FPR formal statistics。
- 执行 data split and leakage guard。
- 只打包最终门禁前辅助证据 records、tables、reports 和 decision artifacts, 不重复打包上游视频或 official bundle。

该 Notebook 是最终 paper gate 的证据后处理层。其产物仍属于 governed evidence, 但不直接执行 validation-scale / pilot-paper 最终通过判定。

### 3.6 paper gate 与打包

Notebook:

```text
paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb
```

用途:

- 恢复 5 个 SSTW runtime 拆分阶段包、`motion_threshold_calibration_colab`、`formal_comparison_scoring_colab` 和 `paper_evidence_postprocess_colab` 阶段包, 但只执行最终门禁与打包相关阶段。
- 执行 validation artifact rebuild dry run。
- 对 validation-scale、probe-paper 或 pilot-paper 进行最终 gate 判断。
- 执行 validation_scale -> probe_paper、probe_paper -> pilot_paper 或 pilot_paper -> full_paper 的跳转判定。
- 构建 validation-scale gate figure 和 package manifest。
- 打包完整结果到 Google Drive。

当前 probe-paper 配置:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'probe_paper'
```

该 Notebook 必须在 5 个 SSTW runtime 拆分 Notebook、5 个主实验 external baseline official reference Notebook、`formal_comparison_scoring_colab.ipynb` 和 `paper_evidence_postprocess_colab.ipynb` 完成后运行, 因为 gate 需要读取前序 artifacts。
执行 gate 前, 前序 evidence postprocess 阶段已经校验并复制 `motion_calibration` run root 中已冻结的
`motion_threshold_calibration_decision.json` 到当前 `validation_scale`、`probe_paper` 或 `pilot_paper`
run root。该步骤不重新估计阈值, 只把独立 calibration split 的阈值 artifact 固化到当前
gate 所需的 governed artifacts 中。

## 4. validation-scale 到 probe-paper 再到 pilot-paper 的切换

validation-scale 是进入 paper 级运行前的小样本全流程打通门禁, 不支持正式效果主张。通过后, 只能把 5 个 SSTW runtime 拆分 Notebook、5 个主实验 baseline official reference Notebook、formal comparison scoring Notebook、paper evidence postprocess Notebook 和 paper gate Notebook 的 profile 从 validation-scale 切到 probe-paper:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'probe_paper'
```

probe-paper 是 `target_fpr=0.1` 的小样本论文闭合验证层。只有 probe-paper 通过并生成 `probe_paper_to_pilot_paper_transition_decision.json` 后, 才能继续把 profile 切到 pilot-paper:

```python
SSTW_WORKFLOW_PROFILE_VALUE = 'pilot_paper'
```

需要切换的 Notebook:

```text
paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb
paper_workflow/colab_notebooks/generative_video_quality_scoring_colab.ipynb
paper_workflow/colab_notebooks/sstw_mechanism_postprocess_colab.ipynb
paper_workflow/colab_notebooks/runtime_attack_colab.ipynb
paper_workflow/colab_notebooks/runtime_detection_colab.ipynb
paper_workflow/colab_notebooks/*_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb
paper_workflow/colab_notebooks/paper_evidence_postprocess_colab.ipynb
paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb
```

切换后仍按相同顺序执行:

```text
generative_video_generation_colab.ipynb
generative_video_quality_scoring_colab.ipynb
sstw_mechanism_postprocess_colab.ipynb
runtime_attack_colab.ipynb
runtime_detection_colab.ipynb
5 个主实验 *_formal_reference_colab.ipynb
formal_comparison_scoring_colab.ipynb
paper_evidence_postprocess_colab.ipynb
paper_gate_and_package_colab.ipynb
```

`probe_paper` 是 FPR=10% 小样本论文闭合验证层; `pilot_paper` 是 FPR=1% 小规模跑完整 full paper 协议并产出 pilot 级论文结果。二者与后续 full-paper 运行的核心区别只应是样本规模和评价等级, 不应更换判定逻辑、baseline 接口或 artifact 结构。

## 5. 每轮运行后必须检查的 artifacts

每轮 Colab 运行结束后, 应在对应 profile 的 run root 下检查以下文件。validation-scale 的典型目录是:

```text
/content/drive/MyDrive/SSTW/runs/generative_video_model_probe/validation_scale/artifacts
```

关键 decision artifacts:

```text
generative_video_colab_runtime_decision.json
external_baseline_official_resource_bootstrap_decision.json
external_baseline_official_runtime_closure_requirements.json
external_baseline_official_bundle_generation_decision.json
external_baseline_official_result_bundle_preflight_decision.json
external_baseline_comparison_decision.json
external_baseline_self_containment_decision.json
fair_detection_calibration_decision.json
formal_method_baseline_comparison_decision.json
formal_baseline_difference_interval_decision.json
validation_internal_ablation_decision.json
adaptive_attack_decision.json
replay_and_sketch_gate_decision.json
claim3_downgrade_decision.json
statistical_confidence_interval_decision.json
low_fpr_formal_statistics_decision.json
motion_consistency_exclusion_decision.json
data_split_and_leakage_guard_decision.json
validation_artifact_rebuild_dry_run_decision.json
validation_scale_gate_decision.json
validation_scale_to_probe_paper_transition_decision.json
```

在 validation-scale 中, `pilot_paper_gate_decision.json` 不应作为当前 gate 的核心判定。
`validation_scale_gate_decision.json == PASS` 后只能生成
`validation_scale_to_probe_paper_transition_decision.json`, 然后进入 `probe_paper`; `probe_paper` 通过后再生成 `probe_paper_to_pilot_paper_transition_decision.json` 并进入 `pilot_paper`;
不能直接进入 `full_paper`。切换到 `pilot_paper` 后, 才需要重点检查
`pilot_paper_gate_decision.json`。

validation-scale 还应生成以下派生产物:

```text
figures/validation_scale_gate_figure.json
manifests/validation_scale_package_manifest.json
```

package 输出应位于:

```text
/content/drive/MyDrive/SSTW/<workflow_profile>/<stage_package_id>
```

package 文件名采用 `<workflow_profile>_<stage_package_id>_<YYYYMMDD_HHMMSS>_<git_short_commit>.zip` 格式, 便于定位同一轮 Colab 产物。
Notebook 间复用优先读取阶段交接包:

```text
/content/drive/MyDrive/SSTW/<workflow_profile>/<stage_package_id>/<workflow_profile>_<stage_package_id>_*.zip
```

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

因此 `validation_scale`、`probe_paper`、`pilot_paper` 和 `full_paper` 的样本数量不同, 进度总数会自动变化, 不需要在 Notebook 中硬编码 24、168 或 1000。

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
2. 没有配置 5 个主实验现代 baseline command, 或缺少完整的项目内 official bundle cache, 导致 official reference / formal comparison scoring / paper gate 阻断。
3. Colab 冷启动后没有重新 clone 仓库或安装依赖, 导致模块导入失败。
4. 直接运行 `paper_gate_and_package_colab.ipynb`, 但 runtime、formal comparison scoring、paper evidence postprocess 或 baseline artifacts 不存在。
5. Colab 断开前没有打包到 Google Drive, 导致本地临时结果丢失。
6. 手动修改最终 detection score 或正式 records, 导致 harness 审计失败。

## 8. 诊断 Notebook 的使用边界

以下 Notebook 只作为诊断或历史机制调试入口, 不属于当前 paper workflow 的正式顺序:

```text
paper_workflow/colab_notebooks/wan21_flow_adapter_preflight_colab.ipynb
paper_workflow/colab_notebooks/sampling_time_constraint_colab.ipynb
```

当 Wan2.1 无法加载、callback 捕获不到 latent、time grid 或 sampler signature 记录异常时, 可以先运行 `wan21_flow_adapter_preflight_colab.ipynb` 排查 adapter。排查通过后, 仍应回到第 3 节的主流程。


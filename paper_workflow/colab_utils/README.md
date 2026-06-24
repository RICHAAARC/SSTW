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

### 3.3 外部 baseline 正式评分

Notebook:

```text
paper_workflow/colab_utils/external_baseline_formal_scoring_colab.ipynb
```

用途:

- 读取统一 workflow profile。
- 执行 external baseline source intake。
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

## 6. 常见失败原因

1. 未先运行或未保留 `motion_calibration` artifact, 导致 motion threshold reuse 失败。
2. 没有配置 6 个现代 baseline command, 导致 external baseline preflight 或 paper gate 阻断。
3. Colab 冷启动后没有重新 clone 仓库或安装依赖, 导致模块导入失败。
4. 直接运行 `paper_gate_and_package_colab.ipynb`, 但 runtime 与 baseline artifacts 不存在。
5. Colab 断开前没有打包到 Google Drive, 导致本地临时结果丢失。
6. 手动修改最终 detection score 或正式 records, 导致 harness 审计失败。

## 7. 诊断 Notebook 的使用边界

以下 Notebook 只作为诊断或历史机制调试入口, 不属于当前 paper workflow 的正式顺序:

```text
paper_workflow/colab_utils/wan21_flow_adapter_preflight_colab.ipynb
paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb
```

当 Wan2.1 无法加载、callback 捕获不到 latent、time grid 或 sampler signature 记录异常时, 可以先运行 `wan21_flow_adapter_preflight_colab.ipynb` 排查 adapter。排查通过后, 仍应回到第 3 节的主流程。

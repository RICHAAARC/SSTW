# SSTW：状态空间同步 Flow Matching 轨迹水印

SSTW 是一个面向生成式视频水印论文实验的 governed research project。项目目标是在真实 Flow Matching 视频生成链路中验证状态空间同步轨迹水印机制, 并用可审计的 records、tables、figures、reports 和 manifests 支撑论文结论。

本仓库不是单纯的 Notebook 集合。Notebook 只作为 Colab 入口; `main/` 只保存最小方法包,
可脱离 Notebook 的阶段编排位于 `workflows/`, 普通 GPU 服务器由
`scripts/run_generative_video_server_workflow.py` 运行同一流程。

## 主干门禁

当前主干门禁为:

```text
protocol_governance
-> mechanism_validation
-> probe_paper
-> pilot_paper
-> full_paper
-> submission_freeze
```

其中 paper profile 的运行规模如下:

| profile | target FPR | 生成单元 | clean negative event | 作用 |
|---|---:|---:|---:|---|
| `probe_paper` | 0.1 | 10 prompt × 6 seed = 60 | 60 个独立视频 | FPR=0.1 的完整三层论文闭合。 |
| `pilot_paper` | 0.01 | 50 prompt × 12 seed = 600 | 600 个独立视频 | FPR=0.01 的完整三层论文闭合。 |
| `full_paper` | 0.001 | 200 prompt × 30 seed = 6000 | 6000 个独立视频 | FPR=0.001 的完整三层论文闭合与正式主结果。 |

三个 paper profile 只能在样本规模、统计功效和 target FPR 上不同。attack 协议、baseline 接口、公平校准、内部消融、图表构建和门禁逻辑必须保持同构。

## 推荐 Colab / Notebook 流程

Notebook 说明见:

```text
paper_workflow/colab_notebooks/README.md
```

从零开始的正式顺序为:

```text
motion_threshold_calibration_colab.ipynb  # 仅当 motion calibration artifact 缺失或需要重校准时运行
-> generative_video_generation_colab.ipynb
-> generative_video_quality_scoring_colab.ipynb
-> runtime_attack_colab.ipynb
-> runtime_detection_colab.ipynb
-> 5 个主实验 external baseline formal reference Notebook
-> formal_comparison_scoring_colab.ipynb
-> paper_evidence_postprocess_colab.ipynb
-> paper_gate_and_package_colab.ipynb
```

硬件与并行原则由 `paper_workflow/colab_notebooks/README.md` 中的 Notebook 矩阵统一定义。简要原则是:

1. Wan2.1 生成、motion threshold calibration、VidSig 和 VideoShield official reference 需要真实 GPU。
2. 质量评分、VideoSeal、REVMark、WAM-frame 在小规模下可能 CPU 可运行, 但正式运行建议使用 GPU。
3. runtime attack、formal comparison scoring 和 paper gate 以 CPU 为主; runtime detection 与 paper evidence postprocess 会执行 Wan replay / adaptive detector query, 正式运行必须使用 GPU。
4. 5 个 external baseline Notebook 在 `generative_video_generation_colab`、`runtime_attack_colab` 和 `runtime_detection_colab` 阶段包都完成后可以并行运行, 但每个 baseline 只打包自己的 official bundle。
5. `formal_comparison_scoring_colab.ipynb` 必须等待 5 个 baseline official reference 阶段包全部完成后再运行。

## 无 Notebook 的服务器运行入口

首次接入 GPU 时先运行最小机制验证。该 profile 使用 2 个 prompt × 2 个 seed,
对每组生成 full method、without velocity、endpoint-only 和 clean reference,
共 16 个视频；输出只证明机制执行路径可用, 不属于论文证据。结果根目录必须位于
仓库外:

```bash
export SSTW_MODEL_REVISION=<Wan模型的40位commit>
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW-runs \
  --workflow-profile method_mechanism_validation \
  --pipeline method_mechanism_validation \
  --model-revision "$SSTW_MODEL_REVISION"
```

运行前环境必须通过 `requirements/paper_runtime_environment_lock.json` 预检,
包括 Python 3.11、PyTorch 2.6、CUDA 12.4、算力不低于 7.0 和显存不低于 14 GiB。

在 Google Colab 上使用
`paper_workflow/colab_notebooks/method_mechanism_validation_colab.ipynb`。
该 Notebook 的第一个 cell 挂载 Google Drive，最后一个 cell 执行相同服务器
pipeline，并校验结果 zip 与 manifest 已落盘到
`MyDrive/SSTW/method_mechanism_validation/method_mechanism_validation_colab/`。

普通 GPU 服务器可直接使用命令行入口, 不依赖 Colab Notebook:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile probe_paper \
  --pipeline paper_protocol_complete
```

只检查计划、不运行重型任务:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile probe_paper \
  --pipeline paper_protocol_complete \
  --dry-run
```

更多命令见:

```text
scripts/README.md
```

## 目录职责

```text
main/                   可独立抽离的 SSTW 论文核心方法, 不包含 runner、攻击、统计或 CLI
runtime/                可复用运行时基础设施, 只向内依赖 main
evaluation/             攻击、指标、统计与结果协议
configs/                profile、protocol、baseline 和 workflow 配置
experiments/            实验 runner、paper protocol 和后处理逻辑
external_baseline/      external baseline source intake、official runtime 和 adapter
workflows/              可脱离 Notebook 的服务器阶段编排
paper_workflow/         最外层 Notebook / Colab 薄入口与专用挂载、鉴权 helper
scripts/                数据准备、结果检查、打包和服务器 workflow 入口
docs/                   工程治理、实验协议和复现说明
tools/harness/          可执行治理审计
tests/                  分层测试目录
.codex/                 Agent 协作契约与 skill 文件
audit_reports/          本地审计输出, 默认不提交
outputs/                本地运行输出, 默认不提交
```

## 必需检查

修改代码或治理文档后运行:

```bash
pytest -q
python tools/harness/run_all_audits.py
```

`pytest -q` 默认只运行轻量测试, 不应把真实 GPU 运行放入默认测试路径。
`paper_artifact_rebuild_package` 不携带 tests、`tools/harness/` 或 `paper_workflow/`。
必须先在开发仓库完成上述两项检查，再抽离服务器重建包；抽离包通过
`scripts/run_generative_video_server_workflow.py` 直接运行同一 repository workflow。

## 论文产物治理原则

1. records 是论文结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定到 governed artifacts。
4. external baseline 必须在项目内完成 clone / build / run / adapt / record, 不能依赖外部补交结果。
5. `metric_status: measured_formal` 只能由正式 workflow 转写生成; `non-run record` 只能记录阻断原因, 不能替代正式 baseline 结果。
6. Notebook 不得直接手写正式 records、thresholds、tables、figures 或 reports。

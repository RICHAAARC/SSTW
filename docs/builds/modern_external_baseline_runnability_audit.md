# modern external baseline 可运行性审计

本文档记录 `validation_scale` 阶段 6 个现代 external baseline 的当前可运行性判断。

## 1. 审计边界

本审计只判断当前仓库是否已经具备在 Colab 中通过项目内
`clone / build / run / adapt / record` 产出正式 baseline 结果的路径。

正式 baseline 结果必须满足:

```text
metric_status: measured_formal
external_baseline_score_status: measured_formal
```

`non-run record`、`unsupported`、`measured_proxy`、手工 JSON 和外部补交结果都不能替代正式结果。

## 2. 源码可获取性核验

已使用 `git ls-remote --heads` 核验 6 个官方仓库的配置 branch 仍可访问, 且当前 HEAD 与
`external_baseline/source_registry.json` 中登记 commit 一致。

这只说明官方源码入口可 clone, 不等价于 baseline 已能真实跑通。真实跑通还需要官方依赖、
checkpoint、key、message、训练权重、官方中间产物和项目 adapter 全部闭合。

## 3. 单项可运行性判断

| baseline | 当前 Colab 默认路径能否直接产出 `measured_formal` | 是否需要进一步适配 | 主要缺口 |
|---|---:|---:|---|
| VideoSeal | 有条件可以 | 否 | 需要 Colab 成功安装 VideoSeal 依赖并下载官方 checkpoint。 |
| VidSig | 有条件可以 | 否 | 已补齐项目内 `external_baseline.vidsig_official_runtime`, 默认运行官方 `generate_ms.py` 生成 VidSig clean / watermarked videos, 再施加项目 runtime attack 并调用官方 `attack.py`; 仍需要公开 checkpoint、Hugging Face 模型下载和可完成 Text-to-Video 的 GPU。 |
| VideoMark | 有条件可以 | 否 | 已补齐项目内运行官方 PRC key、embedding、extraction 和 temporal tamper 流程; 仍需要 Colab 成功下载模型并完成官方脚本。 |
| VideoShield | 不能直接完成 | 是 | 需要官方 generation / inversion / maintained info 流程, 不能只对 SSTW 普通视频后处理检测。 |
| SPDMark | 不能直接完成 | 是 | 需要训练出的 decoder / extractor 权重和 ground-truth bits。 |
| SIGMark | 有条件可以 | 否 | 已补齐项目内 `external_baseline.sigmark_official_hunyuan_runtime`, 可运行官方 Hunyuan `gen -> extract` 并转写 official bundle; 仍需要高显存 GPU 或可完成 HunyuanVideo 的 Colab 规格。 |

## 4. 当前适配器状态

每个 baseline 已具备两类项目内适配入口:

```text
external_baseline/primary/<baseline>/adapter/run_sstw_eval.py
external_baseline/official_eval_adapters/<baseline>.py
```

其中 `primary/<baseline>/adapter/run_sstw_eval.py` 是 SSTW 外层统一接口,
`official_eval_adapters/<baseline>.py` 是官方源码、官方 API、官方结果缓存或 native command
到 SSTW JSON 的桥接层。

这些 adapter 当前是 fail-closed 设计: 缺少官方源码、权重、key、message、checkpoint
或官方输出时必须失败, 不能输出 proxy 分数。

SIGMark 是一个特殊项: 其正式参考 Notebook 默认不再只读取预先存在的
`*-bit_accuracy.npz`, 而是通过
`external_baseline.sigmark_official_hunyuan_runtime` 在项目内复制官方源码到 runtime
工作副本, 构造与 SSTW runtime records 对齐的 prompt set, 运行官方
`main.py --mode=gen` 与 `main.py --mode=extract`, 再把官方 bit accuracy 输出写成
project-owned official bundle。该 bundle 仍不是最终论文记录, 后续必须由统一
external baseline runner 转写为 `metric_status: measured_formal`。

VidSig 也是特殊项: 它是生成过程中嵌入签名的方法, 因此 adapter 已收紧为 fail-closed。
正式结果不得来自“把 SSTW / Wan 视频直接送入 VidSig detector”。`vidsig_formal_reference_colab.ipynb`
默认调用 `external_baseline.vidsig_official_runtime`, 先运行官方 `generate_ms.py` 生成
VidSig 自己的 clean / watermarked videos, 再对 VidSig watermarked videos 应用同名
`video_compression_runtime`、`temporal_crop_runtime`、`frame_rate_resampling_runtime`,
最后调用官方 `attack.py` 写出 project-owned official bundle。

## 5. 结论

当前项目已经具备标准论文 baseline 对比的工程框架, 但 6 个 baseline 的正式结果尚未全部闭合。

下一步应按 baseline 逐个补齐:

1. 官方依赖安装和构建命令。
2. 官方权重、key、message 或生成中间产物。
3. `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND` 或等价项目内 official bundle 生成逻辑。
4. 覆盖全部 `validation_scale` runtime comparison units 的 score JSON。
5. 6 个 independent formal reference Notebook 先各自生成项目内 official bundle, 并默认调用统一 runner 转写当前可用的 `measured_formal` records。
6. `paper_gate_and_package_colab.ipynb` 在恢复 6 个 official reference 阶段包后执行全量统一转写、self-containment 判定和打包; `external_baseline_formal_scoring_colab.ipynb` 仅作为诊断或历史聚合入口。
7. `external_baseline_self_containment_decision.json` 通过。

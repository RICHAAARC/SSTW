# modern external baseline 可运行性审计

本文档记录 `probe_paper` 阶段 5 个主实验现代 external baseline 的当前可运行性判断。

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

已使用 `git ls-remote --heads` 核验 3 个官方仓库的配置 branch 仍可访问, 且当前 HEAD 与
`external_baseline/source_registry.json` 中登记 commit 一致。

这只说明官方源码入口可 clone, 不等价于 baseline 已能真实跑通。真实跑通还需要官方依赖、
checkpoint、key、message、训练权重、官方中间产物和项目 adapter 全部闭合。

## 3. 单项可运行性判断

| baseline | 当前 Colab 默认路径能否直接产出 `measured_formal` | 是否需要进一步适配 | 主要缺口 |
|---|---:|---:|---|
| VideoSeal | 有条件可以 | 否 | 需要 Colab 成功安装 VideoSeal 依赖并下载官方 checkpoint。 |
| VidSig | 有条件可以 | 否 | 已补齐项目内 `external_baseline.vidsig_official_runtime`, 默认运行官方 `generate_ms.py` 生成 VidSig clean / watermarked videos, 再施加项目 runtime attack 并调用官方 `attack.py`; 仍需要公开 checkpoint、Hugging Face 模型下载和可完成 Text-to-Video 的 GPU。 |
| VideoShield | 有条件可以 | 否 | 已补齐项目内 `external_baseline.videoshield_official_runtime`, 默认调用官方 VideoShield watermark、ModelScope 生成、latent inversion 与 temporal matching 流程; 仍需要 Colab 成功下载 Hugging Face 模型并完成 GPU 反演。 |

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

`*-bit_accuracy.npz`, 而是通过
工作副本, 构造与 SSTW runtime records 对齐的 prompt set, 运行官方
`main.py --mode=gen` 与 `main.py --mode=extract`, 再把官方 bit accuracy 输出写成
project-owned official bundle。该 bundle 仍不是最终论文记录, 后续必须由统一
external baseline runner 转写为 `metric_status: measured_formal`。

VidSig 也是特殊项: 它是生成过程中嵌入签名的方法, 因此 adapter 已收紧为 fail-closed。
正式结果不得来自“把 SSTW / Wan 视频直接送入 VidSig detector”。`vidsig_formal_reference_colab.ipynb`
默认调用 `external_baseline.vidsig_official_runtime`, 先运行官方 `generate_ms.py` 生成
VidSig 自己的 clean / watermarked videos, 再对 VidSig watermarked videos 应用同名
runtime attack。`probe_paper` 默认只要求 `video_compression_runtime`、
`temporal_crop_runtime`、`frame_rate_resampling_runtime`; `pilot_paper` 和
`full_paper` 必须从 protocol config 的 `required_runtime_attack_names` 读取扩展
attack 集合,
最后调用官方 `attack.py` 写出 project-owned official bundle。

VideoShield 同样是生成过程中嵌入水印的方法。`videoshield_formal_reference_colab.ipynb`
默认调用 `external_baseline.videoshield_official_runtime`, 以同一 prompt / seed 为锚点
运行官方 VideoShield watermark 与 ModelScope 生成路径, 再对 VideoShield 自己的
watermarked video 施加项目 runtime attack, 最后调用官方 latent inversion 与 temporal
matching 逻辑写出 project-owned official bundle。该 bundle 仍不是最终论文记录, 后续
必须由统一 external baseline runner 转写为 `metric_status: measured_formal`。

attack 的 `decode_acc` 转写为 project-owned official bundle。该实现必须保持
fail-closed: 若某个 required runtime attack 在官方 `temporal_results.json` 中没有
逐 attack 条目, 不允许退回到聚合均值。

## 5. 结论

当前项目已经具备标准论文 baseline 对比的工程框架。主实验必跑 baseline 已收敛为 3 个,


下一步应按 baseline 逐个补齐:

1. 官方依赖安装和构建命令。
2. 官方权重、key、message 或生成中间产物。
3. `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND` 或等价项目内 official bundle 生成逻辑。
4. 覆盖全部 `probe_paper` runtime comparison units 的 score JSON。
5. 5 个主实验 independent formal reference Notebook 先各自生成项目内 official bundle; `formal_comparison_scoring_colab.ipynb` 再统一转写 `measured_formal` records。
6. `formal_comparison_scoring_colab.ipynb` 在恢复 5 个主实验 official reference 阶段包后执行全量统一转写、self-containment 判定、公平校准和差值区间统计; `paper_evidence_postprocess_colab.ipynb` 恢复 formal comparison scoring 阶段包并生成辅助证据; `paper_gate_and_package_colab.ipynb` 恢复 runtime、motion threshold、formal comparison scoring 和 paper evidence postprocess 阶段包并执行最终门禁和打包。旧的通用 external baseline scoring Notebook 已删除, 不再作为诊断或正式入口保留。
7. `external_baseline_self_containment_decision.json` 通过。

# SSTW 历史 paper-profile 工程进度快照

> [!CAUTION]
> **Historical / legacy status — 不是当前 SSTW 方法状态基线。**
>
> 本文保留旧 paper-profile 工程状态，不能用于判断当前方法完成度或授权执行。当前
> 唯一权威入口：
>
> - `docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
> - `docs/builds/frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`
>
> 当前机制链是 `payload M → PRC drive u_n → low-dimensional state dynamics s_n
> → DiT Patch/3D-RoPE or relative-attention carrier → inference-time Flow velocity
> deflection → output-side Patch-relation observation → clock path + state observer
> → key-conditioned trajectory evidence`。真实 Gate 0 FAIL 只否定历史
> `decoder-Jacobian additive atom + local RGB mean feature + held-out transfer`
> 组合；Patch-relation 主机制尚未实现或运行。generic public low-frequency
> carrier bank 仅为待审 baseline / fallback。
> 以下正文中所有“当前”“正式”“基线”“候选”“下一轮”等表述都只描述本文
> 保存的历史快照，不代表现行 SSTW 状态、方法入口或执行授权。

本文档只记录该历史 paper-profile 工程快照，不保存已移除的更早门禁、proxy
postprocess 产物或旧 Notebook 编排。若需要考古历史提交，应使用 Git 历史；本
文档不能作为现行 SSTW 构建依据。

## 历史主干门禁

```text
protocol_governance
-> mechanism_validation
-> probe_paper
-> pilot_paper
-> full_paper
-> submission_freeze
```

## 历史正式 Notebook 顺序

```text
motion_threshold_calibration_colab.ipynb  # 仅在缺少 motion threshold artifact 时运行
-> generative_video_generation_colab.ipynb
-> generative_video_quality_scoring_colab.ipynb
-> runtime_attack_colab.ipynb
-> runtime_detection_colab.ipynb
-> 5 个主实验 modern external baseline formal reference Notebook
-> formal_comparison_scoring_colab.ipynb
-> paper_evidence_postprocess_colab.ipynb
-> paper_gate_and_package_colab.ipynb
```

## 历史正式证据来源

- SSTW 本方法 detection 由 `runtime_detection_colab` 产出正式 video-content detector records。
- external baseline 由各自 formal reference Notebook 在项目内完成 clone / build / run / adapt / record, 再由 `formal_comparison_scoring_colab` 统一转写为 `metric_status = measured_formal`。
- 公平比较由 `fair_detection_calibration` 在每个方法自己的 clean negative 分布上校准到相同 target FPR, 再报告 held-out attacked positive TPR。
- 内部消融由 formal internal ablation records 支撑, 不使用 proxy postprocess 记录。
- adaptive attack 只接受正式执行记录; 若缺少正式执行记录, 只能生成阻断原因, 不能替代论文结论。

## 非主干后处理边界

该历史快照中的仓库只保留当时的正式 paper profile 流程。非主干 proxy 后处理
入口、旧矩阵补造入口和旧小样本 claim gate 不属于该历史快照的工程能力，也不得
作为现行 SSTW 构建、门禁或论文结果依据。旧落盘文件如出现在 Google Drive 历史
结果包中，只能视为历史遗留输入，不能支持现行方法门禁。

## 历史完成判断口径

- `probe_paper`: `target_fpr = 0.1`, 用于小样本论文闭合验证, 产物结构必须与后续层级同构。
- `pilot_paper`: `target_fpr = 0.01`, 用于中等规模稳定性验证。
- `full_paper`: `target_fpr = 0.001`, 用于正式论文主结果。

三者的差异应限定为样本规模和目标 FPR 等级, 不应更换 detection、attack、baseline、公平校准、内部消融、统计和打包机制。

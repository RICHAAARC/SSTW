# Generative Video Model Probe

> 完整论文机制的实现顺序、三层 Claim 证据链和 fail-closed 规则见
> `docs/builds/complete_paper_mechanism_implementation.md`。三个 paper profile 使用同一机制,
> `probe_paper` 也必须在 FPR=0.1 下闭合 Claim-1、Claim-2 和不降级的 Claim-3。

本目录保存 generative_video_model_probe 生成式视频模型探测的可审计运行入口。当前无 GPU 时只生成 blocked decision, 不生成正向机制结论。

## Frame-state synchronized generative-Flow video watermark

SSTW 当前主路线把两条时间轴严格分开：

```text
video frame/window time
  → 定义低维水印状态、插帧/删帧/变速同步

generative Flow time
  → 只负责在冻结视频生成模型的采样过程中写入整条帧状态轨迹
```

方法设计和算法原语分别见：

- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`

当前已有该路线的
`configs/protocol/sstw_frame_state_signed_observability_construction.json` 和
本地核心合同。合同已唯一冻结公开单-array NPZ dictionary 的初始化/shape、真实
8-step sigma/`delta_sigma`/late-tapered waveform、scheduler-state signed exposure，
以及 clean-noise/T0 apply-only
Gate 公式；dictionary 只称固定8次迭代后时间加权的 Jacobian-aligned direction，
三层公共 signed gates 与 saved-video-only T0 gates 也有 exact applicability。
CPU/NumPy 层现已实现 atom、strict FP32 control、528D feature、C0 T0 与 A
apply-only Gate 0 原语，并冻结互异 C0/A execution identities。当前仍无 Wan
adapter、runner、observer 或 GPU 结果；construction/GPU/Colab 执行授权为 false，
不能由现有 server/Notebook 调用。旧
output-feature impulse observability、root-cause、spatiotemporal 和
frozen-feedback runner 均保留为 Flow-stage-indexed carrier/feature 的历史
construction/负对照，不得作为帧状态同步已通过的证据。下一实现必须从一个
视频时间窗口、一个 signed 状态维度的独立 execution identity 冻结与代码审核
完成并通过本地审计开始；不能直接实现时间攻击或完整 observer。下一独立任务
才允许在用户授权下实现最薄 GPU runner。

## Output-feature impulse observability

`output_feature_impulse_observability_construction.py` 是独立的首次14-video Gate A
construction runner：只生成单 identity signed impulses，记录实际 FP32 exposure，
经保存视频和冻结 Wan output-side VAE 构建五个 checkpoints，并调用审核过的
`A_actual`/Gate A evaluator。它不实现 replay、wrong-key、Gate B/C、observer、
attack、fixed-FPR、baseline、正式结果或阶段推进。

`gate_a_root_cause_amplitude_feedback_diagnostic.py` 保留该真实 Gate A FAIL，
并运行独立的六视频 `lambda=.06` 根因判别。它只比较 early0/late0 的一阶 odd、
二阶 common、后续 feedback 与 decode/output mismatch 候选；历史 `.12` 只读且
不重跑。所有分类允许多因或不确定，始终 `gate_a_pass=false`、非正式且禁止阶段推进。

`existing_six_video_spatiotemporal_signed_response_diagnostic.py` 不生成视频，
只安全解压并精确验证完整 f06a0934 六视频结果，在 CPU 上从真实保存的 RGB24
逐帧计算固定 video-time signed response。它只输出
`temporal_feature_salvage_candidate`、`carrier_redesign_required_candidate`
或不确定/多候选诊断；不接 server handler、Notebook、GPU、Drive，也不把诊断
写成 Gate A 或方法证据。

`frozen_feedback_signed_response_diagnostic.py` 是下一步独立五输出
construction-only 判别：显式消费并验证完整 f06a0934 normal-feedback FAIL，
只运行一条8-step clean Wan trajectory，并把其 CFG-combined base velocity
共享给正负 early0/late0 四条独立 scheduler replay。四条counterfactual均不再
调用 transformer。五项依次完成 full latent、decode、真实MP4 RGB24回读、
output-side VAE re-encode 和公开summary，再按冻结真值表记录feedback isolation、
decoder/carrier mismatch、停止当前additive carrier或indeterminate候选。该入口
不是Gate A重试，所有结果固定非正式、Gate A false且禁止阶段推进。

## Prompt-orthogonal state-trajectory smoke

`prompt_orthogonal_state_trajectory_smoke.py` 保留为已执行候选路线及历史 control。
它从已审核的
controlled embedding 结果与明确失败的 temporal-code isolation 结果构建固定
`2 prompts × 2 seeds × (watermarked + clean)` 计划；generation 使用8步，检测使用
20步 key-independent fixed trace 和两通道向量解调。该入口仅由 `colab_test`
白名单调用，结果始终 `formal_result=false` 且禁止阶段推进；真实 FAIL 不得被后续
候选路线覆盖或改写成成功结果。

## Minimal trajectory replay smoke

只执行已有4-source包的标准攻击与真实 VAE/replay 诊断:

```bash
python -m experiments.generative_video_model_probe.trajectory_replay_smoke \
  --package-path outputs/<method_mechanism_validation_package>.zip \
  --run-root outputs/trajectory_replay_smoke \
  --config-path configs/protocol/sstw_minimal_trajectory_paper.json
```

该入口不生成新视频、不运行 external baseline、不使用 test split，也不连接其他项目。
output feature construction 原语要求每视频记录在汇总前绑定冻结 probe ID、feature
schema digest 和数值行摘要；feature、A_actual 与14项 plan 必须完全同序，禁止
事后按矩阵位置补写 identity。
缺少锁定 GPU 运行时或 owner key 时会生成环境阻断型 `NO_GO` 报告，禁止代理 replay。

## 生成模型分工

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 是三层主张与固定 FPR 主表模型。
- `Lightricks/LTX-Video` 是参数规模较小的 Flow Matching 跨模型泛化模型。
- LTX 使用三维 packed token latent, 但在进入 SSTW 算法原语前必须通过可逆
  layout 转换为 `[B, C, T, H, W]`; endpoint、path 和 replay 均不得使用代理分数。
- 跨模型子集按 calibration/test 分层抽样, 只支撑模型泛化结论, 不替代当前
  profile 在 Wan 主模型上的固定 FPR 闭合结论。

正式运行会生成:

```text
artifacts/cross_model_generalization_decision.json
tables/cross_model_generalization_table.csv
```

# Generative Video Model Probe

本目录保存 generative_video_model_probe 生成式视频模型探测的可审计运行入口。当前无 GPU 时只生成 blocked decision, 不生成正向机制结论。

## 当前方法唯一入口

当前 SSTW 方法只以下列两份文档为权威入口：

- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`

机制链固定为：

```text
payload M
-> PRC drive u_n
-> low-dimensional state dynamics s_n
-> DiT Patch/3D-RoPE or relative-attention carrier
-> inference-time Flow velocity deflection
-> output-side Patch-relation observation
-> clock path + state observer
-> key-conditioned trajectory evidence
```

该 Patch-relation 主机制的首个 Gate 0 construction runner 已实现；真实 Colab
运行已在 Gate 统计前暴露并关闭 RoPE dtype 问题，随后又在首个 active step 的
phase projection 处 fail-closed，因此尚无 Gate 0 方法结果。后续 clock/state
observer 仍未实现。
真实 Gate 0 FAIL 只否定历史
`decoder-Jacobian additive atom + local RGB mean feature + held-out transfer`
组合；它不是 Patch-relation embedding、clock path、state observer 或
state-space synchronization 的失败结果。generic public low-frequency carrier
bank 仅是待审 baseline / fallback，不是当前主方法。

当前主路线的首个本地合同位于
`configs/protocol/sstw_patch_relation_gate0_construction.json`，配套
`main/methods/state_space_watermark/patch_relation_carrier.py` 与
`evaluation/protocol/patch_relation_gate0_contract.py`。它冻结公开 zero-sum
Patch relation、Wan temporal RoPE phase tuple、保存视频逐帧 Patch-pair DCT
relation feature、C0 `T_rel` 与 identity A apply-only Gate 0 统计。
`patch_relation_gate0_construction.py` 将这些原语接到真实 diffusers 0.35.2
Wan：精确执行 C0/A 各4项，每step对 cond/uncond 分支分别运行 zero-relation
base 与受控 relation，重算 scheduler 实际消费的 FP32 CFG velocity、预算、
真实返回next-state差、预算、state-update exposure 与冻结
timestep/internal-index输入绑定，再从精确输出目录中的MP4 RGB24回读重构
feature与输出绑定。
`colab_test` 只提供固定 Notebook 的 request-driven 薄路由；成功包在 `/content`
完整生成后才写单 ZIP+manifest 到 Drive，runtime失败保持 recovery-only。
当前状态只是 runner implemented / pending user Colab run，不得解释为端到端
方法结果或 Gate 0 已执行。

`patch_relation_phase_response_preflight.py` 是独立的非 Gate 单步入口。它固定
C0 clean-A 的真实 step 0 input/sample/timestep，执行两次 zero-phase base，并在
下一确定性低相位上对正负 control 各执行两次；记录实际 RoPE tuple delta、
base/control CFG repeat差、正负方向和既有 counterfactual state-update预算统计，
随后在调用真实 scheduler 前终止。它不decode、不导出视频、不选择新phase，也不
允许直接重跑8视频 Gate 0。

`wan_model_load_cache_preflight.py` 是更早的基础设施入口：父进程监控独立
model-load worker、Hugging Face 本地cache增长、worker RSS/CPU 与loader phase，
在2700秒总时限或cache文件数/字节连续600秒无变化时执行有界TERM/KILL/reap。
worker显式完成冻结commit的`snapshot_download`后，仅从冻结本地cache加载pipeline，
并在scheduler/offload/tiling闭合后释放模型；不执行forward、step、decode、
export或Gate。PASS必须重放精确8阶段start/finish有序ledger；phase/partial/lock/
RSS/CPU只报告，不能延长cache无进展deadline。`colab_test` 在任何runner前以create-only方式写本地
validation bootstrap；ZIP与manifest经同文件系统staging成对readback后才原子提升
到Drive。已发布结果即使bootstrap最终清理失败也不得被recovery重复打包。该
preflight不是方法结果。

当前唯一活动入口是`patch_relation_phase_response_preflight`：它是阶段1最小单步
phase-response preflight，不decode、不导出视频、不进入Gate。更早的
`wan_model_load_cache_preflight` 已作为阶段0已完成/paused historical 入口保留，
只维持解析、dry-run 与旧测试 double 兼容；不会作为真实 server 当前入口，也不会
自动触发完整 Gate 0。

历史 paper-profile 的实施顺序与 Claim 闭合规则仍保存在
`docs/builds/complete_paper_mechanism_implementation.md`，但该文档只用于历史
参考，不是当前方法入口或 Patch-relation 执行计划。

## 历史 decoder-Jacobian Gate 0 路线

下述实现保留为已执行历史路线和失败证据，不再作为当前方法基线。它使用
`configs/protocol/sstw_frame_state_signed_observability_construction.json` 和
本地核心合同，冻结公开单-array NPZ dictionary 的初始化/shape、真实
8-step sigma/`delta_sigma`/late-tapered waveform、scheduler-state signed exposure，
以及 clean-noise/T0 apply-only
Gate 公式；dictionary 只称固定8次迭代后时间加权的 Jacobian-aligned direction，
三层公共 signed gates 与 saved-video-only T0 gates 也有 exact applicability。
CPU/NumPy 层现已实现 atom、strict FP32 control、528D feature、C0 T0 与 A
apply-only Gate 0 原语，并冻结互异 C0/A execution identities。
`frame_state_signed_observability_construction.py` 将这些原语接到真实
diffusers 0.35.2 Wan scheduler：精确执行 C0/A 各4项、C0 clean-A 后构造一次
public atom（Jacobian decode 固定 untiled，完成后恢复 VAE tiling）、逐step重算
actual exposure，并由 C0 估计唯一 saved-video T0 后在 A
apply-only。入口曾加入既有薄 `colab_test` request；Notebook 不含科学逻辑且未改。
该路线真实 Colab Gate 0 已 FAIL：latent signed gate 通过，但 decoded/saved
feature 与 held-out transfer 未通过。该结论只停止这一
decoder-Jacobian additive atom 与 local RGB mean feature 的组合，结果非正式且
禁止 stage progression。旧
output-feature impulse observability、root-cause、spatiotemporal 和
frozen-feedback runner 均保留为 Flow-stage-indexed carrier/feature 的历史
construction/负对照，不得作为当前 Patch-relation 路线已通过或已失败的证据。

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
decoder/carrier mismatch、停止该历史 additive carrier 或 indeterminate 候选。该入口
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

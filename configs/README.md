# Configs

此目录保存论文实验配置模板。

## 当前方法配置边界

当前 SSTW 方法的唯一权威入口是：

- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`

当前机制链固定为：

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

这条 Patch-relation 路线尚无执行 config、runner 或真实运行结果。真实 Gate 0
FAIL 只否定历史
`decoder-Jacobian additive atom + local RGB mean feature + held-out transfer`
组合，不是 Patch-relation embedding、observer 或状态空间同步的结果。generic
public low-frequency carrier bank 仅为待审 baseline / fallback，不属于当前主方法。

`protocol/sstw_patch_relation_gate0_construction.json` 是当前主路线首个独立、
design/local-primitives-only 合同：冻结 diffusers 0.35.2 Wan
`transformer.rope` 输出后的单一 temporal 3D-RoPE phase-pair 写入点、一个公开
zero-sum Patch pair、保存视频 RGB24 的逐帧 Patch-pair DCT relation feature，
以及 C0 构造 `T_rel` / identity A apply-only 的最小 Gate 0 数学。它尚未实现
runtime adapter、runner、GPU/Colab/Notebook/Drive，所有正式结果、stage、
observer、攻击、fixed-FPR、baseline execution 与 paper claim 授权均为 false。
本地 NumPy 测试不构成 Patch-relation 方法证据。

下列配置说明保留历史实验和诊断边界；任何既有 execution flag 都不能授权或冒充
当前 Patch-relation 路线已进入执行。

`protocol/sstw_minimal_trajectory_paper.json` 是独立的 calibration-only replay
smoke profile。它不继承或修改 probe/pilot/full 的公共 top-tier 契约，只允许
已有4-source包、full/endpoint-only/clean、H.264/temporal crop 和单一20步 replay。
其 GO 只允许继续构建最小论文协议，不能支持 fixed-FPR 或 paper claim。

`protocol/sstw_prompt_orthogonal_state_trajectory_primitives.json` 与
`protocol/sstw_prompt_orthogonal_state_trajectory_smoke.json` 冻结 temporal-code
失败后的新候选路线。它只允许 8 条 no-attack generation、20-step replay、一个
owner 加八个 wrong-owner candidates 与 clean control；通过也只允许设计独立
calibration，不能支持攻击、fixed-FPR、阶段推进或论文 claim。

`protocol/sstw_output_feature_impulse_observability_construction.json` 冻结
Observer-Synchronized State-Space Trajectory Watermark 的上游 construction
合同：3个 Flow macro intervals、二维独立水印状态、六维阶段 block、候选密钥无关
的公开 output feature、14-video signed impulse triage 和 Gate A/B/C 授权状态机。
它同时绑定真实 Wan 8-step sigma/phase/delta-sigma waveform、prompt/seed 无关的
CPU canonical basis KDF、现有 Wan VAE streaming output extractor、五个 primary
checkpoint 与可选 replay diagnostic，以及有限 noise floor 的 Gate A 精确公式。
output feature 还必须由每视频 governed record 提供冻结 probe ID 与行绑定摘要；
14行、A_actual 和 plan 的身份与顺序必须精确一致，禁止事后按矩阵位置贴标签。
该配置曾允许由既有薄 Notebook 调用独立 Gate A construction handler；真实
14-video 运行已得到 FAIL 并作为历史 control 保留。config 中的 execution flag
不构成重跑授权，本地 tests 也不能据此写成方法有效或阶段推进。

该历史 config 使用 Flow early/middle/late 作为 carrier 设计轴，不是当前主路线
“视频帧状态同步＋生成 Flow 嵌入”的执行配置。当前主路线只在
`docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
及配套 algorithm primitives 中形成设计合同；尚未新增 config、runner、Notebook
handler 或 GPU 授权。未来不得复用本 config 的 Gate A 名称或 execution flag
冒充帧状态路线已进入执行阶段。

`protocol/sstw_frame_state_signed_observability_construction.json` 是已执行并失败的
历史 frame-state Gate 0 本地原语与 execution 合同：冻结非递归 `protocol_contract`
摘要、public context schema、单个中心视频窗口、一个公开 decoder-Jacobian
carrier atom、时间保持的保存视频 RGB24 feature、`C0/A` 两个隔离 identity 和
精确 `4+4` probe 计划。公开 NPZ 的单一 array/shape、固定 SHA-256 Rademacher
初始化与受限-support 8轮 Jacobian power iteration 后时间加权方向（不声称精确
leading singular）、Jacobian 期间关闭 spatial tiling 且随后恢复的确定 decode
边界、scheduler `delta_sigma` state-update
signed exposure、有限 clean floor 和 `T0` apply-only prediction 公式均在
`protocol_contract` 内冻结。latent/decoded/saved 三个 checkpoint 的 exact
representation 与公共 signed gate matrix 也已固定；唯一 T0 transfer gates 只
适用于 saved-video 528D primary。C0/A 的不同 prompt/seed、相同初噪声规则、
模型/scheduler/exporter，以及真实8-step sigma/delta-sigma 和独立 late-tapered
Flow waveform 已精确冻结；本地公开 atom、NumPy FP32 control/exposure、feature、
T0 与 gate 原语已实现。最薄 Wan runner 与既有 `colab_test` request handler 曾
用于真实8-video运行；该运行的 latent signed gate 通过，但 decoded/saved feature
与 held-out transfer 未通过，因此本组合已经 Gate 0 FAIL。其 execution flags
不构成重跑或当前方法授权；Notebook 无逻辑变更，observer、
攻击、fixed-FPR、baseline、paper claim、formal result 与 stage progression
仍全部为 false。本 config 和本地测试也不能作为当前 Patch-relation 方法结果。

`protocol/sstw_gate_a_root_cause_amplitude_feedback_diagnostic.json` 冻结首次
Gate A FAIL 后的独立六视频根因判别：只运行一个预声明 `lambda=.06` 半幅，
以完整 commit `47485be2...` Gate A FAIL 包作为只读 `.12` 基线，比较 early0/late0
在 latent、decode、save、re-encode 与 output feature 的 odd/common scaling。
它不重跑 `.12`，不重试 Gate A，也不允许跨 identity、observer、攻击或阶段推进。

`protocol/sstw_existing_six_video_spatiotemporal_signed_response_diagnostic.json`
冻结对上述完整六视频结果的 CPU-only、只读分析：逐帧完整 RGB24、固定三等长
video-time 区间和固定区间内相邻帧差分都使用同一 clean-centered odd/common
公式与既有 signed gate。source commit、完整文件快照、六项顺序与视频摘要均精确
绑定；单帧偶然通过不能形成候选，分析不得自动选择 feature、重试 Gate A、执行
frozen-feedback、更新 Drive 或推进阶段。

`protocol/sstw_frozen_feedback_signed_response_diagnostic.json` 冻结五输出
feedback-isolation construction：完整 f06a0934 normal-feedback FAIL 是显式
只读source；一次clean 8-step denoising产生共享base-velocity trace，四条正负
early0/late0 counterfactual不再调用模型。配置精确绑定16次CFG component
forward、32条branch update、`lambda=.06`、实际FP32预算与五层response gate。
任意结果都保持Gate A/formal/stage false，只能记录预声明的非唯一根因候选。

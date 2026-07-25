# Configs

此目录保存论文实验配置模板。

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

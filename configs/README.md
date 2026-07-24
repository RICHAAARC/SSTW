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
该配置现在允许由既有薄 Notebook 调用独立 Gate A construction handler；
`impulse_triage_execution_allowed=true` 只表示入口已获授权并等待用户 Colab
运行。本地 tests 不能据此写成 Gate A 已执行、方法有效或阶段推进。

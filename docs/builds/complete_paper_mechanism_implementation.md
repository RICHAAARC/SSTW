# SSTW 完整论文机制实施说明

## 1. 机制边界

`probe_paper`、`pilot_paper` 和 `full_paper` 绑定同一份
`configs/protocol/sstw_complete_paper_mechanism.json`。核心方法默认参数单独保存在
`configs/methods/sstw_core_method.json`。该文件不包含 profile、claim、attack、baseline
或消融语义；论文变体组合与证据要求只属于 `configs/protocol/` 契约，从而保证
`main/` 与最小方法包可以独立抽离。

三者执行相同的生成约束、检测器、replay、攻击协议、内部消融和结果门禁，只允许以下项目不同:

1. 目标 FPR;
2. prompt、seed、clean negative 与 attack event 数量;
3. 由样本数量和目标 FPR 产生的统计功效。

`probe_paper` 的目标 FPR 为0.1。它不是机制探针，而是完整三层主张在较小
样本规模下的闭合论文结论。三个正式 profile 只存在完整 Claim-3 路径。

## 2. 三层主张与真实数据流

### 2.1 Claim-1: 速度场弱约束形成可检测水印

生成阶段使用 `FlowVelocityConstraintRuntime` 包装 Wan 的 Flow scheduler。包装器修改的是
Transformer 经 classifier-free guidance 后、进入 `scheduler.step` 之前的 `model_output`，
不是 callback 后置 latent。每个 step 同时记录约束前后范数、key 方向投影、真实状态更新和
路径观测。

攻击后视频由 Wan VAE 重新编码到 endpoint latent。检测器使用与生成阶段相同的 tubelet
key code 计算 endpoint 投影和 payload bit accuracy。阈值只从 calibration negative 冻结，
held-out test 不允许更新阈值。

速度约束的因果归因使用同模型、prompt、seed、scheduler 和 time grid 的成对生成。完整方法与
`without_velocity_constraint` 在 pipeline 调用前必须具有相同的 GPU generator state digest,
且两个源视频内容摘要必须不同。两侧统一应用 `sstw_full_method` 的同一个冻结状态空间检测器,
因此配对差值只对应速度约束干预, 不允许通过为对照重新拟合检测器制造不可比概率。所有
held-out full-method 视频必须达到100%配对覆盖, 任一配对设计不一致都会阻断 Claim-1。

### 2.2 Claim-2: 路径证据在固定 FPR 下提供独立增益

`paired_path_evidence_gain_records.jsonl` 在同一个 full-method attacked video evidence 上同时
应用完整检测器和 `without_path_evidence` 检测器。后者只把逐 phase 的 path score 与
path-endpoint consistency 置零，保留相同视频、固定 reverse replay、endpoint、velocity、
replay likelihood、replay reliability、time grid、coverage、状态空间模型类型和 admissibility
机制。两个检测器都只从同一批 full-method calibration records 拟合各自的嵌套特征模型并冻结
阈值，因此比较不会混入生成机制变化、不同测试视频、不同攻击或测试集调参。

`endpoint_only_control` 仍作为“本方法不是 endpoint-only 水印”的生成级机制对照，但它同时
改变速度约束、replay 和状态观测，不能用于 Claim-2 的独立路径因果归因。

Claim-2 以同视频配对的检测判定差作为主统计量。只有固定 FPR 下配对检测增益的
95% 区间下界大于0，且原始分数增益区间下界也大于0时才通过。配对必须覆盖全部
held-out full-method positives，任何非路径特征变化、阈值来源错误或配对缺失都会阻断 Claim-2。
`without_path_evidence` 是检测器级嵌套消融，直接复用 full-method 视频和 calibration records，
不会增加视频生成量。其余8个生成级内部消融仍按各 profile 的预注册独立 source 数量执行。

### 2.3 Claim-3: 攻击后视频恢复可靠 replay 后验

`formal_flow_evidence_runner` 对攻击后视频执行以下步骤:

1. 使用 Wan VAE 恢复 endpoint latent;
2. 使用原 prompt、Wan Transformer 和 CFG 计算真实 base velocity, 再按候选 key 复现生成阶段的弱约束 velocity;
3. 只用不含候选 key 的 base velocity 在16、20、24步网格上恢复固定 reverse inversion 路径;
4. 从同一个固定初态分别执行 null forward 与候选 key forward hypothesis, 在仅由模型特定 calibration clean-video null residual 拟合并冻结的 endpoint-relative 各向同性高斯观测模型下计算真实 replay log-likelihood ratio;
5. 路径分数只投影固定 reverse states, 候选 key 不参与观测路径构造;
6. 由候选残差相对预注册观测方差的标准化量、ensemble endpoint 方差和 likelihood-ratio 离散度计算 replay 可靠性，并把全局多网格可靠性与逐 step 高斯拟合可靠性的乘积直接用于衰减 path step 观测和路径积分;
7. 执行 wrong key、wrong prompt 和改变 scheduler shift 的 wrong sampler/time-grid 对照;
8. 将逐 Flow phase 观测序列分别送入 calibration split 拟合的 H0/H1 线性高斯状态空间模型, 执行 Kalman filtering 与 RTS smoothing, 再计算两假设的边际对数似然比;
9. 状态转移、噪声统计和 admissibility 阈值均按 source-video group 等权拟合; 仅在按 H0/H1 分层且严格组外的状态空间似然比上拟合 Platt 概率校准, 不允许用全量模型回填 OOF 分数, test split 不更新状态模型、校准器或 fixed-FPR 阈值;
10. 验证生成阶段 HMAC-SHA256 trajectory sketch 的签名及模型、prompt、seed、sampler 和时间网格上下文。

只有真实 attacked-video replay、所有 HMAC 验证和三类错误条件对照达到门禁要求时，
`claim3_full_support_allowed` 才能为 `true`。

正式 H0 分布由四个有物理语义的 negative family 构成:
`clean_unwatermarked_candidate_key_hypothesis`、
`watermarked_video_wrong_key_hypothesis`、
`watermarked_video_wrong_prompt_hypothesis` 和
`watermarked_video_wrong_sampler_time_grid_hypothesis`。后三类直接复用正确条件下的固定
reverse states, 只改变 forward hypothesis。禁止按 trial 索引把同一种 clean negative
重命名为多个 family。calibration threshold 和 held-out FPR 均先取每个 source-video
cluster 内所有负假设的最大分数, 再把视频作为唯一独立单位。

## 3. 实施顺序

正式运行必须按以下顺序执行:

1. 准备 prompt/seed suite;
2. 从 Google Drive 私有文件
   `/content/drive/MyDrive/SSTW/.sstw_private/trajectory_authentication.json`
   自动加载 `SSTW_TRAJECTORY_AUTHENTICATION_KEY` 和
   `SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID`。Windows 本地映射为
   `G:\我的云端硬盘\SSTW\.sstw_private\trajectory_authentication.json`;
3. 执行 paper profile 生成;
4. 执行质量、运动与语义指标;
5. 执行 runtime attacks;
6. 执行 runtime detection Notebook。paper profile 下，该入口会自动分派到
   `formal_flow_evidence_runner`, 不再使用早期低频视频投影检测器;
7. 对每个独立 held-out 视频真实生成 adaptive 候选文件并查询冻结 Flow 检测器。
   endpoint-preserving、removal、probing、evasion 和 recompression 均保存完整 query log;
   copy/spoof 与 collusion 使用不同输入视频生成新的跨视频融合文件。copy/spoof 按 clean
   recipient 计数, collusion 按互不重叠的视频对计数。候选质量必须在写盘后
   重新解码的视频上计算, 不能使用编码前帧冒充 codec 后质量; Wan 与 LTX 分别读取各自的
   状态空间后验与 fixed-FPR 阈值, 不允许跨模型共用 calibration artifact;
8. 生成 SSTW measured-formal、内部消融、外部 baseline 与统计记录;
9. 执行 replay/sketch gate;
10. 执行 paper profile gate 和 profile transition gate。

runtime detection 的核心命令为:

```powershell
python -m experiments.generative_video_model_probe.formal_flow_evidence_runner `
  --run-root <run_root> `
  --config-path configs/protocol/probe_paper_generative_probe.json `
  --prompt-suite-path <prompt_seed_suite.json>
```

## 4. 关键 governed artifacts

| 产物 | 作用 |
| --- | --- |
| `records/trajectory_trace.jsonl` | 真实 scheduler model output 与状态更新证据。 |
| `records/trajectory_sketch_records.jsonl` | 不包含完整 latent 的 HMAC 认证路径摘要。 |
| `records/formal_flow_evidence_records.jsonl` | endpoint、path、replay、controls 和 posterior 的统一记录。 |
| `thresholds/formal_flow_detector_thresholds.jsonl` | 仅由 calibration split 拟合的 H0/H1 状态模型、组外 Platt 校准参数和 calibration negative 冻结的 fixed-FPR 阈值。 |
| `thresholds/replay_gaussian_likelihood_calibrations.jsonl` | 按生成模型隔离、由 calibration clean-video 簇等权拟合的 replay 高斯噪声参数; 噪声拟合只运行预注册20步主网格以避免长期缓存状态或重复三网格计算, held-out test 与 adaptive attack 只读复用。 |
| `records/paired_path_evidence_gain_records.jsonl` | 同视频、同攻击、同目标 FPR 的 Claim-2 配对证据。 |
| `records/paired_velocity_causal_evidence_records.jsonl` | 完整方法与无速度约束视频的 Claim-1 因果配对证据。 |
| `records/formal_adaptive_attack_query_records.jsonl` | 逐视频 adaptive 候选、质量约束与冻结检测查询日志。 |
| `artifacts/three_layer_mechanism_evidence_decision.json` | Claim-1 与 Claim-2 的前置结论, Claim-3 等待认证门禁。 |
| `artifacts/replay_and_sketch_gate_decision.json` | Claim-3 完整支持的唯一 replay/sketch 门禁。 |
| `artifacts/complete_paper_mechanism_claim_decision.json` | Claim-1、Claim-2、Claim-3 全部 PASS 后的统一闭合结论。 |
| `artifacts/paper_profile_gate_decision.json` | 当前 paper profile 的最终闭合门禁。 |

## 5. Fail-closed 规则

以下任一情况都会阻断 paper profile:

1. scheduler 不是 FlowMatch scheduler;
2. attacked video 无法由 Wan VAE 编码;
3. 缺少 calibration/test split 或任一正式消融变体;
4. held-out empirical FPR 超过 profile 目标;
5. Claim-2 配对增益区间不支持正增益;
6. replay 只来自 owner-side trace diagnostic;
7. 缺少 HMAC key、签名不匹配或认证上下文不一致;
8. wrong key、wrong prompt 或 wrong sampler 对照不能稳定区分;
9. 缺少完整 Claim-3 的任一正式证据。
10. 状态观测不是来自固定 replay 路径、序列少于2步、未完成 Kalman filtering / RTS smoothing, 或 replay 分数不来自预注册高斯似然模型。

这些规则的主要目的不是保证实验必然得到正结果，而是保证任何 PASS 都只能来自真实运行
证据。若实验结果不支持某一主张，项目必须保留 FAIL，而不能通过代理分数或标签生成记录。

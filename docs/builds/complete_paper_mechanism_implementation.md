# SSTW 完整论文机制实施说明

## 1. 机制边界

`probe_paper`、`pilot_paper` 和 `full_paper` 绑定同一份
`configs/methods/sstw_complete_paper_mechanism.json`。三者执行相同的生成约束、检测器、
replay、攻击协议、内部消融和结果门禁，只允许以下项目不同:

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

### 2.2 Claim-2: 路径证据在固定 FPR 下提供独立增益

`paired_path_evidence_gain_records.jsonl` 在同一个 full-method attacked video evidence 上同时
应用完整检测器和 endpoint-only 检测器。二者分别使用 calibration negative 冻结的阈值，
因此比较不会混入不同测试视频、不同攻击或测试集调参。

Claim-2 以同视频配对的检测判定差作为主统计量。只有固定 FPR 下配对检测增益的
95% 区间下界大于0时才通过; 原始分数增益区间作为诊断量保留。内部消融
生成在 calibration/test 各使用10/50/500个独立 source 展开全部8个变体，不计入主结果的
60/600/6000个生成单元。

### 2.3 Claim-3: 攻击后视频恢复可靠 replay 后验

`formal_flow_evidence_runner` 对攻击后视频执行以下步骤:

1. 使用 Wan VAE 恢复 endpoint latent;
2. 使用原 prompt、Wan Transformer 和 CFG 计算真实 base velocity, 再按候选 key 复现生成阶段的弱约束 velocity;
3. 只用不含候选 key 的 base velocity 在16、20、24步网格上恢复固定 reverse inversion 路径;
4. 从同一个固定初态分别执行 null forward 与候选 key forward hypothesis, 计算 replay log-likelihood ratio;
5. 路径分数只投影固定 reverse states, 候选 key 不参与观测路径构造;
6. 由候选循环误差、null 循环误差、ensemble endpoint 方差和 likelihood-ratio 离散度计算 replay 可靠性;
7. 执行 wrong key、wrong prompt 和改变 scheduler shift 的 wrong sampler/time-grid 对照;
8. 将上述观测送入按 source-video group cross-fitting 的 logistic + Platt 概率后验;
9. 验证生成阶段 HMAC-SHA256 trajectory sketch 的签名及模型、prompt、seed、sampler 和时间网格上下文。

只有真实 attacked-video replay、所有 HMAC 验证和三类错误条件对照达到门禁要求时，
`claim3_full_support_allowed` 才能为 `true`。

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
   copy/spoof 与 collusion 使用不同输入视频生成新的跨视频融合文件;
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
| `thresholds/formal_flow_detector_thresholds.jsonl` | 仅由 calibration negative 冻结的标准化和 fixed-FPR 阈值。 |
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

这些规则的主要目的不是保证实验必然得到正结果，而是保证任何 PASS 都只能来自真实运行
证据。若实验结果不支持某一主张，项目必须保留 FAIL，而不能通过代理分数或标签生成记录。

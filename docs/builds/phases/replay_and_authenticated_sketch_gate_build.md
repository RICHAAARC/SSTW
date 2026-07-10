# Replay 与认证轨迹 sketch 正式门禁

## 1. 文件级职责

本阶段负责验证 Claim-3：攻击后视频能否恢复可靠的 Flow replay 概率后验，并由认证轨迹 sketch 约束正确 key、prompt、sampler 与 time grid。该门禁是 `probe_paper`、`pilot_paper` 和 `full_paper` 的共同必选机制，不存在 Claim-3 降级通道。

三个 paper profile 使用完全相同的机制、记录类型、控制变量与通过条件。它们只在目标 FPR、独立视频数量和由此产生的统计置信度上不同。

## 2. 固定路径 replay 假设检验

真实 replay 必须按以下顺序执行：

1. 从攻击后视频编码 endpoint latent。
2. 仅使用不含候选 key 的 base velocity 构造固定 reverse inversion 路径。
3. 从同一固定初态分别运行 null forward 与候选 key forward hypothesis。
4. 计算候选循环误差、null 循环误差与 replay log-likelihood ratio。
5. 路径观测只读取固定 reverse states，候选 key 不得改变观测路径。
6. wrong key、wrong prompt、wrong sampler 与 wrong time grid 必须复用同一固定 reverse 路径，避免循环构造证据。

此处设计的主要考虑在于：候选 key 只能改变待检验的 forward hypothesis，不能参与生成检验所使用的 reverse observation。该结构属于可复用的假设检验写法，可迁移到其他基于 inversion 的生成模型认证任务。

## 3. 可靠概率后验

正式后验必须由冻结 calibration split 拟合，并在 held-out test split 上只推理、不更新。至少需要输出：

```text
watermark_posterior_probability
watermark_posterior_log_odds
posterior_entropy
replay_log_likelihood_ratio
replay_reliability
posterior_calibration_brier
posterior_calibration_log_loss
posterior_calibration_ece
```

`watermark_posterior_probability` 必须具有明确的参考先验和校准来源，不能把未经校准的相似度分数改名为 posterior。source video 是唯一统计簇；同一视频上的多 key、多攻击和多 replay control 只能作为簇内重复观测。

## 4. 认证 sketch 与控制组

trajectory sketch 必须使用配置外提供的 HMAC 密钥生成和验证。正式门禁至少要求：

```text
authenticated_trajectory_sketch_status == ready
trajectory_sketch_verification_status == pass
wrong_key_replay_records_ready == true
wrong_prompt_replay_records_ready == true
wrong_sampler_replay_records_ready == true
wrong_time_grid_replay_records_ready == true
replay_negative_fpr_controlled == true
calibrated_probability_posterior_ready == true
test_time_threshold_update_blocked == true
```

未认证 logging 只能作为调试记录，不能支持 Claim-3。错误控制组必须真实运行对应候选假设，不能通过复制正确 replay 分数、修改标签或使用手工常数构造。

## 5. 必须输出的 governed artifacts

```text
records/trajectory_sketch_verification_records.jsonl
records/replay_uncertainty_records.jsonl
records/wrong_key_replay_records.jsonl
records/wrong_sampler_replay_records.jsonl
records/wrong_prompt_replay_records.jsonl
records/wrong_time_grid_replay_records.jsonl
tables/replay_verification_table.csv
artifacts/replay_and_sketch_gate_decision.json
reports/replay_and_sketch_gate_report.md
```

所有 supported claim 必须映射到上述 records、decision、table 或 report。若任一必需控制、校准证据或认证验证缺失，当前 paper profile 必须判定为 `FAIL`。

## 6. 当前真实实现

正式实现入口包括：

```text
main/methods/state_space_watermark/replay_inversion.py
main/methods/state_space_watermark/wan_flow_replay_backend.py
main/methods/state_space_watermark/flow_state_posterior.py
main/methods/state_space_watermark/formal_detector.py
experiments/generative_video_model_probe/formal_flow_evidence_runner.py
experiments/generative_video_model_probe/replay_and_sketch_gate.py
```

`main/` 只保存方法原语；真实 Wan replay、记录生成和 paper gate 位于 `experiments/`。Notebook 仅调用 `workflows/`，不会在单元格中实现 replay、后验校准或门禁逻辑。

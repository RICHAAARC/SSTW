# Prompt-Orthogonal State-Trajectory Watermark 算法原语

## 0. 文档状态

本文档把候选方法拆成可单独测试、可证伪的算法原语。2026-07-24 已完成
P0--P9、本地功能测试、真实 Wan 调用路径和 Colab 白名单接线；Notebook 未修改，
GPU 尚未执行。这表示 `gpu_mechanism_smoke_ready_not_executed`，不表示 mechanism
capability 或 supported claim。

本文档中的 proposed module、record concept 和 config concept 只是实现规划。正式
Python 模块、JSON 字段或 thresholds 在落地前必须：

1. 登记到 `docs/field_registry.md`；
2. 进入 protocol config；
3. 增加 mutation rejection tests；
4. 通过默认 pytest 与 harness；
5. 获得独立 GPU 运行授权。

## 1. 原语总览

| ID | 原语 | 核心职责 | 当前状态 |
|---|---|---|---|
| P0 | Key Domain Separation | 从 master key 派生 prompt-independent 子 key | core implemented |
| P1 | Low-Rank State Rotation Operator | 生成状态依赖、非固定模板的 keyed tangent | core implemented |
| P2 | Prompt-Nuisance Orthogonal Projection | 消除当前 state/velocity 公共方向 | core implemented |
| P3 | Continuous Balanced Trajectory Code | 跨 generation/replay grid 重建连续零均值码 | core implemented |
| P4 | Budgeted State-Trajectory Injection | 在 velocity hook 中注入受限 trajectory innovation | core implemented |
| P5 | Key-Independent Replay Trace | 对所有候选共享一次 inversion/replay | Wan shared-trace core implemented |
| P6 | Replay Innovation Sequence | 提取观测转移减 clean model transition 的残差 | Euler adapter core implemented |
| P7 | Synchronous Vector Demodulation | 保留多通道时间响应并完成候选匹配 | core implemented |
| P8 | Candidate Identity Decision | 冻结 coverage、prompt gap、clean 与 owner/wrong gate | implemented, not GPU-executed |
| P9 | Governed Evidence Packaging | 保存逐步向量、候选与 provenance，不写正式 claim | implemented, not GPU-executed |

## 2. P0：Key Domain Separation

### 2.1 输入与输出

输入：

```text
master_key
method_domain_label
latent_layout_id
state_operator_schema_id
trajectory_code_schema_id
```

输出：

```text
state_operator_subkey
trajectory_code_subkey
domain_separation_digest
```

### 2.2 不变量

- 相同 master key 与相同方法/latent schema domain 必须得到相同子 key。
- generation step count 与 replay step count 不得改变子 key。
- prompt、prompt digest、seed 和当前 latent 内容不得进入 KDF。
- 精确 model ID、revision/hash 和 sampler implementation version 不得进入 KDF。
- 模型、revision 与 sampler 必须作为 provenance/compatibility context 单独记录。
- state/operator 与 trajectory/code 必须使用不同 domain label。
- 任何子 key 不得进入 records、logs 或 manifests。
- records 只允许保存不可逆 digest。

### 2.3 失败语义

以下情况 fail-closed：

- latent layout 或 operator/code schema 缺失；
- prompt 或 seed 被观察到进入 KDF payload；
- domain labels 冲突；
- digest 不稳定。

### 2.4 轻量测试

- prompt mutation 不改变子 key；
- seed mutation 不改变子 key；
- model revision mutation 不改变子 key；
- latent layout/schema mutation 改变子 key；
- state/operator 与 trajectory/code 子 key 不相等；
- generation 8-step 与 replay 20/40-step 共享子 key。

## 3. P1：Low-Rank State Rotation Operator

### 3.1 数学定义

由 `state_operator_subkey` 生成两个单位向量 \(a_K,b_K\)，并执行 Gram-Schmidt：

\[
\|a_K\|_2=\|b_K\|_2=1,\qquad
\langle a_K,b_K\rangle=0.
\]

定义：

\[
A_K=a_Kb_K^\top-b_Ka_K^\top.
\]

对 state \(z\)：

\[
q_K(z)=A_Kz
=a_K\langle b_K,z\rangle-b_K\langle a_K,z\rangle.
\]

### 3.2 输出

```text
state_tangent
operator_plane_digest
operator_rank
state_tangent_norm
state_tangent_orthogonality_residual
```

以上名称是 record concepts，不是已登记字段。

### 3.3 不变量

- 不物化完整高维矩阵；
- operator rank 固定为 2；
- \(q_K(z)\) 与 \(z\) 数值正交；
- 不同 key 的 plane collision 低于预声明 tolerance；
- shape、dtype、device 与输入 state 一致；
- batch 中每个样本独立计算，不跨样本混合；
- operator 不依赖 prompt。

### 3.4 失败语义

- key vectors 线性相关；
- state tangent norm 不足；
- orthogonality residual 超限；
- non-finite；
- candidate key plane collision。

失败 step 不得回退到固定 PRF direction。

### 3.5 轻量测试

- NumPy tensor substitute 验证精确低秩公式；
- torch CPU 验证 shape/device/dtype；
- dot\((z,q_K(z))\) 在 tolerance 内为零；
- wrong key 改变 plane digest 与 tangent；
- prompt mutation 不改变 operator plane。

## 4. P2：Prompt-Nuisance Orthogonal Projection

### 4.1 输入

```text
current_state
base_model_velocity
state_tangent
projection_tolerance
```

### 4.2 最薄 nuisance basis

首轮只使用：

\[
\mathcal{N}=\operatorname{span}\{z,v_\theta(z,t,p)\}.
\]

通过稳定 Gram-Schmidt 或 QR 获得 basis \(Q_N\)，并计算：

\[
\bar q=(I-Q_NQ_N^\top)q_K(z).
\]

归一化得到：

\[
u_K(z,t,p)=\bar q/(\|\bar q\|_2+\epsilon).
\]

### 4.3 不变量

- nuisance basis 对候选 key 完全相同；
- state 和 base velocity 不得被 in-place 修改；
- 投影后方向同时与 state、base velocity 正交；
- projection retained ratio 必须被记录；
- retained ratio 低于冻结下限时 step inactive；
- inactive 不得伪称 guard passed。

### 4.4 不允许的扩展

首轮禁止：

- 从当前测试 prompts 拟合 PCA nuisance basis；
- 用 correct/wrong 得分选择 projection rank；
- 使用 prompt-specific sign correction；
- 使用 test-time learned whitening 替代几何投影；
- 投影失败时回退到未投影 key direction。

### 4.5 轻量测试

- state/velocity 正交残差；
- velocity 与 state 共线时 basis 仍稳定；
- tangent 完全落在 nuisance subspace 时 inactive；
- correct/wrong candidate 共享相同 nuisance basis digest；
- non-finite 输入 fail-closed。

## 5. P3：Continuous Balanced Trajectory Code

### 5.1 输入与输出

输入：

```text
trajectory_code_subkey
continuous_flow_phases
active_scheduler_weights
frozen_basis_family
frozen_code_dimension
```

输出：

```text
continuous_code_values
master_codeword_digest
continuous_function_digest
schedule_projection_digest
weighted_mean_residual
weighted_code_energy
minimum_active_code_magnitude
```

### 5.2 基函数

候选基函数必须连续、预声明且跨 grid 可重建。首轮可从以下固定 family 中选择一个，
但选择必须发生在 GPU 结果之前：

\[
\phi_{2j-1}(t)=\sin(2\pi jt),\qquad
\phi_{2j}(t)=\cos(2\pi jt).
\]

Master codeword \(c_K\) 由 key 派生并单位化：

\[
\|c_K\|_2=1.
\]

原始连续函数：

\[
f_K(t)=\sum_m c_{K,m}\phi_m(t).
\]

对每个真实 schedule 仅执行加权去均值和全局正缩放：

\[
a_{K,i}
=
\frac{
f_K(t_i)-\sum_jw_jf_K(t_j)/\sum_jw_j
}{
\max_j|f_K(t_j)-\bar f_K|+\epsilon
}.
\]

实现同时保存逐通道加权中心化后的 basis values。Replay 解调必须使用这组 centered
basis，而不是未中心化的原始 sin/cos；否则中心化常数会在受限 phase window 内形成
伪方向。全局正缩放只改变 matched amplitude，不改变 codeword 方向。

### 5.3 不变量

- function digest 不含 grid、index、prompt 或 seed；
- schedule projection digest 可因 8/20/40 grid 不同而改变；
- weighted mean residual 在 tolerance 内为零；
- active count、minimum magnitude 与 energy 达到冻结下限；
- wrong key 必须改变 master codeword；
- codebook pairwise correlation 在预声明上限内；
- 不执行 result-adaptive carrier search。

### 5.4 失败语义

- schedule active support 不足；
- weighted centering 导致 code collapse；
- 8/20/40 grid 的连续函数 digest 不一致；
- candidate pool 无法提供足够低相关 wrong keys。

### 5.5 轻量测试

- 8/20/40 grid function digest 同源；
- 每个 grid 独立零均值；
- wrong key 分离；
- prompt/seed mutation 不改变 function；
- phase perturbation 连续，不出现 index discontinuity。

## 6. P4：Budgeted State-Trajectory Injection

### 6.1 输入

```text
base_model_velocity
prompt_orthogonal_state_direction
continuous_code_value
flow_phase_weight
scheduler_step_weight
relative_norm_budget
cumulative_energy_budget
```

### 6.2 候选控制

\[
\delta v_i
=
\lambda_{\max}
\lambda(t_i)
a_{K,i}
u_{K,i}.
\]

先计算候选 delta，再用一个统一非负 scale 满足：

\[
\|\delta v_i\|_2
\leq
r_{\max}\|v_i\|_2,
\]

\[
\sum_iw_i\|\delta v_i\|_2^2
\leq
B_{\mathrm{trajectory}}.
\]

不得分别缩放 code channels 后再声称 joint budget passed。

### 6.3 不变量

- 不含独立 DC channel；
- 不调用 endpoint controller；
- inactive phase 严格返回原始 model output；
- 最终 delta 与 prompt-orthogonal state direction 保持正向；
- norm 与 cumulative energy 记录基于原始 model output 到最终 output；
- generation trace 保存连续 code 和 operator digests，但不保存 key。

### 6.4 失败语义

- active code 非零但 direction inactive；
- final delta 方向翻转；
- norm/energy 超限；
- scheduler weight 缺失；
- generation/replay function digest 不同。

### 6.5 轻量测试

- NumPy substitute 验证最终 delta；
- torch integration 验证真实 tensor；
- zero/inactive code 严格 no-op；
- cumulative energy 单调；
- direction/norm/energy 三 guard；
- 不触发旧 endpoint/DC 路径。

## 7. P5：Key-Independent Replay Trace

### 7.1 输入与输出

输入：

```text
video
prompt
generation_model
generation_model_revision
replay_schedule
```

输出：

```text
replay_states
base_model_velocities
schedule_intervals
candidate_independent_reliability_weights
replay_provenance
```

### 7.2 不变量

- 每个视频只执行一次 inversion/replay；
- correct/wrong keys 共享 states、velocities、schedule 和 reliability；
- candidate loop 不调用模型；
- replay failure 不得产生部分 positive；
- generation/replay model revision 必须一致；
- 20-step 是首轮冻结 replay grid。

### 7.3 与现有代码关系

复用 `wan_flow_replay_backend.py` 的固定 reverse path 语义，但把旧的整条
`null_forward_key_independent` 可靠性改为同一步 base-predicted transition residual；
它仍完全 candidate-independent，并与 innovation 共用同一次 base velocity。不得让
candidate key 改写 replay states 或可靠性权重。

当前 `prompt_orthogonal_replay.py` 在每个固定 reverse transition 上只调用一次 Wan
base velocity，并在同一步内让全部 candidates 共享 state、velocity、innovation 与
reliability。候选 plane 在 CPU 缓存并逐个短暂搬到运行设备，避免九组高维 plane
同时常驻 GPU；结果包只允许保存标量 step summaries 和二维解调量。20步
key-independent reverse trace 构造固定20次基础 velocity 调用；候选评估在同一
transition 上同时计算 base-predicted residual、innovation 与 candidate-independent
reliability，九候选共同评估固定20次。单视频总计40次，且 trace builder 不接收任何
候选 key。

## 8. P6：Replay Innovation Sequence

### 8.1 数学定义

\[
r_i
=
\frac{z_{i+1}-z_i}{\Delta\sigma_i}
-v_\theta(z_i,t_i,p).
\]

如果 scheduler update 不是一阶 Euler，必须使用 scheduler-specific transition
residual，而不是假装上式精确成立。每个支持的 scheduler 必须提供显式 adapter。

### 8.2 输出概念

```text
innovation_step_vector_or_projection_context
innovation_norm
base_transition_norm
innovation_relative_norm
scheduler_transition_context_complete
replay_reliability_weight
```

不得把完整高维 latent 写入正式结果包。可保存：

- 必要的 keyed/candidate-independent projections；
- norm、digest 和 completeness flags；
- owner-side authenticated sketch 中的受控摘要。

### 8.3 不变量

- innovation 使用未加候选 watermark 的 base model velocity；
- scheduler transition context 完整；
- reliability 与 candidate key 无关；
- prompt 只出现在 base model velocity；
- step ordering 与 continuous flow phase 单调一致。

### 8.4 失败语义

- scheduler adapter 缺失；
- delta sigma 为零或符号不一致；
- replay states/velocity shape 不一致；
- transition residual non-finite；
- candidate key 影响 innovation。

## 9. P7：Synchronous Vector Demodulation

### 9.1 输入

```text
replay_innovation_sequence
candidate_state_directions
candidate_continuous_code
candidate_independent_reliability
candidate_independent_whitener
```

### 9.2 逐步响应

\[
y_{K,i}=\langle r_i,u_{K,i}\rangle.
\]

### 9.3 多通道解调

\[
h_{K,m}
=
\sum_iw_i\rho_i\phi_m(t_i)y_{K,i}.
\]

实现使用相同、与 candidate key 无关的 scheduler/reliability weights 构造二维 basis
Gram matrix，并解冻结的 weighted least-squares system。这个解析去相关只校正真实
schedule 上的基函数非正交性，不从 test identities 拟合 whitening。

不允许在该步骤只返回：

```text
sum(y)
mean(y)
single signed path score
```

### 9.4 Whitening

Whitener 必须满足：

- 来自 calibration split 或冻结解析 null；
- 对全部 candidate keys 相同；
- 不使用当前 identity owner label；
- 不为每个 prompt 单独翻转 sign；
- test split 只允许 apply，不允许 fit。

首轮 mechanism smoke 若尚无独立 calibration，可使用 identity transform，但必须明确
`whitening_not_fitted_smoke_only`，不得临时用四个测试 identity 拟合。

### 9.5 Candidate score

候选 score 可以是：

\[
T_K
=
-\min_{\alpha\in\mathcal{A}_{\mathrm{frozen}}}
\|\tilde h_K-\alpha c_K\|_2^2,
\]

或预声明的 cosine/matched-filter score。允许的 \(\alpha\) 估计方式必须在实现前
冻结；不得在 test score 上连续优化到最有利值。

### 9.6 不变量

- 保存 demodulation vector；
- owner/wrong 使用相同 aggregation；
- ties 计为 owner 失败；
- code dimension 固定；
- vector completeness 是 gate；
- scalar score 不能绕过 vector completeness。

## 10. P8：Candidate Identity Decision

### 10.1 Coverage

首轮机制 smoke：

```text
4 watermarked identities
x (1 owner + 8 wrong keys)
= 36 candidate summaries

4 identities x 8 wrong keys
= 32 pair records

4 identity ranking records
```

Clean controls 具有独立 records，不得混入上述 owner/wrong denominator。

### 10.2 Gate concepts

```text
coverage_ready
operator_separation_ready
continuous_code_ready
innovation_sequence_ready
vector_demodulation_ready
owner_over_wrong_pair_fraction
owner_top1_identity_fraction
prompt_pair_fraction_gap
clean_owner_like_response_ready
quality_and_energy_ready
```

这些概念已映射到 `prompt_orthogonal_*` records 与字段注册；未登记的新统计不得在
GPU 结果后临时加入 gate。

### 10.3 判定顺序

```text
runtime/input failure
-> construction incompleteness
-> candidate separation failure
-> clean false response
-> prompt generalization failure
-> owner/wrong identity failure
-> quality/energy failure
-> mechanism smoke pass
```

通过只允许：

```text
prompt_orthogonal_mechanism_smoke_passed_independent_calibration_design_allowed
```

失败必须：

```text
prompt_orthogonal_mechanism_smoke_failed_stop_instance
```

不得自动切换 carrier、strength、operator rank 或 whitening。

## 11. P9：Governed Evidence Packaging

### 11.1 必需 artifact categories

```text
construction decision
execution decision
generation plan
generation trace summaries
replay provenance
innovation step records
candidate vector summaries
owner/wrong pair records
identity ranking records
failure records
mechanism decision
manifest
diagnostic report
```

### 11.2 边界

- 所有结果先写 Colab 本地 runtime；
- 成功后只上传单 ZIP 与 companion manifest；
- failure recovery 保持独立、非正式、不可阶段推进；
- Notebook 不实现算法或 thresholds；
- result package 不保存 secret key、完整 latent 或模型权重；
- `formal_result=false`；
- `stage_progression_allowed=false`；
- `claim_support_status` 必须明确为 mechanism smoke。

### 11.3 重建

Decision 与 report 必须由 records 重建：

```text
records + frozen config
-> candidate summaries
-> pair and identity records
-> mechanism decision
-> diagnostic report
```

禁止手工编辑 decision 或 report 数值。

## 12. 原语依赖图

```text
P0 Key Domain Separation
├── P1 State Rotation Operator
└── P3 Continuous Trajectory Code

P1 + base state/velocity
-> P2 Prompt-Nuisance Projection

P2 + P3
-> P4 Budgeted Injection

video + model + prompt
-> P5 Key-Independent Replay
-> P6 Innovation Sequence

P0 + P1 + P2 + P3 + P6
-> P7 Vector Demodulation
-> P8 Identity Decision
-> P9 Evidence Packaging
```

## 13. 建议模块边界

以下路径已按职责落地；GPU 结果是下一步外部运行边界：

| Proposed module | 职责 |
|---|---|
| `main/methods/state_space_watermark/state_rotation_operator.py` | 已实现 P0/P1/P2；P2 与 operator 共置以避免重复高维 plane。 |
| `main/methods/state_space_watermark/continuous_trajectory_code.py` | 已实现 P3 连续码字。 |
| `main/methods/state_space_watermark/state_trajectory_injection.py` | 已实现 P4 AC-only 注入与联合预算。 |
| `main/methods/state_space_watermark/replay_innovation.py` | 已实现 P6 FlowMatch Euler adapter 并接入共享 Wan replay trace。 |
| `main/methods/state_space_watermark/trajectory_vector_demodulation.py` | 已实现 P7 二维解调核心并接入候选身份 decision。 |
| `main/methods/state_space_watermark/prompt_orthogonal_replay.py` | 已实现 P5--P7 的共享 Wan replay candidate evaluator；每步基础 velocity 只计算一次。 |
| `experiments/generative_video_model_probe/prompt_orthogonal_state_trajectory_smoke.py` | 已实现 P8/P9 最小实验、双 source 校验与非正式 decision。 |

`main/` 模块不得导入 `experiments/`、`workflows/` 或 Notebook helper。
P1 的 keyed plane 始终在 CPU 规范生成后再搬到 generation/replay 设备；不得分别用
CPU 与 CUDA RNG 仅凭同 seed 假定数值同源。
P4 的 watermarked 与 clean model output 均以 FP32 control 进入 FlowMatch scheduler；
guard 衡量原始 model output 到最终 scheduler control 的实测差值，禁止先在 FP32
声明通过、再把微小 delta 量化回 bf16。

## 14. 实现前冻结清单

在写任何 core code 前，必须明确并测试：

- [x] method/latent/operator/code domain separation payload；
- [x] low-rank operator shape/grouping；
- [x] nuisance basis 与 projection tolerance；
- [x] continuous basis family 与 code dimension；
- [ ] active support、minimum magnitude、energy 和 correlation thresholds；
- [x] `lambda_max`、relative norm 与 cumulative energy budgets；
- [x] FlowMatch Euler scheduler-specific innovation equation；
- [x] demodulation vector definition；
- [x] smoke identity-whitening policy；
- [x] owner/wrong/clean gate；
- [x] controlled-result 与 temporal-failure 双 source binding；
- [x] Colab 本地运行后单 ZIP+manifest 与既有 recovery boundaries；
- [x] 通过仅允许独立 calibration 设计、失败停止当前实例的 claim downgrade rule。

任何未冻结项目不得在 GPU 结果后根据观察值补选。

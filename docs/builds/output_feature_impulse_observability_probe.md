# Output-Feature Impulse Observability Probe 规范

## 0. 目标

本 probe 只回答：

> 在相同生成 identity 下，六个冻结 Flow 区间/通道 impulse 的实际低能量控制，
> 能否经完整生成、decode、保存、re-encode 后，在候选密钥无关的冻结 output
> feature 中产生可归因、非退化且高于最低运行噪声的响应？

它不回答 owner/wrong 方法有效性、跨 prompt 泛化、攻击鲁棒性、正式 FPR 或论文 claim。

## 1. 首次 triage：固定14视频

单一 identity 固定同一：

- prompt；
- seed；
- initial noise；
- generation model revision；
- scheduler 与 inference steps；
- 分辨率、帧数、fps、codec 和所有保存参数。

canonical construction identity 固定为：

- prompt `probe_paper_paper_master_prompt_003`；
- positive prompt UTF-8 SHA-256
  `c4f3a636c9c4393ebf98448f2c30c6648f7e9141a2886bac0cd950001ec03980`；
- seed `probe_paper_paper_master_test_seed_01` / value `2201`；
- generation model `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，revision
  `0fad780a534b6463e45facd96134c9f345acfa5b`；
- `FlowMatchEulerDiscreteScheduler` shift=3 的既有冻结 signature；
- 8 inference steps、33 frames、512×320、guidance 5.0、fps 8；
- `diffusers.utils.export_to_video` + `imageio_ffmpeg`。

同一个 generator seed 必须为14项重建相同 initial noise，禁止连续消费一个 generator
导致每项噪声不同。预训练 generation model 与 VAE 参数都冻结；仅允许 inference-time
sampler/velocity control。

计划顺序严格为：

| 顺序 | 角色 | 数量 |
|---:|---|---:|
| 1 | clean-A | 1 |
| 2 | clean-B | 1 |
| 3–8 | 早/中/晚 × 通道0/1 positive impulse | 6 |
| 9–14 | 同六通道 negative impulse | 6 |

总计14视频。clean repeat 只估计最低成本的同 identity runtime/save noise：

\[
\sigma_{\mathrm{clean,min}}
\propto
\|\phi(V_{\mathrm{cleanA}})-\phi(V_{\mathrm{cleanB}})\|.
\]

它不能冒充正式噪声分布、calibration set 或 fixed-FPR negative population。

## 2. Plan 与执行前不变量

每个 impulse 只激活一个 \(B_{K,j}\) 的一个 state channel。positive/negative 使用相同
名义预算，名义 signed amplitude 固定为 `±0.12`，同时仍受：

- velocity norm ratio `0.02`；
- cumulative Flow energy ratio `0.000015`；
- direction cosine `0.999`；
- actual FP32 delta guard。

计划不得根据输出选择更强幅度、不同 interval、第四个 channel、额外 seed 或新的
feature。

本轮轻量原语只构造 plan 和验证实际 exposure；没有 GPU runner、Notebook handler
或 Drive request。

14个 output feature 必须分别由对应单视频 governed feature record 提供。每条记录
在 feature 形成时绑定 `impulse_probe_id`、冻结 feature schema digest 与256维
little-endian float64 feature bytes 的 SHA-256 行摘要。汇总 validator 要求
`probe_ids` 与本节冻结 plan 完全同序，并要求 feature/design/plan 三方 identity
一致；禁止在拿到14×256矩阵后按位置补写 ID。ID 与数值一起重排违反冻结顺序，
只重排数值则违反原记录的行绑定摘要。摘要不是密钥认证，未来 adapter 必须从
governed per-video record 读取，不能对待检矩阵事后重算。

## 3. Step-wise exposure 与 \(A_{\mathrm{actual}}\)

8-step schedule 不是由 trace 自报。config 已冻结完整 sigma grid、step index、
`flow_phase`、`delta_sigma`、三宏区间 assignment 与每区间8维 unit waveform；
validator 根据 config 重算 digest。任一 trace 少于/多于8步，或 step order、phase、
delta-sigma、interval、waveform/digest 不一致，立即 fail-closed。

对每个 impulse 保存完整 step trace：

\[
a^{\mathrm{intended}}_{i,t},\quad
a^{\mathrm{actual}}_{i,t},\quad
\|\delta v_{i,t}\|,\quad
\gamma_{i,t}^{\mathrm{projection}},\quad
E_{i,t}^{\mathrm{cumulative}},
\]

其中 polarity 位于 state-update coordinate，且

\[
a^{\rm actual}_{i,t,k}
=\Delta\sigma_t\langle
  (\hat v_{i,t}^{\rm FP32}-v_{i,t}^{\rm FP32}),U_{K,k}
\rangle.
\]

因此 velocity control 符号显式包含 `sign(delta_sigma)`，不能直接对调用者提供的
数值求和。trace 还必须包含 base velocity norm、remaining energy、actual velocity
basis coordinates、direction cosine、norm/energy guard；冻结 adapter 从实际 FP32
delta 重算这些量，不信任 caller boolean。canonical basis 的序列化 float32 列先以
float64 norm 归一化并单次 cast 为 FP32 effective direction；注入和六通道坐标共同
使用这一方向。坐标与 actual norm 均以 float64 reduction 计算，direction cosine
只允许机器舍入范围内的 `[-1,1]` clamp。非零 waveform 是 scheduled active：若
base norm、剩余能量或实际 FP32 可表示控制导致非零控制不存在，必须 fail-closed，
不得改写为 inactive no-op。

actual/intended ratio 逐 step 计算。intended 为零但 actual 非零时立即失败。所有
cumulative energy 必须有限、非负且单调。

六维压缩必须通过方法合同中的 waveform cosine、sign symmetry、amplitude symmetry、
cross-channel leakage、rank 与 condition gates。失败时保留 step-wise matrix：

\[
A_{\mathrm{stepwise,actual}},
\]

并停止 Gate A；不得用 nominal 6-D impulse matrix 替代。

## 4. 传递检查点

同一 actual design 必须在下列位置读取响应：

| Checkpoint | 目的 | Primary |
|---|---|---|
| `T_latent` | final latent 是否仍含响应 | 否 |
| `T_decoded` | VAE decode 是否抹除响应 | 否 |
| `T_saved_video` | 保存/codec 是否抹除响应 | 否 |
| `T_reencoded` | output-side VAE encode 是否保持响应 | 否 |
| `T_output_feature` | 冻结 output-only feature 的实际传递 | **是** |
| `T_replay_diagnostic` | replay 内部局部一致性定位 | 否 |

前五个 primary checkpoint 的表示/shape 已冻结为7维 latent summary、48维 pre-save
RGB summary、48维 saved RGB24 summary、256维未归一化 reencoded summary 和256维
L2 output feature。前五项缺失或不有限为 construction failure。
`T_replay_diagnostic` 是可选项：缺失或失败不影响 Gate A，且其任何结果都不能支持
positive。只有 `T_output_feature` 能支持 primary gate。

## 5. Gate A：样本内因果可观测

令

\[
d_c=\|y_{\mathrm{cleanA}}-y_{\mathrm{cleanB}}\|_2,\qquad
\sigma_c=\max(d_c/\sqrt2,10^{-6}).
\]

对六个通道定义
\(r_k=(y_k^+-y_k^-)/2\)，并令
\(R=[r_0,\ldots,r_5]\)。统计量和聚合规则固定为：

- output SNR：\(\min_k\|r_k\|_2/\sigma_c\)；
- noise-normalized minimum singular：
  \(\sigma_{\min}(R)/\sigma_c\)；
- effective rank：奇异值不低于
  \(\max(10^{-6},2\sigma_c)\) 的数量；
- output transfer condition：对每列再除以实际
  \(a_{k,+}^{actual}-a_{k,-}^{actual}\) 后所得 transfer matrix 的条件数；
- antisymmetry cosine：六个
  `cos(y+ - mu0, -(y- - mu0))` 的最小值；
- antisymmetry residual：六个
  \(\|(y^+-\mu_0)+(y^--\mu_0)\|/
    (\|y^+-\mu_0\|+\|y^--\mu_0\|)\) 的最大值。

即使 \(d_c=0\)，分母也只能使用有限的 `1e-6` noise floor，不能产生 infinity。
还必须满足绝对 response `min ||r_k|| >= 1e-4` 和绝对最小奇异值 `>=1e-6`，
所以零噪声或数值分辨率以下的 clean repeat 不会自动通过。

Gate A 必须同时检查：

1. `A_actual` 完整、rank=6、condition number ≤20；
2. 每列 output response 相对 clean-A/B noise floor 的 SNR ≥3；
3. noise-normalized minimum singular value ≥2；
4. effective rank=6；
5. output transfer condition number ≤20；
6. positive/negative antisymmetry cosine ≥0.9；
7. normalized antisymmetry residual ≤0.25；
8. 五个 primary 传递 checkpoint 全部完成；replay diagnostic 可选；
9. 响应不是仅由总 delta norm、质量下降或公共视频内容产生。

高维 feature 空间中的数学满秩单独不构成通过。若 Gate A 失败，立即停止当前 carrier
或当前 feature；不能训练 feature、换 layer、调幅度或扩大样本修补。

Gate A 通过只允许设计第二个独立 identity，不允许直接实现 observer。

## 6. Gate B：跨 identity 可识别

第二个 identity 必须在执行前独立冻结，并与第一个 identity 分离。第一个 identity
可定义公共坐标、normalization 或 transfer map；第二个只能 apply。

禁止：

- confirmation/test identity Procrustes；
- 结果后符号翻转；
- 自适应 permutation/alignment；
- 用 confirmation identity 重新归一化；
- 将 confirmation identity 混入未来 observer calibration。

公共映射下至少检查：

- stage prediction fraction ≥`5/6`；
- sign prediction fraction ≥`5/6`；
- principal angles ≤30°；
- normalized Gram difference ≤0.25；
- held-out transfer prediction error ratio ≤0.5；
- output-feature owner/wrong selectivity。

第二 identity 只构成 construction confirmation，不能外推成 prompt/seed/model
泛化。

## 7. 独立 key-selectivity construction

首次14视频不得塞入 wrong key。Gate A 通过后，才允许单独设计最低成本的
key-selectivity construction：

- 运行前冻结一个 wrong key，固定 domain-separated candidate index `0`；
- 不从候选池按结果选择；
- 只测早/中/晚各一个代表性通道；
- owner 与 wrong 使用相同预算；
- 检查 output-feature transfer 是否区分 owner/wrong。

latent 随机正交、总 norm 差或质量下降都不能替代 output selectivity。该 construction
也是 Gate B 完成的必要组成，但不是 owner/wrong 方法有效性 smoke。

## 8. Gate C：组合轨迹与阶段顺序可辨识

正式名称固定为：

```text
组合轨迹与阶段顺序可辨识
```

不得称“组合动力学”。在 Gate A、B 通过后，至少比较：

- ordered composite；
- same-energy permuted composite；
- 必要时增加 reversed composite。

必须检查：

- composite response 相对 single-impulse transfer 的叠加性；
- 总预算仍在正式允许范围；
- cross-channel cancellation；
- 顺序差异相对 clean noise ratio ≥3；
- 差异不能由总 norm 或视频质量变化解释。

即使 Gate C 通过，也只能证明阶段顺序在 output feature 中可辨识，不能声称
\(F_K/G_K\) 状态动力学贡献。

## 9. 后续 observer 授权边界

只有 Gate A、Gate B、Gate C 全部通过，才允许另起设计任务：

```text
F_K/G_K
→ batch observer
→ prediction-error / matched-dynamic score
```

首轮 score 不得称正式 LLR。正式动力学贡献必须由 complete observer、independent
interval template 和 static endpoint 三方消融证明。

攻击、pilot、fixed-FPR、baseline、formal result 与 stage progression 在此之前全部
保持关闭。

## 10. 阶段状态机

```text
construction_contract_local_audit
  ↓ 独立只读审核通过
independent_readonly_audit
  ↓ 用户明确授权
commit_push_authorization_pending
  ↓ 提交推送完成且用户另行授权GPU
impulse_triage_execution_authorization_pending
  ↓ runner/handler审核完成，等待用户执行固定Notebook
impulse_triage_execution_authorized_pending_user_colab_run
  ↓ 14-video真实运行
sample_internal_causal_observability_gate
  ├─ FAIL → stop current carrier/feature
  └─ PASS → cross_identity_construction_confirmation_pending
               ↓ 第二identity + 独立key-selectivity
             Gate B
               ├─ FAIL → stop
               └─ PASS → composite_trajectory_order_identifiability_pending
                            ↓ Gate C
                            ├─ FAIL → stop
                            └─ PASS → state_dynamics_and_batch_observer_design_pending
```

首次14-video runner 已在 commit `47485be2...` 上真实执行并得到 Gate A FAIL；
该失败必须保留，不能由后续候选路线覆盖或改写为成功。当前只允许另起独立的
`Gate A root-cause amplitude/feedback diagnostic`，用固定半幅和只读历史 FAIL
区分幅度、feedback 与 carrier/feature mismatch 候选。该诊断不是 Gate A 重试，
本地 tests/harness 也仍只审核合同和 wiring，不提供 construction observability
或方法有效性证据。

固定半幅六视频已在 commit `f06a0934...` 上真实完成。实际 exposure 接近严格半幅，
但保存 RGB 与 output feature 的 odd/common 未按局部线性缩放；late latent
six-basis 仍 signed，而 output feature common-dominated；early latent 同时受
feedback 候选混淆。结果只支持多候选根因，Gate A FAIL 不变。

下一步首先对这六个既有 MP4 做候选密钥无关的 CPU-only 逐帧、固定三区间和相邻帧
差分分析。该分析不得事后选择 feature；若固定 video-time aggregations 也不能在
early/late 同时满足原 signed gate，则正式停止当前 distributed random 3×2 carrier
+ global latent-time mean/L2 feature 组合，转入 public decoder-Jacobian-aligned
dictionary 与 fixed-whitened video-time feature 的独立 construction 设计。

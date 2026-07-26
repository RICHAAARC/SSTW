# 帧状态 signed observability construction 计划

## 0. 状态与授权边界

本文实例化
`frame_state_synchronized_generative_flow_video_watermark` 主路线的最薄
construction 合同。权威算法语义仍以：

- `frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`
- `frame_state_synchronized_generative_flow_video_watermark_method_design.md`

为准。

当前状态严格为：

```text
method_contract_design_only
formal_result=false
stage_progression_allowed=false
runtime_implementation_authorized=false
```

本计划只允许静态 config、canonicalization、validator 和轻量测试。它没有实现
carrier runtime、视频生成 runner、observer、Notebook handler 或 Colab request，
也不授权 GPU、Colab、Drive、攻击、fixed-FPR、baseline 或 paper claim。

## 1. 本轮要回答的唯一问题

本轮只问：

> 一个公开、候选密钥无关的帧窗口 carrier atom，能否在独立视频 identity 上，
> 通过最终保存视频的局部时序特征保持非零、反对称且可由 construction transfer
> 预测的 signed response？

它不问：

- 多窗口是否可分；
- wrong key 是否可拒绝；
- 状态转移是否有效；
- observer 是否优于逐帧模板；
- 插帧、删帧、裁剪或变速是否可同步；
- fixed-FPR 是否成立。

因此正式名称使用 `frame_state_signed_observability_construction`，不在文件名中使用
弱阶段编号，也不复用历史 Flow-stage Gate A 名称。

## 2. 非递归 protocol digest

config 顶层的 `protocol_contract` 是唯一 digest 输入：

\[
\mathrm{protocol\_digest}
=
\operatorname{SHA256}
\left(
\operatorname{canonical\_json}
(\mathrm{protocol\_contract})
\right).
\]

canonical protocol object 只允许：

- JSON object/array；
- UTF-8 string；
- boolean；
- base-10 integer。

禁止 float、NaN、Infinity、重复 key、Unicode normalization、额外空白或非冻结
序列化规则。key 按 Unicode code point 排序，分隔符严格为 `,` 和 `:`。

下列值不进入该摘要：

```text
protocol_digest
public_context_record
public_nonce_random
context_digest
execution_identity_contract
authorization_boundary
runtime_records
runtime_paths
timestamps
```

因此 public context 只携带已经计算完成的 `protocol_digest`，不存在把 nonce、
context digest 或 digest 自身递归纳入 protocol object 的路径。

本合同冻结摘要为：

```text
b57b1d620850296509e4e8d55749cea8f7b3ca849cb10ed908c0470715221b6f
```

validator 必须同时：

1. 从 `protocol_contract` 重算；
2. 与 config 自报值比较；
3. 与代码内冻结常量比较。

只修改 config 并同步伪造自报摘要不能通过。

## 3. Public context

运行时 public context 的 exact key set 为：

```text
context_schema_id
method_id
protocol_digest
public_nonce_random
state_clock_rate_denominator
state_clock_rate_numerator
state_window_count
```

当前设计冻结：

```text
context_schema_id =
  "sstw_frame_state_public_context_v1"
method_id =
  "frame_state_synchronized_generative_flow_video_watermark"
state_clock_rate_numerator = 8
state_clock_rate_denominator = 1
state_window_count = 1
```

`public_nonce_random` 必须是嵌入前生成的128-bit nonce，以32位小写 hex 表示。
本 config 不生成、不保存具体 nonce。一个 identity 的四个 counterfactual 输出
共享同一 public context；不同 identity 不得共享运行记录或事后重采样 nonce。

主检测输入仍是：

```text
video + key + public_context_record
```

prompt、seed、initial latent、内部 Flow trace 和 carrier trace 不得进入该 record。

## 4. Identity 隔离与最小视频计划

计划精确包含两个互不混用的 identity，每个 identity 四项，共8项：

| index | identity | probe | polarity | 职责 |
|---:|---|---|---:|---|
| 0 | \(C_0\) | `construction_clean_a` | 0 | dictionary reference、clean repeat |
| 1 | \(C_0\) | `construction_clean_b` | 0 | minimum runtime-noise reference |
| 2 | \(C_0\) | `construction_positive` | +1 | \(T_0\) positive response |
| 3 | \(C_0\) | `construction_negative` | -1 | \(T_0\) negative response |
| 4 | \(A\) | `signed_observability_clean_a` | 0 | held-out causal intercept |
| 5 | \(A\) | `signed_observability_clean_b` | 0 | held-out runtime-noise reference |
| 6 | \(A\) | `signed_observability_positive` | +1 | apply-only positive response |
| 7 | \(A\) | `signed_observability_negative` | -1 | apply-only negative response |

同一 identity 内四项必须使用相同 prompt、seed、initial noise、模型 revision、
scheduler 和 public context。\(C_0\) 与 \(A\) 的 prompt/seed 必须不同，并在未来
独立 execution contract 中冻结。

当前 config 只包含以 `_placeholder` 结尾的身份占位字段。占位字段阻止 runtime
执行，不支持 claim；不得由 runner 临时填充。用户另行授权 execution identity
冻结前，`construction_execution_allowed` 保持 false。

\(C_0\) 在 design contract 中承担以下 construction 职责，但这不是执行授权：

- 构造一个公开 dictionary atom；
- 产生固定 construction center；
- 估计 \(T_0=H_0D_0\)。

identity \(A\) 只能 apply，禁止：

- 重选 atom、window、feature 或 scale；
- 重算 \(T_0\)；
- 对 held-out response 做 sign flip、Procrustes 或 whitening；
- 将 \(A\) 回流到 \(C_0\)。

## 5. 单窗口与公开 carrier atom

视频合同冻结为：

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
revision 0fad780a534b6463e45facd96134c9f345acfa5b
33 frames
512 × 320
8 fps
8 Flow inference steps
latent layout [1,16,9,40,64]
```

唯一输出帧窗口为 frame indices `[11,22)`。公开映射采用：

\[
n_{\mathrm{latent}}
=
\left\lfloor
\frac{n_{\mathrm{video}}\,9}{33}
\right\rfloor,
\]

因此 carrier temporal support 固定为 latent indices `[3,4,5]`，权重为：

\[
[0.25,\ 0.50,\ 0.25].
\]

唯一 atom \(D_0\) 的 construction 规则是：

```text
public Wan VAE decoder local Jacobian
→ restrict to latent temporal support [3,4,5]
→ target the pre-save decoded local feature as a differentiable surrogate
→ fixed public initialization
→ 8 fixed power iterations
→ finite-iteration decoder-Jacobian-aligned direction
→ apply fixed [0.25,0.50,0.25] temporal weights after iteration 8
→ largest-absolute-coordinate-positive sign
→ float32 unit vector
```

固定初始化不是调用方任意输入。它在受限 support
`[1,16,3,40,64]` 上按
`SHA256(UTF8(domain) || NUL || uint64_be(counter))` 连续取 digest bits（MSB first），
以0/1分别映射到 \(-1/+1\) 的 Rademacher 序列；受限 support 外严格置零。
bit stream 按 compact restricted-support C-order
`batch,channel,restricted-time,height,width` 填充；这等价于遍历完整
`[1,16,9,40,64]` C-order 并跳过非 support time 坐标。相邻 SHA-256 digest 的
bits 无 byte padding 地连续拼接，直到 support 填满。
counter 从0开始；若最大绝对值坐标并列，使用 C-order 最小 flat index 决定
正号规范。
每轮执行 \(P J^\mathsf{T}J P\)（\(P\) 为受限 latent-time support 投影）后
用 float32 L2 归一化，精确执行8轮，不允许按运行
结果选择 iteration；任何零范数立即 fail-closed。固定时间权重先乘到 latent
slices 3/4/5，再做一次全局 float32 L2 归一化，最后应用 sign canonicalization。
因此最终 \(D_0\) 的准确身份是“fixed 8-iteration, temporally weighted
decoder-Jacobian-aligned direction”，不是精确 leading singular vector，也不
声称是 \(P J^\mathsf{T}JP\) 的最终 eigenvector。NPZ 只允许一个 array：
`frame_state_public_atom`，完整 shape 为 `[1,16,9,40,64]`。

Jacobian 的 differentiable target 也不是任意 decoded tensor：它固定采用
diffusers 0.35.2 `AutoencoderKLWan`，对 clean final latent 先执行
`latent * vae.config.latents_std + vae.config.latents_mean`，再取得 VAE decode
sample。随后在 Jacobian graph 内精确镜像 Wan video processor 的
`clamp(raw_sample/2+1/2,0,1)`，转换为 `[batch,frame,height,width,channel]`
float32；clamp 属于 Jacobian 路径。对 postprocessed frames `[11,22)` 按与保存
视频 primary feature 相同的4×4 cell 和 RGB 顺序形成528维 local temporal
surrogate。它不读取保存编码结果，只用于公开 atom construction；saved-video
readback 仍是 primary Gate 边界。

该 atom 可以依赖 \(C_0\) clean final latent 作为 construction reference，
但不能依赖 candidate key、identity \(A\) 或运行后 Gate 得分。保存视频编码不可
直接微分，因此 Jacobian 只对 pre-save decoded feature 构造；真正的
\(T_0\) 和 Gate 0 仍必须经过 saved-video readback 测量。

由于 \(D_0\) 是公开 dictionary，它必须形成独立 public NPZ artifact，绑定 exact
shape、float32 little-endian C-order values 和 SHA-256。per-video records 只引用
该 artifact digest，不重复写入或偷偷重建另一套 atom。仅记录 digest 而不提供
public dictionary artifact 不满足本合同。

## 6. 时间保持的 output feature

primary feature 直接来自保存后真实回读的 RGB24 视频，不使用 prompt-conditioned
replay，也不做 output VAE latent-time global mean。

对 frames `[11,22)` 的每一帧：

1. 按文件顺序读取 RGB24；
2. 转 float64 并除以255；
3. 划分固定4×4等面积空间 cells；
4. 保留每个 cell 的 RGB channel mean；
5. 按 frame、cell-row、cell-column、RGB 顺序拼接。

输出维数为：

\[
11\times4\times4\times3=528.
\]

不做时间平均，不做逐视频 L2 normalization，不根据结果选择 frame、cell、
channel 或 frequency。raw feature 本身不做中心化。当前最薄合同使用 identity
whitening，不拟合 covariance；\(C_0\) clean-A/B 算术均值只作为 construction
transfer intercept，各 signed gates 则在当前 identity 自己的 clean-A/B 算术
均值坐标内计算，identity \(A\) 不复用 \(C_0\) center。

## 7. \(T_0\) 与实际 exposure

高维 \(H_0\) 仍只是概念算子。唯一可估计对象是：

\[
T_0=H_0D_0.
\]

这里的 \(\phi\) 只表示528维 `saved_video_local_temporal_feature`；latent scalar
与 decoded checkpoint 不各自估计 transfer。

\(C_0\) 使用实际 scheduler-state signed exposure：

\[
a^{\mathrm{actual}}
=
\sum_n
\Delta\sigma_n
\left\langle
\Delta v^{\mathrm{actual}}_n,D_0
\right\rangle,
\qquad
\Delta\sigma_n=\sigma_{n+1}-\sigma_n.
\]

probe 的 \(+1/-1\) 定义在 scheduler **state-update** 坐标，不是 velocity
坐标。由于 Wan FlowMatch 的 \(\Delta\sigma_n\) 可为负，intended velocity sign
固定为 `signed_state_coefficient * sign(delta_sigma)`，从而 state update 的
极性与 probe identity 一致。每步 dot product 使用真实 float32
`constrained_velocity-base_velocity`，跨步 exposure 以 float64 累加。

\[
\hat T_0
=
\frac{
\phi(V^+_{C_0})-\phi(V^-_{C_0})
}{
a^{\mathrm{actual}}_+
-
a^{\mathrm{actual}}_-
}.
\]

若实际 exposure 差为0，必须 fail-closed。identity \(A\) 不重新估计 transfer；
其冻结预测为：

\[
\hat o_A
=
\hat T_0
\frac{
a^{\mathrm{actual}}_{A,+}
-
a^{\mathrm{actual}}_{A,-}
}{2}.
\]

不得使用 nominal \(\lambda\)、unsigned delta norm 或自报 guard 替代 actual
signed exposure。不得生成独立 `H0` artifact、statistics 或 claim。

## 8. 数值路径

冻结：

```text
transformer_compute_dtype = bfloat16
scheduler_state_dtype = float32
carrier_control_dtype = float32
actual_delta_measurement_dtype = float32
lambda = 12 / 100
velocity_norm_ratio_budget = 2 / 100
flow_energy_budget_ratio = 15 / 1000000
```

clean、zero-control、positive、negative 必须使用相同 cast、控制函数和 scheduler
update 路径。clean/zero-control 的 actual delta 必须精确为0；active actual delta
必须非零。所有 norm、energy、direction 和 exposure 必须从：

```text
constrained_velocity_float32 - base_velocity_float32
```

重算，并同时满足局部窗口预算和全局预算。

本 design-only 批次尚未冻结真实8-step `sigma` trace 和 \(w(\tau)\) waveform；
`scheduler_id` 与 `num_inference_steps` 不能被调用方解释为可自行选择的默认值。
它们必须在未来独立、经审核的 execution contract 中逐步冻结并进入更新后的
protocol digest。在此之前，actual exposure 公式只是方法合同，runtime 必须保持
禁止；不得用隐式全1 waveform 或历史 Flow-stage schedule 补空。

## 9. Checkpoints 与 Gate 0

每个正负 pair 必须在以下 exact representation 保留 signed response：

| checkpoint | source boundary | raw representation | dimension / flatten |
|---|---|---|---|
| `final_latent_carrier_projection` | VAE decode 前 final latent float32 | full C-order final latent 与 public atom 的 float64 dot；先形成 raw scalar，随后按 identity clean center | 1 / scalar |
| `decoded_local_temporal_feature` | VAE decode 后、video export 前的上述 postprocessed float32 \([0,1]\) frames | `[11,22)` 的4×4 cell RGB mean，以 float64 计算 | 528 / frame,cell-row,cell-column,channel |
| `saved_video_local_temporal_feature` | 保存视频真实 RGB24 readback | RGB24 除255后的4×4 cell RGB float64 mean | 528 / frame,cell-row,cell-column,channel |

primary boundary 是保存视频。定义 identity \(A\) 的 clean intercept：

\[
\mu_{A,j}=\frac12(r_j(V_{\mathrm{cleanA}})+r_j(V_{\mathrm{cleanB}})).
\]

\[
\delta_{+,j}=r_j(V_+)-\mu_{A,j},\qquad
\delta_{-,j}=r_j(V_-)-\mu_{A,j}.
\]

\[
o_j=\frac12(\delta_{+,j}-\delta_{-,j}),\qquad
c_j=\frac12(\delta_{+,j}+\delta_{-,j}).
\]

三个 checkpoint 都必须应用下列公共 signed morphology/noise gates：

- antisymmetry cosine \(\ge 0.90\)；
- antisymmetry residual \(\le 0.25\)；
- common/odd ratio \(\le 0.50\)；
- odd/clean-noise ratio \(\ge 3.0\)；

这些公共 gates 在 \(C_0\) 与 \(A\) 各自的 clean-centered 坐标内都计算；
\(C_0\) 的结论仅是 construction readiness，不能产生 Gate 0 PASS。最终 Gate 0
decision identity 固定为 \(A\)。

唯一 \(T_0\) 是 saved-video primary 的528维 transfer。只有
`saved_video_local_temporal_feature` 额外应用：

- \(T_0\) direction cosine \(\ge 0.90\)；
- actual-exposure transfer relative error \(\le 0.50\)。

不得为 scalar latent projection 或 decoded checkpoint 另估 \(T_0\)，也不得把
primary T0 gates 偷换为三个层级各自可选的 transfer。前两层只作因果链路
signed diagnostics；saved-video primary 同时承担公共 signed gates 与唯一
apply-only transfer gates。\(T_0\) 只由 \(C_0\) 估计，在 identity \(A\) 上
严格 apply-only。

其中 clean-noise norm 固定为
\(\tfrac12\|r_j(V_{\mathrm{cleanA}})-r_j(V_{\mathrm{cleanB}})\|_2\)，
所有 clean-noise 分母使用该值与冻结 \(10^{-6}\) floor 的最大值，禁止零分母
自动通过。odd/clean-noise ratio 使用
\(\|o\|_2/\max(n_{\mathrm{clean}},10^{-6})\)。
\(T_0\) direction cosine 只在 saved-video primary 比较 observed \(o_A\) 与
上式 \(\hat o_A\)；
transfer relative error 固定为
\(\|o_A-\hat o_A\|_2/\max(\|o_A\|_2,10^{-6})\)。

三个 checkpoints 的公共 signed gates 与 primary 的两项 T0 gates 必须全部通过；
单 checkpoint 通过不足以支持 Gate 0。FAIL
必须停止当前 carrier/feature，不能调窗口、阈值或强度后覆盖。

即使未来 Gate 0 PASS，也只允许设计双窗口 Gate A，不支持 observer、攻击、
fixed-FPR、baseline、paper claim 或项目阶段推进。

## 10. 当前完成定义

本任务完成只意味着：

- config 可被 strict loader 读取；
- protocol digest 非递归且可重算；
- public context canonical bytes 可 fail-closed 验证；
- 8项 plan 可确定性重建；
- mutation、额外字段、重复 JSON key、float 和非 canonical context 被拒绝；
- README 和 field registry 与合同一致。

它不是 carrier 已实现、视频已生成、Gate 0 已执行或方法有效性证据。

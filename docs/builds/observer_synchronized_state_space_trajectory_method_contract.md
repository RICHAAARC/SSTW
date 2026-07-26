# Observer-Synchronized State-Space Trajectory Watermark 方法合同

## 0. 文档状态与证据边界

### 0.1 主路线时间轴纠正

本文件主体记录并保留历史的 **Flow-stage-indexed construction**、真实 Gate A FAIL
及其后续根因诊断合同。它不再作为 SSTW 当前主路线的完整方法定义。

当前主路线的方法设计决策已经调整为：

```text
视频帧/窗口时间：定义低维水印状态演化和同步目标
生成 Flow 时间：只负责把整条帧状态轨迹写入视频
```

新的规范文档为：

- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
- `docs/builds/frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`

本文件以下历史 Gate A 使用 Flow early/middle/late 作为主动设计轴，并使用
latent-time global mean feature。其 FAIL 只否定对应 carrier/feature construction，
不能冒充新帧状态路线的执行结果，也不能被新路线覆盖为成功。时间轴调整是新路线
的方法设计决策，不是该历史 FAIL 直接证明的实验结论。

本文定义一个独立候选方向：

```text
Observer-Synchronized State-Space Trajectory Watermarking
```

本文记录的历史冻结点状态仅为：

```text
current_distributed_carrier_global_feature_gate_a_failed_stop
```

它不是旧 prompt-orthogonal 路线的改名，也不覆盖该路线的真实失败。旧路线及其
recovery/result 必须保留为历史 control。历史结果只能支持：

- replay reliability `0.9904` 表示回放内部局部一致性较高；
- 该数值没有证明 replay 与真实 generation trajectory 对齐；
- 真实 owner 失败只说明当前状态依赖载波在该回放观测下不可识别；
- 不能据此声称抽象的状态空间水印可行或不可行。

在该历史冻结点，本文和配套轻量代码均满足：

- `formal_result=false`；
- `stage_progression_allowed=false`；
- 不支持 paper claim；
- 真实14视频 Gate A 与后续固定半幅六视频诊断均已执行，Gate A FAIL 保持不变；
- 当时只授权对既有六个保存视频做候选密钥无关的 CPU-only 时空 signed-response
  分析，并冻结后续实验规范；不授权新的 GPU/Colab/Drive 操作；
- 不授权实现 observer、正式 LLR、攻击、pilot、fixed-FPR 或 baseline。

上述“当时授权”只描述本历史合同冻结时的边界，不构成现行授权。现行主路线边界以
`docs/builds/frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`
和
`docs/builds/frame_state_synchronized_generative_flow_video_watermark_method_design.md`
为准。

## 1. 研究问题与结构分离

预训练视频 Flow/扩散模型参数保持冻结，不训练或微调；每个14-video plan item 都必须
从相同 seed 重新构造 generator，得到相同 initial noise，而不是连续消费同一 RNG。
宿主生成状态 \(z(t)\) 与
独立低维水印状态 \(s(t)\) 必须分离：

\[
\dot z(t)=v_\theta(z(t),t,p)+\delta v_K(t;s(t)),
\qquad
s(t)\in\mathbb R^{d_s}.
\]

当前只研究推理期 sampler/velocity control 注入是否能在最终保存视频上留下
可测的因果传递，不构造 \(F_K/G_K\)，也不假定 \(s(t)\) 已具有可识别动力学。

Primary detection 的允许输入固定为：

```text
final saved video + key + pre-frozen public feature extractor
```

prompt-conditioned replay 只能作为 construction diagnostic。即使 replay 指标很好，
也不能支持 primary positive。

## 2. 两个时间轴

必须区分：

| 时间轴 | 当前职责 |
|---|---|
| Flow/diffusion time \(t\) | 本轮唯一主动设计轴，分为3个宏观区间 |
| video frame time \(\tau\) | 既有失败结果的 CPU-only 逐帧/固定三区间/相邻差分诊断；尚非 primary feature |

Flow 区间固定为：

\[
I_0=[0,\tfrac13),\quad
I_1=[\tfrac13,\tfrac23),\quad
I_2=[\tfrac23,1].
\]

首轮不得加入删帧、插帧、DTW、变帧率、frame-time alignment 或攻击。旧冻结
extractor 对文件顺序中的全部解码帧做单一全局均值；该表示已在真实 Gate A 中失败，
且不构成 video-time 分析或鲁棒性证据。现有六视频的逐帧诊断只用于决定下一次
最小 construction 测什么，不能事后选出新的正式 feature。

## 3. 独立水印状态与阶段 block

冻结：

\[
J=3,\qquad d_s=2,\qquad r=Jd_s=6.
\]

Key \(K\) 定义六维正交 latent basis \(U_K\in\mathbb R^{D\times6}\)。construction
latent layout 固定为 Wan `[1,16,9,40,64]`，按 C-order 展平，\(D=368640\)。
`U_K` 的实现唯一冻结为：

1. master key 至少16个 UTF-8 bytes；
2. HMAC-SHA256 domain
   `sstw_observer_synchronized_impulse_construction_basis` 派生
   `construction_owner_stage_basis` subkey，不输入 prompt、seed、model 或 grid；
3. CPU 上用 `SHAKE256(subkey || frozen-domain)` 产生 big-endian uint64，
   再按固定 Box-Muller 顺序形成 float64 normal matrix；
4. 按列0到5执行 float64 modified Gram-Schmidt；
5. 每列绝对值最大坐标必须为正，随后转 little-endian float32；
6. 六列 C-order bytes 的 SHA-256 为 basis digest；只保存 digest，不把 basis 写入结果；
7. runtime 从上述 canonical float32 列以 float64 norm 归一化、单次 cast 为 FP32
   effective directions；注入与坐标测量共同使用该方向，坐标和 actual norm 均用
   float64 reduction，禁止混用 raw float32 列与另外归一化的注入方向。

预冻结 wrong key 只允许 HMAC domain-separated index `0`，不得从候选结果选择。
每阶段使用不同的二维选择器：

\[
E_0=[e_0,e_1],\quad
E_1=[e_2,e_3],\quad
E_2=[e_4,e_5],
\]

\[
B_{K,j}=U_KE_j.
\]

禁止写成 \(U_KR_j\) 后把三个二维旋转冒充不同子空间。必须记录并门禁：

- 每个 \(B_{K,j}^{T}B_{K,j}\) 相对 \(I_2\) 的 Gram 误差；
- 跨阶段 \(\|B_{K,i}^{T}B_{K,j}\|_2\) coherence；
- 总 velocity/Flow energy；
- wrong-key latent coherence。

冻结上限分别为 `1e-5`、`0.05`、现有正式预算
`velocity_norm_ratio_budget=0.02`、
`flow_energy_budget_ratio=0.000015`，以及 wrong-key latent coherence `0.25`。
这些 latent 条件只是必要条件，绝不是 output observability 的充分条件。

## 4. 冻结的 construction feature

construction extractor 记为：

\[
\phi_{\mathrm{construction}}(V)\in\mathbb R^{256}.
\]

它采用公开、冻结且不训练的 Wan output-side VAE 规范：

| 项目 | 冻结值 |
|---|---|
| encoder ID | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers::vae` |
| revision | `0fad780a534b6463e45facd96134c9f345acfa5b` |
| encoder class/subfolder | `AutoencoderKLWan` / `vae` |
| 软件/dtype | diffusers `0.35.2`；VAE encode `bfloat16`；normalized latent `float32` |
| 输入 | 恰好33帧、512×320、RGB24；尺寸/帧数不符 fail-closed，禁止 resize |
| 解码 | imageio v3，RGB24，文件顺序全部帧 |
| encode 实现 | 复用 `encode_video_to_wan_endpoint_latent` 的 CPU-resident spatiotemporal streaming；`patch_size=None` |
| streaming | temporal chunk 4；tile 128×128；stride 96×96；CUDA peak/free guards 16/12 GiB |
| video normalization | float32 RGB / 127.5 - 1 |
| posterior statistic | deterministic mode / first moment |
| latent normalization | subtract `latents_mean`, multiply inverse `latents_std` |
| layer/tensor | normalized VAE endpoint latent `[B,16,T,H,W]` |
| temporal pooling | latent-time 单一全局均值 |
| spatial pooling | 4×4 等面积 latent spatial 网格；边界为 `floor(i*size/4):floor((i+1)*size/4)` |
| pooling dtype | float64 accumulator，输出再转 schema 指定值 |
| channel statistic | 保留16个 latent channel 的 cell mean |
| output dimension | 16×4×4=256 |
| output normalization | L2；epsilon `1e-12` 零拒绝；norm tolerance `1e-6` |

完整 schema 由
`feature_schema_digest=9ee107f5a9a4d2434fa96265a55bdbee11e6a97c064da741cf6505177ae7ce26`
绑定，并由 validator 核验。

每个 output feature 必须先作为单视频 governed record 产生，并在该记录中同时绑定
冻结的 `impulse_probe_id`、上述 schema digest 和该视频的256维 little-endian
float64 feature bytes。行绑定摘要固定为：

```text
SHA256(
  canonical_json(algorithm, impulse_probe_id, feature_schema_digest)
  || 0x00
  || feature_float64_le_c_order_bytes
)
```

14行输入的 `probe_ids` 必须与冻结 plan 完全同序、无重复、无缺失、无未知项；
`A_actual.probe_ids`、output feature `probe_ids` 和冻结 plan 三者必须完全一致。
禁止在汇总矩阵形成后按位置事后贴标签。若数值行与原始 ID 一起换位，冻结顺序
检查失败；若只换数值而保留原始 ID/摘要，行绑定摘要检查失败。该摘要用于完整性
绑定而非认证；未来 runtime adapter 必须从同一 per-video governed record 传入，
不能由未受信 caller 在 Gate A 前按待检矩阵重新生成。

必须满足：

- 候选密钥无关；
- 在 impulse generation 前冻结；
- 不使用 impulse 样本训练；
- 不读取 owner label；
- 不在结果出现后选择 layer、channel、frequency、pooling 或 normalization；
- confirmation identity 只能 apply，不能重新拟合 extractor。

此时只能称 construction feature，不能称 observer。

### 4.1 真实 Gate A 后的方法决策

真实结果已经停止下列组合继续进入 Gate A：

```text
distributed random 3×2 latent carrier
+ global latent-time mean/L2 Wan VAE feature
```

这不是停止独立低维水印状态、output-only primary 或 Flow-time trajectory 的研究
方向；停止的是该 carrier/feature 组合。固定半幅诊断显示 late six-basis latent
仍有 signed response，但保存视频与 output feature 由 common/even response 主导；
early latent 同时受到后续 feedback 混淆。该证据只支持多候选根因，不支持唯一归因。

下一候选冻结为：

```text
output/decoder-Jacobian-aligned public carrier dictionary
+ key-controlled combination
+ video-time-preserving public feature
```

公开 dictionary 与 public feature 必须在运行前冻结且候选密钥无关；key 只能控制
预冻结 dictionary 的组合系数，不能按结果选择 atom、frame、block、channel、band
或 layer。新 feature 禁止对每个视频单独做 L2 方向归一化；尺度必须来自独立 clean
calibration identity 的 fixed whitening/normalization，并在 held-out identity 上
只 apply。

identity 职责严格分离：

1. construction identity A：定义 public dictionary、clean calibration 与 fixed
   whitening；
2. held-out Gate A identity B：只检验因果 signed observability，不重新拟合；
3. Gate B identity C：只做跨 identity confirmation，不混回 A/B。

状态维数按成本与可证伪性依次推进：

```text
2 stages × 1 dimension
→ 3 stages × 1 dimension
→ 3 stages × 2 dimensions
```

前一项未通过不得扩维。\(F_K/G_K\)、observer、matched-dynamic score 与任何 LLR
仍全部后置于新的 construction A/B/C 证据。

## 5. 端到端传递链

construction observability 必须覆盖：

```text
Flow区间注入
→ 后续生成
→ final latent
→ VAE decode
→ 保存/编码
→ VAE encode
→ output feature
```

分别定义并记录：

- \(T_{\mathrm{latent}}\)
- \(T_{\mathrm{decoded}}\)
- \(T_{\mathrm{saved\_video}}\)
- \(T_{\mathrm{reencoded}}\)
- \(T_{\mathrm{output\_feature}}\)
- \(T_{\mathrm{replay\_diagnostic}}\)

冻结响应表示为：

| checkpoint | 表示 |
|---|---|
| \(T_{\mathrm{latent}}\) | final latent 对六列 basis 的 float32 投影 + latent L2 norm，共7维 |
| \(T_{\mathrm{decoded}}\) | 保存前 RGB float32 的 time-global、4×4 cell mean，float64 累计，共48维 |
| \(T_{\mathrm{saved\_video}}\) | 解码 RGB24/255 的同一48维 summary |
| \(T_{\mathrm{reencoded}}\) | normalized Wan latent 的16×4×4 pooled mean，未做 L2，共256维 |
| \(T_{\mathrm{output\_feature}}\) | 上述256维按冻结 epsilon 做 L2 后的 \(\phi_{\mathrm{construction}}\) |
| \(T_{\mathrm{replay\_diagnostic}}\) | 可选六维 prompt-conditioned basis projection，仅诊断 |

前五项是 primary construction 所需链路。`T_replay_diagnostic` 缺失或失败不得阻断
output-only Gate A，其任何数值也不能支持 positive。Primary Gate A、B、C 的正向
支持只能来自 \(T_{\mathrm{output\_feature}}\)。

## 6. clean 截距与实际设计矩阵

令14个视频的 output features 为
\(Y\in\mathbb R^{q\times14}\)，实际 exposure 为
\(A_{\mathrm{actual}}\in\mathbb R^{6\times14}\)。传递模型固定为：

\[
Y=\mu_0\mathbf1^T+T A_{\mathrm{actual}}+E.
\]

`clean-A`、`clean-B` 在执行前预声明，首轮取：

\[
\hat\mu_0=\frac12(\phi(V_{\mathrm{cleanA}})
                     +\phi(V_{\mathrm{cleanB}})).
\]

主要单列估计使用正负中心差分：

\[
\hat t_k^{\pm}=
\frac{\phi(V_k^+)-\phi(V_k^-)}
     {a_{k,+}^{\mathrm{actual}}-a_{k,-}^{\mathrm{actual}}}.
\]

处理幅度不对称和 cross-channel leakage 时必须使用带 clean 截距的实际矩阵：

\[
\hat T=(Y-\hat\mu_0\mathbf1^T)A_{\mathrm{actual}}^+.
\]

禁止用未去除的公共视频内容作为 transfer column，也禁止用 nominal epsilon 代替
实际 exposure。

## 7. 实际注入与六维压缩条件

generation schedule 固定为 diffusers `0.35.2`
`FlowMatchEulerDiscreteScheduler`、既有 shift=3 signature。8-step sigma grid 为：

```text
[1.0, 0.9475425481796265, 0.8827877640724182,
 0.8008373379707336, 0.6937931180000305,
 0.5480455756187439, 0.33797216415405273,
 0.008928571827709675, 0.0]
```

真实 `delta_sigma`、区间中点 `flow_phase`、step index 0..7 与宏区间 assignment
`[0,0,0,0,1,1,2,2]` 全部进入
`waveform_schema_digest=0d8fc01ce4756cf532f6828aba9c17f77de87a1ac5f1a71e8a63e45ed8a0585d`。
三个 temporal waveform 是各自宏区间内为1、区间外为0的8维预声明向量。

probe polarity 定义在 **state-update coordinate**。由于 Euler update
\(\Delta z_t=\Delta\sigma_t\Delta v_t\)，目标 velocity 符号固定为
\(p\,\mathrm{sign}(\Delta\sigma_t)\)，从而正 probe 的 state exposure 保持为正。
每步 intended norm 唯一为：

\[
n_t^{\rm int}=\min\left(
\|v_t\|_2(0.02)(0.12)|w_t|,
\frac{\sqrt{E^{\rm remaining}_t}}{|\Delta\sigma_t|}
\right),
\]

\[
E^{\rm remaining}_t=
\max\left(0,\;0.000015\,
 [E^{\rm ref}_{<t}+\Delta\sigma_t^2\|v_t\|_2^2N^{\rm remaining}_t]
-E^{\rm control}_{<t}\right).
\]

FP32 backoff 后必须以实际
`constrained.float() - base.float()` 重算 delta norm、六个 basis velocity
coordinates、direction cosine 与能量。实际 signed exposure 不是 norm，而是：

\[
a^{\rm actual}_{t,k}=
\Delta\sigma_t\langle\Delta v_t^{\rm actual},U_{K,k}\rangle.
\]

未来 runtime adapter 必须绑定 digest
`9bc8560cd6ee57be42d064d18b81aa4b3270f2a55f83431dd2cd67e2f4f2017d`，
并重算 reference base norm、remaining energy 和 caller guard；不得信任一个自报
boolean。只有 waveform 精确为零才允许 inactive exact no-op；非零 waveform 下若
base norm、remaining energy、intended control 或实际 FP32 delta 退化为零，必须
结构化 fail-closed。

每一步必须记录：

- signed intended exposure 与 signed actual exposure；
- intended/actual ratio；
- actual delta norm；
- finite-precision projection scale；
- cumulative control energy；
- direction cosine；
- norm/energy guard；
- 正负幅度不对称；
- cross-channel leakage。

只有下列条件全部通过，才允许把逐步变量压缩为六维 interval exposure：

1. trace 精确覆盖上述8步，step/order/phase/delta-sigma/interval/digest 全匹配；
2. actual/intended waveform cosine 不低于 `0.999`；
3. positive 与 sign-inverted negative waveform cosine 不低于 `0.999`；
4. 正负幅度不对称不高于 `0.05`；
5. cross-channel leakage ratio 不高于 `0.05`；
6. 每一步 direction/norm/energy guard 均通过；
7. \(A_{\mathrm{actual}}\) rank 为6且 condition number 不高于20。

任一条件失败只能保留逐步设计变量作为诊断并立即停止 Gate A；函数不得返回可继续
使用的六维 design，禁止把六维 summary 宣称为充分。
signed exposure 不得由无符号 norm 替代。probe 使用
`lambda_max=0.12` 以及既有 norm/Flow-energy budget，不得为了可见性越过未来正式方法
允许预算。

## 8. observer 边界

本合同不实现：

- \(F_K/G_K\)；
- Kalman、RTS 或 batch observer；
- 正式 LLR；
- owner/wrong 方法有效性 smoke；
- attack、pilot、fixed-FPR 或 baseline。

只有三个 construction gates 全部通过，才允许单独设计 \(F_K/G_K\) 与 batch
observer。首个 score 只能命名为：

```text
prediction-error score
或
matched-dynamic score
```

不得称正式 LLR。未来声称动力学贡献必须同时比较：

1. complete observer；
2. independent interval template；
3. static endpoint。

没有这三个消融，不得把阶段顺序可辨识写成状态动力学贡献。

## 9. 历史冻结点授权状态

本节只记录该历史冻结点当时允许的只读/CPU 分析边界：

```text
验证完整 f06a0934 六视频 result
→ CPU-only 逐帧/固定三区间/相邻差分 signed-response 分析
→ 冻结下一次 frozen-feedback 与 carrier/feature redesign 规范
→ 独立只读审核
```

该历史冻结点任务不允许提交、推送、GPU、Colab、Drive 或 Notebook 变更。本地
pytest/harness 通过只证明合同和执行入口一致，不证明 construction observability
或方法有效性。CPU 分析不得自动选择新 feature；真实 Gate A FAIL 保持不变。
任何 frozen-feedback 执行、Jacobian-aligned dictionary construction 或新 identity
运行都必须另行实现、审核并由用户授权。该段不授予现行主路线的 config、runner、
observer、GPU、Colab、Drive 或 Notebook 操作权限。

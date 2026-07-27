# Patch-relation Gate 0 本地合同与算法原语

状态：`local_patch_relation_gate0_primitives_only`

本文件从属于两份现行权威文档，冲突时
`frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`
优先于 method design。本批只实现可由 CPU/NumPy 验证的首个 Patch-relation
写入与 output feature 原语；没有 runner、GPU、Colab、Notebook、Drive 或模型
执行授权。

## 1. 证据边界

真实 commit `346d6c97bbfdef5d3ff0e61ebb6f69b1e7b6cea3` Gate 0 只停止：

```text
decoder-Jacobian additive atom
+ local RGB mean feature
+ held-out transfer
```

它没有实现或否定 3D-RoPE Patch relation、output Patch-relation feature、
PRC/state dynamics、clock path 或 observer。generic public low-frequency carrier
bank 只允许作为待审 baseline/fallback，不是本合同的 dictionary、carrier 或结果。

## 2. v0.35.2 Wan 写入点

官方 diffusers v0.35.2 `WanTransformer3DModel.forward` 的实际顺序是：

```text
rotary_emb = self.rope(hidden_states)
hidden_states = self.patch_embedding(hidden_states)
hidden_states = hidden_states.flatten(2).transpose(1, 2)
for block:
    block(..., rotary_emb)
```

`WanRotaryPosEmbed.forward` 返回 `(freqs_cos, freqs_sin)`，shape 为
`[1,T*H*W,1,attention_head_dim]`。`WanAttnProcessor` 在 self-attention 内把同一
tuple 作用于 query 与 key；cross-attention 不使用该 tuple。对 Wan2.1 1.3B 的
冻结输入，patch size 为 `[1,2,2]`，latent `[1,16,9,40,64]` 映射为
`[T,H,W]=[9,20,32]`、5760 tokens、head dim 128。非 MPS 路径的 RoPE buffer
是 float64。

所以首个且唯一主写入原语冻结为：

```text
scoped_transformer_rope_output_temporal_phase_pair
```

未来 adapter 只能在 `self.rope(hidden_states)` 输出后、其 tuple 进入全部
self-attention block 前，变换冻结 token 与 temporal RoPE pair。它不能改
attention processor、不能注入完整 attention-bias 张量、不能回退 direct
latent/velocity addition。当前 adapter 尚未实现。

## 3. 唯一公开 relation

### 3.1 token 与视频映射

- saved video：33 帧、320×512 RGB24；
- Wan latent：`[1,16,9,40,64]`；
- Conv3D patch：`[1,2,2]`；
- token grid：`[9,20,32]`；
- flatten：latent time、patch row、patch column，column 最快；
- output video window：frames `[11,22)`；
- 对应 latent token time：`[3,4,5]`，即中心 frames 12/16/20；
- 每个 token patch 对应保存视频的 16×16 pixel patch。

唯一 Patch pair 为：

```text
P_a = token(row=9, column=13)
P_b = token(row=9, column=18)
```

每个 active token time 上，公开系数分别为 `+1,-1`，其余为0。C-order 首个
active coefficient 必须为正，且每个 active time 严格 zero-sum。dictionary
由上述规则唯一重建，冻结 descriptor digest 为
`6580aa413ff02197b31c7878389a4d038f50b6ecba5b241cc74a8ff5dcce979a`，
不读取 key、prompt、seed、identity 或视频结果。

### 3.2 phase 写入

head dim 128 在 v0.35.2 中拆成 temporal/spatial dimensions `44/42/42`。
本合同只使用 temporal RoPE pair index 0，即 tuple 的 head entries `[0,1]`。
单一 phase budget 为 `0.015625` radians，不做 strength sweep。

对公开 pair coefficient \(a_p\in\{-1,0,+1\}\) 和 signed state coefficient
\(c\in\{-1,0,+1\}\)：

\[
\Delta\theta_p=0.015625\,c\,a_p.
\]

positive/negative 的 phase delta 必须逐元素严格互为负数；clean 必须走同一函数
并得到 exact-zero delta。对官方实数 tuple：

\[
\begin{aligned}
\cos'&=\cos\cos\Delta\theta-\sin\sin\Delta\theta,\\
\sin'&=\sin\cos\Delta\theta+\cos\sin\Delta\theta.
\end{aligned}
\]

这只定义 phase 原语，不声称模型输出已经形成反对称响应。

### 3.3 key/context 派生

公开 descriptor 不随候选 key 改变。单 signed coefficient 用：

```text
HMAC-SHA256(
  master_key,
  "sstw.patch_relation_gate0.coefficient.v1"
  || 0x00
  || context_digest
  || descriptor_digest
)
```

首 byte 的最低 bit 为1映射 `+1`，否则 `-1`。完整 HMAC digest 记录域分离。
由于 Gate 0 只有一个 signed dimension，不同 key 可以偶然得到相同 sign；
本合同不把单 bit sign 当 wrong-key selectivity 证据，也不授权 wrong-key Gate。

## 4. output-side Patch-relation feature

输入严格为保存视频回读的 C-contiguous
`uint8[33,320,512,3]`，禁止 resize。使用和 token pair 对齐的两个16×16 pixel
patch。对每个 frame `[11,22)`、每个 patch 和 RGB channel，计算正交 DCT-II
`(vertical,horizontal)=(0,1)` 与 `(1,0)` 两个 signed coefficients，分量顺序
严格为 horizontal `(0,1)` RGB，随后 vertical `(1,0)` RGB，然后输出：

\[
q_{n}=\mathrm{DCT}(P_a)-\mathrm{DCT}(P_b)\in\mathbb R^6.
\]

最终 schema 是 little-endian `float64[11,6]`。11帧时间轴必须保留，不做全视频
时间平均，不做逐视频 L2，不使用4×4 RGB mean 作为 primary，也不依据 key 或
Gate identity 选择 patch、channel、frequency 或 frame。

## 5. C0 construction 与 identity A

本批 identity 仍是 `_placeholder`，因此执行禁止。未来必须冻结不同的 C0/A
prompt、seed、initial noise 与用途；A 不得回流 C0。

C0 精确四项：`clean_a, clean_b, positive, negative`。C0 只冻结：

1. public relation descriptor；
2. feature schema；
3. clean center 与 elementwise scale；
4. restricted end-to-end `T_rel`。

whitening center 是 C0 clean A/B 的逐坐标均值；scale 是
`max(abs(clean_a-clean_b)/sqrt(2),1e-6)`。实际 signed exposure 必须由未来
governed runtime adapter 的 realized relation control 提供；当前本地原语只接受
有限、正负方向正确的 caller scalar 做公式验证。它不是已冻结的 runtime exposure
record，不能形成 execution evidence。定义：

\[
T_{\mathrm{rel}}
=
\frac{q_+^{w}-q_-^{w}}{e_+-e_-}.
\]

这是公开 relation atom 经真实生成、decode、保存与 feature 后的受限
`T_rel`，不是独立识别的高维 `H_e2e`。C0 readiness 不是 Gate 0 通过证据。

identity A 也固定四项，只应用 C0 的 descriptor、feature、whitening 与 `T_rel`。
A 可以用自身 clean A/B 在冻结 whitening 坐标内形成 identity intercept，但不得
重选或重拟合任何构造对象。预测：

\[
\widehat q_{\mathrm{odd},A}
=T_{\mathrm{rel}}\frac{e_{+,A}-e_{-,A}}2.
\]

## 6. 最小门禁

所有 `11×6` 数组均为 little-endian float64、C-order；norm 与 dot 在 C-order
flatten 后计算。对冻结 whitening 后的 feature：

\[
\begin{aligned}
\mu&=(q^w_{0a}+q^w_{0b})/2,\\
\delta_+&=q^w_+-\mu,\qquad \delta_-=q^w_- - \mu,\\
q_{\rm odd}&=(\delta_+-\delta_-)/2,\\
q_{\rm common}&=(\delta_++\delta_-)/2.
\end{aligned}
\]

clean noise 为
`max(||q_clean_a^w-q_clean_b^w||_2,1e-6)`；odd/common norm 分别直接取
float64 L2。antisymmetry cosine 使用 `cos(delta_+,-delta_-)`，任一向量零
norm 时返回0；只允许 `1e-12` Cauchy machine-roundoff clamp，超出
`[-1-1e-12,1+1e-12]` 即拒绝。residual 分母为
`max(||delta_+||+||delta_-||,1e-12)`；common/odd 分母为
`max(||q_odd||,1e-12)`；odd/noise 直接除上述有限 clean noise。

identity A 的 predicted odd 精确为
`T_rel * 0.5 * (positive_exposure-negative_exposure)`；transfer cosine 使用同一
safe-cosine 规则，relative error 分母为
`max(||observed_odd||_2,1e-12)`。随后才应用门限：

```text
antisymmetry cosine >= 0.9
antisymmetry residual <= 0.25
common / odd <= 0.5
odd / clean-noise >= 3.0
T_rel prediction direction cosine >= 0.9
T_rel prediction relative error <= 0.5
```

C0 signed gate 只决定构造是否可冻结；identity A 全部门禁通过才是 Gate 0
diagnostic readiness。即使通过，也最多允许另行设计双窗口 Gate A，不能自动
执行。

## 7. 状态机与禁止事项

```text
local contract + NumPy primitives
-> independent read-only audit
-> possible commit/push authorization
-> separate future runtime-adapter design
-> separate user-authorized GPU run
```

当前始终：

```text
formal_result=false
stage_progression_allowed=false
runtime_implementation_authorized=false
gpu_execution_allowed=false
colab_execution_allowed=false
runner/notebook/Drive=false
observer/attack/fixed-FPR/baseline/paper claim=false
```

本地 pytest/harness 只证明合同与算法原语自洽，不是 Patch-relation 方法结果。

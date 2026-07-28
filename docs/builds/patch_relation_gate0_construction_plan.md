# Patch-relation Gate 0 真实 runner 合同

状态：`patch_relation_gate0_runner_implemented_pending_user_colab_run`

本文件从属于两份现行权威文档，冲突时
`frame_state_synchronized_generative_flow_video_watermark_algorithm_primitives.md`
优先于 method design。本批把已审核 CPU/NumPy 原语与 Wan RoPE adapter 接成
最薄8视频 runner，并接入固定 `colab_test` 请求入口。实现完成不等于已运行；
真实 GPU/Colab 仍只由用户显式启动，Notebook 不承载方法逻辑。

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
`[T,H,W]=[9,20,32]`、5760 tokens、head dim 128。非 MPS 路径虽然以
float64 构造 frequency grid，但 v0.35.2 `get_1d_rotary_pos_embed` 在 real
cos/sin 返回前显式 `.float()`；因此 `WanRotaryPosEmbed.forward` 的 tuple
storage dtype 精确为 float32。

所以首个且唯一主写入原语冻结为：

```text
scoped_transformer_rope_output_temporal_phase_pair
```

当前本地 adapter 只在 `self.rope(hidden_states)` 输出后、其 tuple 进入全部
self-attention block 前，变换冻结 token 与 temporal RoPE pair。它不改
attention processor、不注入完整 attention-bias 张量、不回退 direct
latent/velocity addition。scope 精确包围一次 transformer branch forward：
正常和异常退出都恢复原 `rope.forward` 与 scope state；禁止嵌套，禁止一支
scope 内多次成功调用。

branch record 只允许在 body 无异常、原始 `forward` 与 scope state 清理完成、
且 attempt/success 均精确为 1 的 clean exit 后形成。RoPE 已成功但后续
patch embedding/attention/transformer 失败，或 cleanup 自身失败，都永久拒绝
该 scope 的 record，不能仅凭 RoPE 调用计数进入 CFG pair。

v0.35.2 `WanPipeline` 在 guidance=5 时按：

```text
conditional transformer
-> unconditional transformer
-> uncond + 5*(cond-uncond)
-> scheduler.step
```

执行。因此相同 probe/step 的 relation coefficient 必须同时、同值作用于
conditional 与 unconditional 两支；base forward 的两支都使用 zero control。
两支必须绑定同一输入 state/timestep/probe digest，禁止只控制一支而改变 CFG
语义。真实 torch 路径要求 hidden state 为 contiguous BF16
`[1,16,9,40,64]`，RoPE tuple 为同 device、contiguous float32
`[1,5760,1,128]`，并且只允许官方 no-grad inference。公开 phase delta
继续使用 float64 数学 schema；实际旋转稳定计算后必须 cast 回原 float32
tuple storage，不能把 tuple 升格为新的运行时 dtype。

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

## 5. C0 construction、governed schedule 与 identity A

冻结身份为：

- C0：`probe_paper_paper_master_prompt_005` /
  `probe_paper_paper_master_calibration_seed_03=1275`；
- identity A：`probe_paper_paper_master_prompt_006` /
  `probe_paper_paper_master_test_seed_03=2275`。

两者 prompt 文本、negative prompt、SHA-256 与 seed 均进入 protocol digest。
每项 probe 都重新构造同 seed generator；同 identity 的4项 initial noise 必须
相同，C0/A 必须不同。历史 decoder-Jacobian 结果不得回流选择 relation、
feature、threshold 或身份。

C0 精确四项：`clean_a, clean_b, positive, negative`。C0 只冻结：

1. public relation descriptor；
2. feature schema；
3. clean center 与 elementwise scale；
4. restricted end-to-end `T_rel`。

whitening center 是 C0 clean A/B 的逐坐标均值；scale 是
`max(abs(clean_a-clean_b)/sqrt(2),1e-6)`。本地 adapter 现在可以从同一输入绑定
的 base/controlled conditional 与 unconditional transformer 输出重算：

```text
base_cfg = base_uncond + 5*(base_cond-base_uncond)
controlled_cfg = controlled_uncond + 5*(controlled_cond-controlled_uncond)
intended_delta = controlled_cfg-base_cfg
controlled branch outputs = float32(real BF16 transformer outputs)
constrained = controlled_cfg  # pipeline与scheduler实际消费的同一FP32数组
actual_delta = constrained-base_cfg
base_next = frozen_euler_step(sample,base_cfg)
controlled_next = real_scheduler_returned_prev_sample
actual_state_update_delta = controlled_next-base_next
```

以上全部在 C-contiguous FP32 scheduler coordinate 内完成；runner 在模型加载
前必须用 `torch.cuda.is_bf16_supported(including_emulation=False)` 验证原生
BF16、验证CUDA compute capability major至少为8，并确认pipeline selected dtype
exact为`torch.bfloat16`；旧torch签名、仿真BF16或FP16选择均fail-closed。
runner不会让BF16 branch-combine舍入成为scheduler实际路径，并冻结逐step真实
8-step sigma grid：

```text
1.0, 0.9475425482, 0.8827877641, 0.8008373380,
0.6937931180, 0.5480455756, 0.3379721642, 0.0089285718, 0.0
```

每个 `delta_sigma=next-current` canonicalize 为 float32，并与 scheduler 暴露的
实际 grid 精确一致；实际传入scheduler的timestep还必须逐项等于
`scheduler.timesteps[step_index]`和冻结的`sigma*1000`行。scheduler内部
step index在调用前后必须分别为冻结的当前/下一行；重复、错位timestep或内部
index漂移均fail-closed。remaining count 固定为 `8-step_index`。runner 从 step 0
开始顺序累计 reference/control energy，不接受 caller 自报 remaining budget。
norm budget 沿用
`||base_cfg||*0.02*0.12`，Flow energy increment 为
`||controlled_next-base_next||^2`；reference increment同样从实际
`base_next-sample`重算。两条next-state还必须与冻结Euler FP32公式exact一致，
否则不得形成记录。本地边界从未来 governed adapter 提供的
cumulative reference/control energy 与正 remaining-step count 重算 projected
reference、总预算和 remaining energy，不接受任意 caller remaining-budget。
signed active step 在真实 scheduler 调用前执行 phase-domain bounded
projection。maximum phase 仍为 `0.015625`，不是可调 strength。每个当前
state/timestep/CFG branch 输入只计算一次 base conditional/unconditional；
随后对同一 scale 的 `+phase/-phase` 都做真实 RoPE-controlled Wan
conditional/unconditional re-forward，以两符号最坏 norm/energy usage 选共同
scale。禁止把 full-phase velocity 线性缩放，禁止退回 direct-additive
velocity projection。full scale先评估；失败后只按

```text
next_scale =
  current_scale
  * min(norm_budget / worst_actual_delta_norm,
        sqrt(remaining_energy / worst_energy_increment))
  * 0.9
```

候选搜索必须延后到scheduler wrapper已经取得官方pipeline保留的真实FP32
`sample`之后；transformer收到的BF16 hidden state只用于同输入绑定，禁止
BF16转回FP32后冒充scheduler sample。确定性 backoff，最多4次candidate、
minimum nonzero scale为`0.000001`，无
最大化细化轮。direction/nonfinite失败或无非零可行scale立即fail-closed。
rejected candidate不得调用scheduler、推进内部index或形成governed record；
选中后scheduler精确调用一次，且只消费当前probe sign的已验证CFG。scale选择
不直接读取请求sign，但positive/negative属于不同历史trajectory时，当前state
不同，允许得到不同scale；actual exposure仍逐step记录。
每个sign evaluation只能由raw branch velocity、真实FP32 sample和冻结能量
上下文重算后签发同进程一致性 capability；正负evaluation必须共享probe、
step、input、sample、base与能量上下文。selection绑定这两个capability，最终
measurement本身也只能由同一raw四分支factory签发，并将创建时base/control
raw digest及全部字段纳入同进程identity capability；手工构造、replace或在
promotion时把raw-A measurement与raw-B数组调包均必须先于统计重建被拒绝。
首个candidate还会冻结完整shared-context tuple，所有后续backoff attempt
必须与其逐项一致，只允许scale、controlled pair/output及派生响应变化。最终
当前sign evaluation还必须在四个raw cond/uncond branch数组仍存活时现场重算
base raw digest、controlled raw digest和candidate context；即使不同raw
branches在guidance合成后得到完全相同CFG也必须fail-closed。随后再与真实
scheduler返回transition的norm、energy、remaining、direction、guard和数组
digest逐项一致，才可与既有transition seal共同进入compact record。无可行解
诊断中的last scale必须是最后实际评估的scale。上述capability不是抵抗恶意
同进程代码的密码学认证。

direction guard 比较真实next-state差与
`delta_sigma*intended_delta`，阈值保持0.999，不放宽预算。clean 必须 exact
zero actual delta，并完全跳过projection search；active 必须严格非零。

用于 C0/A 的 signed state-update exposure 冻结为：

\[
e
=
\operatorname{sign}(c)\,
\|x^{controlled}_{next}-x^{base}_{next}\|_2.
\]

它使用scheduler返回值绑定的actual FP32 state update而不是名义 phase或仅
velocity公式。局部 measurement 仍明确不是execution evidence；只有 runner
同时验证冻结 schedule/timestep/internal index、cond/uncond两支、
base/controlled同输入、scheduler实际消费的 controlled CFG与返回next-state、
累计能量和完整8-step coverage 后，才提升为本次 non-formal construction
step record。每步提升前，统一validator必须直接消费该步
`base_cfg/controlled_cfg/sample/base_next/controlled_next`的实际FP32数组，
重算所有velocity/state-update norm、direction、energy、exposure和guard；
随后只签发进程内、不可序列化的validated seal并立即释放大数组。batch验证同时
要求seal仍由同进程签发且所有sufficient statistics未变，不能通过协调修改
scalar、budget、guard、record或无密钥digest绕过。五个scheduler/transition
digest也必须由该factory直接从已验证数组的C-order bytes重算，不接受caller
参数。此seal仅是正常caller与`dataclasses.replace`场景的同进程一致性
capability，不是抵抗能够任意monkeypatch Python模块私有状态的密码学认证。
定义：

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

## 7. runner、Colab 与禁止事项

```text
local contract + NumPy/runtime adapter
-> reviewed runner + exact schedule/identity binding
-> independent read-only audit
-> possible commit/push authorization
-> user updates request.ref and runs the unchanged thin Notebook
-> local /content run completes
-> one result ZIP + one minimal manifest copied to Drive
```

runner 对每个 denoising step 按官方 `conditional -> unconditional` 外部顺序。
clean 为每支`base zero -> controlled zero`，共4次 transformer forward并跳过
search。active先做2次base forward，再对每个candidate做正负各两支共4次真实
controlled forward：首candidate总计6次，冻结4-attempt上界为18次。cache必须
关闭，避免同一outer cache context污染base/candidate。正式scheduler只消费
selected/current-sign controlled CFG；base只用于同输入counterfactual测量。

精确视频顺序为：

```text
C0 clean-A, clean-B, positive, negative,
A  clean-A, clean-B, positive, negative
```

每视频形成8条step record、一个保存视频SHA与一个`11x6` feature record；完整
coverage 为8 generation / 64 step / 8 feature。C0只拟合 whitening/T_rel，
A严格apply-only。方法 Gate FAIL 是正常的 non-formal stop 并可打包；runtime
或合同异常进入既有 recovery-only 路径。成功输出只在 `/content` 本地完整后，
由现有 packager 复制单 ZIP + manifest 到 Drive。

当前授权：

```text
formal_result=false
stage_progression_allowed=false
runtime_implementation_authorized=true
runner_implementation_allowed=true
construction/gpu/colab execution=true
notebook handler/Drive direct write=false
observer/attack/fixed-FPR/baseline/paper claim=false
```

这些 true 仅表示提交后可由用户显式运行已审核入口，不表示本轮已启动 GPU。
本地 pytest/harness 只证明合同、runner与adapter边界自洽，不是 Patch-relation
Gate 0 或方法结果。真实 Wan hook、CUDA峰值和端到端输出仍待用户 Colab 首跑。

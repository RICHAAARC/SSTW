# Gate A Root-Cause Amplitude/Feedback Diagnostic

## 1. 目的与非目标

该诊断只定位首次 output-feature impulse Gate A 失败的可能来源。它比较：

- 已冻结且只读的 `lambda=0.12` Gate A FAIL；
- 一个预声明的 `lambda=0.06` 六视频 counterfactual。

它不是 Gate A 重试，也不能产生 Gate A PASS、方法有效性、跨 identity、owner/wrong、
observer、攻击、fixed-FPR、baseline 或论文证据。分类允许多因并存或
`indeterminate`，禁止声称唯一根因。

完整配置由代码内预冻结 canonical JSON digest
`e1a89a4ebc3d30d9007c49ad1b6532da0ee55d53959bc41de24add1c6ca57a43`
绑定。所有 threshold、formula、checkpoint ID/storage/shape、source/base identity、
clean role、claim status 和 authorization 字段任一改变都在执行前 fail-closed；
输出中记录自身 digest 不能替代该常量校验。

## 2. 冻结的六视频计划

顺序精确固定为：

1. `clean_start`
2. `positive_early_flow_channel_0`
3. `negative_early_flow_channel_0`
4. `positive_late_flow_channel_0`
5. `negative_late_flow_channel_0`
6. `clean_end`

prompt003、seed2201、initial generator state、Wan revision、8-step Flow scheduler、
33帧、512×320、guidance 5、fps 8、exporter、basis KDF、三个 Flow macro intervals
均与 commit `47485be2b6f734014b74f73e797174b911d2aeb5` 的14视频 construction
一致。每个视频重新以 seed2201 构造 generator；`clean_end` 只检查顺序漂移或
pipeline 状态污染，两个 clean 不能冒充正式噪声分布。

只新增单一半幅 `lambda=0.06`。既有 `.12` 结果作为固定第二幅度点，禁止重跑
`.12`、扩展 strength/grid/identity/channel 或依据结果增加幅度。

## 3. 实际控制与历史输入绑定

半幅运行继续复用同一 canonical float32 basis、实际
`(constrained.float-base.float)` delta、bounded backoff、velocity norm budget、
Flow energy budget 和 direction cosine guard。`lambda=.06` 直接进入每步
`compute_intended_impulse_control`，不是对 `.12` record 事后乘半。非零 waveform
若不存在可表示的非零控制仍 fail-closed。

Colab request 的 `source_package_path` 必须指向完整 `.12` Gate A result ZIP。
handler 在 `/content` 安全解压后校验：

- manifest、decision、profile、commit 和 FAIL identity；
- config、feature schema、waveform、runtime adapter 与 basis digest；
- 14 generation、112 steps、12 traces、70 checkpoints、14 features；
- exact plan/row order、record IDs、feature row bindings；
- traces 重建的 `A_actual` 与 Gate A FAIL；
- 14 个 packaged MP4 的 SHA-256；
- `formal_result=false` 与 `stage_progression_allowed=false`。

recovery/partial/PASS 包均被拒绝。历史 manifest 中旧的绝对 `/content` 路径不作为
输入路径；只使用当前安全解压根下按 packaged basename 唯一找到的文件。
诊断记录一个内部一致的 source content snapshot digest；这不是外部签名或真实性认证。

## 4. Checkpoints 与显存边界

五个可与历史 `.12` 比较的表示为：

- `T_latent_six_basis`：只取 `T_latent` 前六个 basis projection；
- `T_decoded_48d`；
- `T_saved_video_48d`；
- `T_reencoded_256d`；
- `T_output_feature_256d`。

天然 sign-even 的 latent L2 norm 必须单独记录，禁止纳入 signed antisymmetry。
新 `.06` 还把每视频 full final latent 立即转为 CPU float32 压缩 artifact，并对
full latent、full pre-save float32 frames、实际回读的
`33×320×512×3 uint8 RGB24` 保存 Gram sufficient statistics。pre-save 和 saved
临时数组只存在于 `/content` working directory，完成 float64 chunked Gram 后删除；
GPU 不累积跨视频 tensor。历史包没有 full final latent 或 full pre-save frames，
因此相应跨幅比较必须明确标为 unavailable；历史 MP4 可真实回读 full RGB24。

## 5. Odd/Common 统计

每个 amplitude 使用自己的两个 clean 截距：

\[
\mu=\frac{y_{\mathrm{clean,start}}+y_{\mathrm{clean,end}}}{2},
\quad d_+=y_+-\mu,\quad d_-=y_--\mu.
\]

\[
\mathrm{odd}=\frac{d_+-d_-}{2},\qquad
\mathrm{common}=\frac{d_++d_-}{2}.
\]

每层报告：

- `||odd||`、`||common||`、`||common||/||odd||`；
- `cos(d_+,-d_-)`；
- `||d_++d_-||/(||d_+||+||d_-||)`；
- clean start/end distance。

full tensor/pixel 表示由冻结 row order 的 `6×6` float64 Gram matrix计算同一二次型；
Gram 记录绑定 row IDs、源数组/video digest、shape/dtype 与 chunk size。

## 6. 预冻结解释阈值

先计算 early0/late0 的实际 exposure 平均幅值比
`q = amplitude_.06 / amplitude_.12`。只有 `q∈[0.45,0.55]` 才允许比较幅度规律；
否则该 checkpoint 为 `indeterminate`。

局部线性 support candidate 需要 output feature 上 early0 和 late0 同时满足：

- half/full odd ratio ∈ `[0.35,0.65]`，理想值 `0.5`；
- half/full common ratio ∈ `[0.125,0.375]`，二阶理想值 `0.25`；
- antisymmetry cosine 至少改善 `0.10`；
- residual 至少降低 `0.10`，且不高于历史 residual 的 `0.75`。

这些窗口来自一阶 odd、二阶 even/common 的 Taylor scaling，只是预声明的根因判别
effect-size，不是方法 gate。

quantization/observation-floor candidate 仅表示 half odd 相对历史发生
`≤0.25` collapse 或 `≥0.75` plateau；它不能单独证明 BF16 是根因。

feedback candidate 由 early 相对 late 的 latent six-basis 响应明显恶化支持：
antisymmetry cosine gap ≥`0.20`、common/odd ratio factor ≥`1.5`，或
exposure-normalized early odd transfer ≤ late 的 `0.5`。它最多允许设计后续
frozen-feedback diagnostic，不能在本轮唯一证明 feedback 因果。

carrier/decoder-feature mismatch candidate 要求 late latent six-basis 仍满足
cosine≥`0.9`、residual≤`0.25`、common/odd≤`0.5`，但 decoded/saved/output 任一
common/odd≥`1.0` 或 antisymmetry cosine<`0.9`。

若 `clean_start` 与 `clean_end` 任一冻结表示不一致，均值截距和按运行顺序采集的
正负 pair 已被顺序漂移混淆。此时所有 amplitude/feedback/mismatch 候选都必须
失效，分类固定为 `generation_order_state_pollution_candidate + indeterminate`，
并关闭两项后续设计授权。

## 7. 决策边界

无论分类结果如何：

- `historical_gate_a_pass=false`
- `gate_a_retry=false`
- `gate_a_pass=false`
- `formal_result=false`
- `stage_progression_allowed=false`
- Gate B/C、wrong-key、observer、F/G、LLR、attack、pilot、fixed-FPR、baseline、
  paper claim 全部关闭。

输出最多允许
`frozen_feedback_diagnostic_design_allowed=true` 或
`carrier_feature_redesign_allowed=true`；任何后续执行仍需独立设计、审核和用户授权。

## 8. Colab 边界

固定 `colab_test_runner.ipynb` 不变。Notebook 仍只调用 server CLI；新逻辑位于独立
runner 与白名单 handler。历史 ZIP 从 Drive 复制到 `/content` 后解压，模型下载到
`/content` cache，六视频、VAE re-encode、统计和打包全部先在 `/content` 完成。
成功后才把单个 result ZIP 与最小 manifest 写回 Drive；失败仍走现有
recovery-only 包，不能当作诊断结论或方法证据。

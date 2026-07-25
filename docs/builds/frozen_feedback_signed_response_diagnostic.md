# Frozen-Feedback Signed-Response Diagnostic Specification

## 1. 状态

本文只冻结下一次最小实验的方法合同，不实现 runner、handler、Notebook 或 GPU
执行。任何执行仍需独立实现、代码审核和用户授权。

它不是 Gate A 重试。所有未来产物仍必须：

```text
formal_result=false
stage_progression_allowed=false
gate_a_pass=false
```

## 2. 精确五输出

沿用 f06a0934 的 prompt003、seed2201、Wan revision、scheduler signature、8 steps、
33帧、512×320、guidance 5、fps 8、owner basis 与 `lambda=.06`。只允许：

1. `clean`
2. `positive_early_flow_channel_0`
3. `negative_early_flow_channel_0`
4. `positive_late_flow_channel_0`
5. `negative_late_flow_channel_0`

不得增加 identity、channel、strength、grid、clean repeat 或 composite。

## 3. 共享 clean base-velocity trace

只对 clean trajectory 调用冻结生成模型，记录每步：

\[
(z_t^{clean}, v_{\theta,t}^{clean}, \Delta\sigma_t).
\]

四个 signed counterfactual 从完全相同 initial latent 开始，并共享该 clean
base-velocity/reference-energy trace。每步 intended control、FP32 backoff、
direction/norm/Flow-energy guard 均按 clean trace 重算；正负仍定义在
state-update coordinate。

counterfactual 更新唯一为：

\[
z_{t+1}^{p}
=z_t^{p}
+\Delta\sigma_t
  (v_{\theta,t}^{clean}+\Delta v_{p,t}^{actual}).
\]

干预开始后禁止用 \(z_t^p\) 再调用模型，禁止重新评估
\(v_\theta(z_t^p,t,prompt)\)。因此 signed 输出之间的差异只来自固定 additive
control，不包含后续 model feedback。

五项均须完成 final latent、同一 VAE decode、同一保存编码、RGB24 回读、同一
VAE re-encode 与冻结 public diagnostic summaries。不得把未保存 latent 当作
output-only positive。

## 4. 分支判别

阈值沿用 signed response gate：

- cosine `>=0.9`；
- residual `<=0.25`；
- common/odd `<=0.5`；
- odd response 高于预冻结数值下限。

判定顺序和候选集合冻结如下。`latent_all_signed` 表示 early 与 late 的 full final
latent 均通过上述 signed gate；`latent_all_failed` 表示二者均未通过；
`post_latent_all_signed` 表示 early/late 的 decoded、saved RGB24、reencoded 与
output diagnostic summary 全部通过。

| clean/coverage/guards | full latent | post-latent | 唯一允许输出 |
|---|---|---|---|
| 任一不完整 | 任意 | 任意 | `indeterminate_stop` |
| 完整 | `latent_all_signed` | `post_latent_all_signed` | 仅 `feedback_isolation_candidate` |
| 完整 | `latent_all_signed` | 任一 post-latent 失败 | `feedback_isolation_candidate` + `decoder_carrier_mismatch_candidate` + `multiple_candidates` |
| 完整 | `latent_all_failed` | 未全部通过 | `stop_current_additive_random_carrier` |
| 完整 | early/late latent 结果混合，或与 post-latent 谓词矛盾 | 任意 | `indeterminate_stop` |

第一条 signed latent 恢复只支持“关闭 feedback 后 response 恢复”的 isolation
候选，不唯一证明 feedback 是根因。latent 恢复但 post-latent 失败时两个机制候选
必须并存，禁止选择唯一 primary。所有分支固定
`unique_root_cause_claim_allowed=false`。

不得由结果降低阈值、改变幅度、选择单帧/单块、追加 sixth video 或直接实现新
feature。

## 5. 授权边界

该实验即使未来完成，也不授权 Gate B/C、wrong-key、observer、\(F_K/G_K\)、
state dynamics、attack、fixed-FPR、baseline 或 paper claim。其唯一用途是选择：

```text
继续研究 feedback isolation
或
停止当前 additive random carrier 并进入
output/decoder-Jacobian-aligned public dictionary 设计
```

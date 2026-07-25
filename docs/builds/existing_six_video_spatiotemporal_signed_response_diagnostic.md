# Existing Six-Video Spatiotemporal Signed-Response Diagnostic

## 1. 目的

该诊断只回答“下一次最小 GPU 实验应测 feedback 还是 redesign”。它读取 commit
`f06a0934...` 已生成的六个保存视频，不生成视频、不调用模型、不读取密钥，也不
重试 Gate A。

输入必须是完整 result ZIP 或目录，且精确绑定：

- source commit/config/profile/decision；
- 只读历史 Gate A source snapshot `9797f2ac...`；
- 当前26文件 snapshot `4d5ccdc5...`；
- 6 generation、48 step、4 exposure、30 checkpoint、6 feature；
- 六条 plan/video SHA 与相同 initial generator state；
- clean_start/end 在全部冻结表示中一致；
- `gate_a_pass=false`、`formal_result=false` 与全部后续授权关闭。

recovery、partial、其它 profile、额外文件、重排或任一 byte tamper 均 fail-closed。
ZIP 在写出任何 member 前按规范化 target 去重，并拒绝等价路径、file/directory
冲突、空路径、symlink、越界和规模超限。directory source 与 output 必须双向
隔离：二者不得相等，也不得互为祖先/后代；失败 artifact 不能写回 source。

## 2. 冻结分析轴

RGB24 输入必须恰好为 `[33,320,512,3] uint8`，只做固定
`float64(pixel)/255`。clean 截距为 clean_start/end 算术均值；两者像素不完全相同
时全部结论停止。

对 early0 与 late0 两个正负 pair 同时计算：

1. 33个逐帧完整 RGB 向量；
2. 三个固定帧区间 `[0,11)`、`[11,22)`、`[22,33)`；
3. 32个固定相邻帧差分；
4. 各帧区间内部的相邻差分聚合 `[0,10)`、`[11,21)`、`[22,32)`。

不分析4×4 block，不允许运行后选择 frame、block、RGB channel、frequency band、
threshold 或 candidate key。

## 3. signed odd/common

每个冻结表示使用：

\[
d_+=V_+-\mu_0,\qquad d_-=V_- -\mu_0,
\]
\[
\mathrm{odd}=\frac12(d_+-d_-),\qquad
\mathrm{common}=\frac12(d_++d_-).
\]

记录 positive/negative centered norm、odd/common norm、common/odd、
\(\cos(d_+,-d_-)\) 与
\(\|d_++d_-\|/(\|d_+\|+\|d_-\|)\)。全部 reduction 使用 float64，非有限值
fail-closed。

signed gate 沿用原合同：

- antisymmetry cosine `>=0.9`；
- residual `<=0.25`；
- common/odd `<=0.5`；
- odd norm `>=1e-12`。

单帧或单个 transition 偶然通过不支持候选结论。一个 representation family 只有
在 early0、late0 的三个固定区间共6条记录全部通过时才称 stable。

## 4. 候选分类边界

- 至少一个预声明 interval family 在两个 Flow stage、三个 video-time interval
  全部稳定：`temporal_feature_salvage_candidate`；
- 两个 interval family 均不稳定：
  `carrier_redesign_required_candidate`；
- source、clean、finite 或coverage不完整：`indeterminate`。

这些只是下一实验设计候选。即使 temporal candidate 出现，也不得自动选择该
representation 作为新 primary feature；必须另起 construction identity A 的 public
feature、clean calibration 与 fixed whitening 设计。

所有输出固定：

```text
gate_a_pass=false
formal_result=false
stage_progression_allowed=false
```

Gate B/C、wrong-key、observer、state dynamics、GPU、Colab、Drive、attack、
fixed-FPR、baseline 与 paper claim 全部关闭。

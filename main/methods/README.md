# SSTW 核心方法包

该目录实现状态空间同步 Flow Matching 轨迹水印的最小方法原语：

1. 在 Flow scheduler 更新前施加密钥条件速度场弱约束。
2. 在规范五维 latent 上构造时空 tubelet key code。
3. 从视频 VAE endpoint 执行 key 无关反演与候选假设 replay。
4. 使用独立 clean-video 簇的 null residual 拟合模型特定 replay 高斯噪声,
   核心 replay API 不提供固定经验方差默认值。
5. 使用双假设线性高斯状态空间模型、Kalman filtering、RTS smoothing 和
   分组交叉拟合概率校准形成检测后验。
6. 使用 calibration 负样本冻结 fixed-FPR 阈值，并在 held-out 输入上只读评分。
7. 使用 HMAC-SHA256 认证生成端轨迹摘要。

核心实现只接受显式机制配置和规范化多 phase 观测，不解释任何外层实验名称。
默认参数位于 `configs/methods/sstw_core_method.json`。

最小包执行 Flow 生成、视频 VAE endpoint 解码或 replay 前，应安装完整方法运行依赖：

```bash
pip install -e ".[method-runtime]"
```

该 extra 除 `torch`、`diffusers` 和 `transformers` 外，还包含核心 mp4 解码
路径直接使用的 `imageio` 与 `imageio-ffmpeg`。`video-evaluation` 只补充外层
质量指标依赖，不能替代核心方法运行依赖。

## 核心检测 API

`formal_detector.py` 的检测数据流分为三个显式步骤：

1. `flow_evidence_observation_sequence_from_mappings` 将逐 phase 数值映射转换为
   `FlowEvidenceObservation` 序列。
2. `fit_flow_detector_calibration` 接收观测序列、二元标签和独立视频簇标识，
   拟合概率后验并冻结 fixed-FPR 阈值。
3. `apply_frozen_flow_detector` 只接收一个观测序列和冻结 calibration，返回
   核心分数、阈值与二元判定，不修改模型参数。

数据分区、正式样本角色、论文指标状态和阈值来源记录由外层实验协议适配。
这样核心方法包既可以独立发布，也不会反向依赖论文运行流程。

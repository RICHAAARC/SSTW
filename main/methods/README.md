# SSTW 核心方法包

该目录实现状态空间同步 Flow Matching 轨迹水印的最小方法原语：

1. 在 Flow scheduler 更新前施加密钥条件速度场弱约束。
2. 在规范五维 latent 上构造时空 tubelet key code。
3. 从视频 VAE endpoint 执行 key 无关反演与候选假设 replay。
4. 使用双假设线性高斯状态空间模型、Kalman filtering、RTS smoothing 和
   分组交叉拟合概率校准形成检测后验。
5. 使用 calibration 负样本冻结 fixed-FPR 阈值，并在 held-out 输入上只读评分。
6. 使用 HMAC-SHA256 认证生成端轨迹摘要。

核心实现只接受显式机制配置和规范化多 phase 观测，不解释任何外层实验名称。
默认参数位于 `configs/methods/sstw_core_method.json`。

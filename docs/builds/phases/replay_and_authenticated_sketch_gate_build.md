# replay_and_authenticated_sketch_gate 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段规范 owner-side trajectory audit、model-side replay verification 和 video-only proxy observation 的证据等级, 防止 SSTW 被误解为未认证服务端日志或普通 replay 后处理。

### 1.2 输入

```text
main/core/digest.py
main/core/manifests.py
main/trajectory/trajectory_trace.py
main/generation/trajectory_capture.py
experiments/trajectory_observation_core/
experiments/generative_video_model_probe/
```

### 1.3 构建任务

1. 定义 authenticated trajectory sketch 的记录、摘要和验证状态。
2. 为 replay trajectory 记录 scheduler、time grid 和 uncertainty weight。
3. 定义 wrong sampler replay control。
4. 将 evidence level 写入 manifest 和 claim audit。
5. 限制 video-only proxy 的论文主张边界。

### 1.4 证据等级

```text
level_1_authenticated_cached_trajectory
level_2_model_side_replay_with_uncertainty
level_3_video_only_proxy_observation
```

### 1.5 必须字段

```text
authenticated_trajectory_sketch_status
trajectory_sketch_digest_random
trajectory_sketch_verification_status
replay_uncertainty_weight
replay_scheduler_id
replay_time_grid_id
wrong_sampler_replay_control
```

### 1.6 通过标准

1. level 1 可以支撑高置信 owner-side trajectory audit。
2. level 2 必须记录 replay uncertainty。
3. level 3 只能作为补充证据, 除非经过 fixed-FPR 独立验证。
4. 未认证 trajectory logging 只能作为 control。

## 2. 当前阶段具体完成情况

### 2.1 已有工程基础

仓库中已经存在 digest、manifest、trajectory trace 和 trajectory capture 基础模块, 可作为 authenticated sketch 与 replay uncertainty 的实现基础。

### 2.2 当前阶段补充要求

该阶段需要在后续工程中进一步补齐 authenticated trajectory sketch 的正式字段、manifest 记录和 checker 规则。未认证 logging 不得直接支撑 supported claims。

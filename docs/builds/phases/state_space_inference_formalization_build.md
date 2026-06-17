# state_space_inference_formalization 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段将密钥条件状态空间推断形式化为可审计、可消融、可复现实验模块, 避免方法被解释为普通分数堆叠或通用 SSM 后处理。

### 1.2 输入

```text
configs/protocol/state_space_formalization.json
configs/methods/method_variants_state_space_formalization.json
configs/ablations/key_condition_ablation.json
configs/ablations/state_variable_ablation.json
main/methods/state_space_watermark/
experiments/state_space_formalization/
```

### 1.3 构建任务

1. 明确定义 state variable、transition、observation likelihood 和 posterior score。
2. 实现 filtering 与 smoothing 的统一接口。
3. 构造 key-conditioned 与 key-agnostic 对照。
4. 构造 without transition、without observation、without admissibility 消融。
5. 将每个 supported claim 绑定到 records 和 claim audit 报告。

### 1.4 必须 baseline / ablation

```text
without_key_condition
without_state_transition
without_observation_likelihood
without_admissibility
generic_ssm
mamba_style_temporal_fusion_control
```

### 1.5 通过标准

1. key conditioning 对检测结果有独立贡献。
2. 状态转移和观测似然不能被普通 temporal fusion 完全替代。
3. admissibility 能降低 false positive tail。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有形式化阶段相关模块:

```text
experiments/state_space_formalization/runner.py
experiments/state_space_formalization/formal_audit.py
experiments/state_space_formalization/ablation_builder.py
experiments/state_space_formalization/generalization_runner.py
experiments/state_space_formalization/table_builder.py
experiments/state_space_formalization/package_outputs.py
```

### 2.2 已有方法结构

`main/methods/state_space_watermark/` 已经拆分为 key conditioner、state filter、state smoother、state synchronizer、state transition、state observation 和 detector score 等模块。该拆分有利于后续独立消融。

### 2.3 当前阶段使用边界

该阶段支撑 state-space inference 的机制独立性, 但不能替代 Flow Matching velocity constraint 与 trajectory observation 的证据。

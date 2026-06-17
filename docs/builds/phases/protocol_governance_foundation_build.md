# protocol_governance_foundation 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段的作用是建立 SSTW 项目的治理底座。它不直接证明水印有效性, 而是固定后续所有实验必须遵守的协议、字段、命名、测试、Notebook 边界和 artifact rebuild 规则。

### 1.2 输入

```text
docs/builds/sstw_project_construction_flow.md
docs/builds/sstw_method_mechanism_design.md
.codex/project_contract.md
configs/protocol/*.json
configs/records/*.json
docs/field_registry.md
```

### 1.3 构建任务

1. 固定 sample role、split、target FPR、threshold 来源和 test-time 禁止更新规则。
2. 建立 negative family, 覆盖 clean negative、attacked negative、replay negative、sampler mismatch negative、wrong prompt negative、wrong time grid negative 与 wrong key negative。
3. 建立 event record、state trace、threshold 和 manifest 的字段边界。
4. 建立 placeholder 字段和 random trace 字段命名规则。
5. 建立默认测试分层, 避免 GPU 重型测试进入默认 `pytest -q`。
6. 建立 harness 审计入口, 保证命名、依赖边界、UTF-8、测试约束和 release extraction 规则可被自动检查。

### 1.3.1 negative family 字段

```text
negative_family
```

该字段用于把 calibration negative 拆分为可审计负样本家族, 避免只用 clean negative 低估 replay、sampler mismatch 和 wrong-key 误报风险。

允许取值包括:

```text
none
clean_negative
attacked_negative
replay_negative
sampler_mismatch_negative
wrong_prompt_negative
wrong_time_grid_negative
wrong_key_negative
trajectory_shuffle_negative
```

### 1.4 必须产物

```text
configs/protocol/sstw_protocol.json
configs/protocol/fixed_low_fpr.json
configs/records/event_record_schema.json
configs/records/state_trace_schema.json
configs/records/threshold_schema.json
docs/field_registry.md
tools/harness/run_all_audits.py
tests/constraints/test_harness_contract.py
```

### 1.5 通过标准

1. `pytest -q` 可以在本地 CPU 环境中完成。
2. `python tools/harness/run_all_audits.py` 返回 pass。
3. 没有非 snake_case 正式文件名。
4. placeholder 字段以 `_placeholder` 结尾。
5. random trace 字段以 `_random` 或 `_digest_random` 结尾。

## 2. 当前阶段具体完成情况

### 2.1 已有工程基础

仓库中已经存在协议、字段注册、记录 schema、harness 审计和默认测试入口。当前阶段的主要工程骨架已经形成。

### 2.2 已有治理能力

已具备以下治理能力:

```text
naming_governance
placeholder_random_field_governance
test_case_governance
artifact_rebuild_contract
claim_audit_boundary
dependency_boundary_audit
```

### 2.3 当前仍需保持的约束

1. 后续新增字段必须先进入 `docs/field_registry.md` 或对应 schema。
2. 后续新增 Notebook 不能直接写正式 records、thresholds、tables、figures 或 reports。
3. 后续新增阶段输出不能放入 checked-in `outputs/` 作为正式论文产物。

# Project Contract Template

## Long-Term Goal

本项目采用 governed research project 方法构建论文相关代码库: 先定义契约、目录边界、字段注册、测试分层、Notebook 边界和论文产物重建规则, 再实现论文方法与实验流程。

## Current Stage

- `project_stage`: `project_bootstrap`
- `target_construction_phase`: `core_method_runtime_construction`
- 当前阶段只允许建立目录、文档、skill、harness、测试分层和最小 `main/` 核心包骨架。
- 当前阶段不应引入真实大规模数据、正式实验输出、论文最终图表或发布包。

## Ordered Semantic Stages

1. `project_bootstrap`
2. `core_method_runtime_construction`
3. `experiment_protocol_validation`
4. `paper_artifact_rebuild_gate`
5. `submission_readiness_gate`
6. `minimal_release_extraction`

## Core Directory Rules

1. `main/` 保存论文方法、核心协议、核心评估、表格重建和 CLI 复现能力。
2. `experiments/` 保存阶段性实验 runner、ablation、baseline 或 paper protocol。
3. `paper_workflow/` 保存 Notebook / Colab workflow 入口和 session helper。
4. `scripts/` 保存数据准备、结果检查、结果打包和 release 辅助命令。
5. `tools/harness/` 保存外层治理审计, 不得被 `main/` 反向依赖。
6. `.codex/` 和 `docs/` 保存协作契约与人类可读治理规则。
7. `tests/` 按运行成本和验证目标分层。
8. `audit_reports/` 和 `outputs/` 是本地运行输出目录, 默认不提交。

## Notebook Boundary Rules

1. Notebook 是论文实验的入口和远程执行包装, 不是正式协议逻辑的唯一实现。
2. Notebook 不得直接手写正式 records、thresholds、tables、figures 或 reports。
3. Notebook 应调用 `main/`、`experiments/` 或 `scripts/` 中的 repository modules。
4. Notebook 专用 helper 放在 `paper_workflow/notebook_utils/`。
5. 跨 Notebook 共享的 Colab 或 session helper 放在 `paper_workflow/colab_utils/`。

## Paper Artifact Governance

1. records 是论文结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定到 governed artifacts。
4. 手工拼接正式论文结果表、正式图数据或正式 claim audit 属于阻断违规。
5. manifests 必须记录输入、输出、配置摘要、代码版本和重建命令。

## Naming Governance

1. 正式文件名、目录名、模块名、配置键和字段名应使用 `snake_case`。
2. 禁止用数字阶段名、弱版本后缀、`new`、`old`、`best`、`final` 等词作为正式语义。
3. 方法、实验、报告和配置应使用能表达机制、实验协议或论文职责的名称。

## Placeholder And Random Governance

1. Placeholder 字段必须以 `_placeholder` 结尾。
2. Random trace 字段必须以 `_random` 或 `_digest_random` 结尾。
3. Placeholder 字段不得支持 supported claims。
4. Governed fields 应先登记到 `docs/field_registry.md`。

## Test Governance

1. 默认 `pytest -q` 只运行 `unit`、`constraint` 或 `quick` 测试。
2. `tests/constraints/` 保存静态或轻量治理测试。
3. `tests/functional/` 保存轻量功能测试。
4. `tests/integration/` 保存集成、smoke、slow 或 formal 测试, 默认排除。
5. 测试输出必须使用 `tmp_path` 或 `tmp_path_factory`。

## Required Completion Commands

```bash
pytest -q
python tools/harness/run_all_audits.py
```

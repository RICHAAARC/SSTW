# Project Contract Template

## Long-Term Goal

本项目采用 governed research project 方法构建论文相关代码库: 先定义契约、目录边界、字段注册、测试分层、Notebook 边界和论文产物重建规则, 再实现论文方法与实验流程。

## Current Stage

- `project_stage`: `core_method_runtime_construction`
- `target_construction_phase`: `experiment_protocol_validation`
- 当前阶段允许实现真实 Flow velocity 约束、endpoint/path 检测、replay、认证 sketch、
  fixed-FPR 校准和 paper profile runner, 并使用轻量测试验证核心语义。
- 当前阶段不应提交真实大规模数据、正式实验输出、论文最终图表或发布包。进入
  `experiment_protocol_validation` 前, 必须先在独立 GPU 运行目录完成 probe_paper 的三层证据闭合。

## Ordered Semantic Stages

1. `project_bootstrap`
2. `core_method_runtime_construction`
3. `experiment_protocol_validation`
4. `paper_artifact_rebuild_gate`
5. `submission_readiness_gate`
6. `minimal_release_extraction`

## Core Directory Rules

1. `main/` 只保存可独立抽离的 SSTW 论文核心方法, 不保存 runner、表格、攻击流程或 Notebook helper。
2. `runtime/` 保存模型加载之外的通用运行时基础设施, 可以依赖 `main/`。
3. `evaluation/` 保存攻击、指标、统计和结果协议, 可以依赖 `main/` 与 `runtime/`。
4. `external_baseline/` 保存官方 baseline 适配层, 不得被 `main/` 反向依赖。
5. `experiments/` 保存真实 GPU runner、ablation 和 paper protocol, 可以依赖上述内层。
6. `workflows/` 保存可脱离 Notebook 在 GPU 服务器执行的阶段编排。
7. `scripts/` 保存服务器 CLI、数据准备、结果检查、打包和 release 辅助命令。
8. `paper_workflow/` 只保存 Notebook / Colab 入口薄包装, 是最外层。
9. `tools/harness/` 保存外层治理审计, 不得被任何运行内层反向依赖。
6. `.codex/` 和 `docs/` 保存协作契约与人类可读治理规则。
7. `tests/` 按运行成本和验证目标分层。
8. `audit_reports/` 和 `outputs/` 是本地运行输出目录, 默认不提交。

## Notebook Boundary Rules

1. Notebook 是论文实验的入口和远程执行包装, 不是正式协议逻辑的唯一实现。
2. Notebook 不得直接手写正式 records、thresholds、tables、figures 或 reports。
3. Notebook 只调用 `workflows/` 的规范入口, 不直接实现方法、检测、统计或产物重建。
4. 同一 workflow 必须可由 `scripts/run_generative_video_server_workflow.py` 在普通 GPU 服务器执行。
5. Notebook 专用鉴权和挂载薄包装放在 `paper_workflow/`, 内层不得导入该目录。

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

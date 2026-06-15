# 文件组织契约

## 文档定位

本文档约束论文相关研究项目的目录边界。目标是将论文方法核心、阶段性实验、Notebook workflow、辅助脚本、治理审计和发布材料分离, 避免临时实验代码污染 `main/`。

## 推荐目录

```text
main/                   核心方法、协议、评估、分析、CLI 复现能力
configs/                实验配置、协议配置、数据 manifest 模板
experiments/            阶段性实验 runner、ablation、baseline、paper protocol
paper_workflow/         Notebook / Colab workflow 入口和 session helper
scripts/                数据准备、结果检查、结果打包、release 辅助命令
docs/                   方法说明、复现说明、治理契约、投稿材料说明
tools/harness/          可执行治理审计
tests/                  分层测试目录
.codex/                 Agent 协作契约与 skill 文件
audit_reports/          本地审计输出, 默认不提交
outputs/                本地运行输出, 默认不提交
```

## `main/` 边界

`main/` 只保存论文方法和最终复现需要的核心能力, 包括:

```text
main/core/              records、manifest、schema、digest、registry
main/protocol/          split、threshold、runner 接口、输出布局
main/methods/           论文方法和机制变体
main/analysis/          表格、图数据、报告和 claim audit 构建
main/cli/               命令行复现实验入口
```

## 禁止依赖方向

```text
main/ -> tools/harness/
main/ -> tests/
main/ -> experiments/
main/ -> paper_workflow/
main/ -> audit_reports/
main/ -> outputs/
```

## 允许依赖方向

```text
tools/harness/ -> main/
tests/ -> main/
experiments/ -> main/
scripts/ -> main/
paper_workflow/ -> main/
```

## Notebook 边界

Notebook 可以作为论文实验入口, 但不得成为唯一正式实现路径。正式 records、tables、figures、reports 和 manifests 的写入逻辑应位于 `main/`、`experiments/` 或 `scripts/`。

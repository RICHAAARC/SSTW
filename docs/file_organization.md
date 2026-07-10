# 文件组织契约

## 文档定位

本文档约束论文相关研究项目的目录边界。目标是将论文方法核心、阶段性实验、Notebook workflow、辅助脚本、治理审计和发布材料分离, 避免临时实验代码污染 `main/`。

## 推荐目录

```text
main/                   可独立抽离的 SSTW 最小论文方法包
runtime/                通用运行时基础设施
evaluation/             攻击、指标、统计与结果协议
external_baseline/      官方 baseline 源码边界与适配器
configs/                实验配置、协议配置、数据 manifest 模板
experiments/            阶段性实验 runner、ablation、baseline、paper protocol
workflows/              Notebook 无关、GPU 服务器可执行的阶段编排
paper_workflow/         最外层 Notebook / Colab 薄入口
scripts/                数据准备、结果检查、结果打包、release 辅助命令
docs/                   方法说明、复现说明、治理契约、投稿材料说明
tools/harness/          可执行治理审计
tests/                  分层测试目录
.codex/                 Agent 协作契约与 skill 文件
audit_reports/          本地审计输出, 默认不提交
outputs/                本地运行输出, 默认不提交
```

## `main/` 边界

`main/` 只保存论文方法本身, 包括:

```text
main/methods/state_space_watermark/
                        速度场嵌入、tubelet key、endpoint/path 观测、
                        key 无关反演、候选 replay 假设和概率后验检测
```

## 禁止依赖方向

```text
main/ -> runtime/、evaluation/、external_baseline/、experiments/、workflows/、scripts/、paper_workflow/
runtime/ -> evaluation/、experiments/、workflows/、scripts/、paper_workflow/
evaluation/ -> experiments/、workflows/、scripts/、paper_workflow/
experiments/ -> workflows/、scripts/、paper_workflow/
workflows/ -> paper_workflow/
```

## 允许依赖方向

```text
main/ <- runtime/ <- evaluation/ <- experiments/ <- workflows/ <- scripts/ <- paper_workflow/
external_baseline/ 可被 experiments、workflows 和 scripts 调用, 不能被 main 反向调用。
```

## Notebook 边界

Notebook 只负责挂载 Drive、加载密钥和调用 `workflows/`。正式方法、records、tables、figures、reports 和 manifests 均位于内层模块, 并由服务器 CLI 复用。

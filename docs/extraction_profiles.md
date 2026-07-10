# 抽离 profiles

## Profile 总览

| profile_name | 作用 | 包含 artifact builders | 目标读者 |
| --- | --- | --- | --- |
| `development_repository` | 完整开发、测试、治理与 Colab 入口。 | 是 | 项目作者与协作者。 |
| `paper_artifact_rebuild_package` | 在普通 GPU 服务器执行正式协议并重建论文产物。 | 是 | 审稿人和复现实验者。 |
| `minimal_method_package` | 只理解或复用 SSTW 核心方法。 | 否 | 方法读者。 |

## `development_repository`

包含 `main/`、`runtime/`、`evaluation/`、`external_baseline/`、`experiments/`、`workflows/`、`scripts/`、`paper_workflow/`、测试和治理工具。排除 `audit_reports/`、`outputs/`、缓存、虚拟环境和构建目录。

## `paper_artifact_rebuild_package`

默认包含：

```text
main/
runtime/
evaluation/
external_baseline/
configs/
experiments/
workflows/
scripts/
docs/artifact_rebuild.md
docs/field_registry.md
docs/file_organization.md
docs/release_boundary.md
docs/extraction_profiles.md
docs/intermediate_state_governance.md
tests/functional/
README.md
pyproject.toml
```

默认排除 `.codex/`、`tools/harness/`、`paper_workflow/`、重型测试、私有数据、运行输出和 Notebook 缓存。该 profile 必须能够脱离 Notebook 在 GPU 服务器运行。

## `minimal_method_package`

默认包含：

```text
main/methods/
configs/methods/
README.md
pyproject.toml
```

默认排除 `runtime/`、`evaluation/`、`external_baseline/`、`experiments/`、`workflows/`、`scripts/`、`paper_workflow/`、治理工具、测试和运行输出。

## 依赖方向

允许的方向为：

```text
runtime/ -> main/
evaluation/ -> main/, runtime/
external_baseline/ -> main/, runtime/, evaluation/
experiments/ -> main/, runtime/, evaluation/, external_baseline/
workflows/ -> experiments/, evaluation/, external_baseline/
scripts/ -> workflows/, experiments/, evaluation/
paper_workflow/ -> workflows/
tools/harness/ -> 任意受治理路径
tests/ -> 对应被测层
```

禁止任何内层反向依赖外层，尤其是 `main/` 导入 runtime、evaluation、experiments、workflows、scripts、paper_workflow 或 harness。该规则由 dependency boundary audit 强制检查。

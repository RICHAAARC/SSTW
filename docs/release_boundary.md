# 发布边界

## 与抽离 profile 的关系

本文件说明发布原则，`docs/extraction_profiles.md` 定义可执行的抽离 profile。开发仓库、最小方法包和实验重建包具有不同职责，不能把整个开发仓库直接作为最小方法实现。

## `minimal_method_package`

该包是最小论文方法实现，只保留可独立理解和复用的 SSTW 核心原语：

```text
main/methods/
configs/methods/
pyproject.toml
```

它不包含 runner、攻击流程、统计、baseline、Notebook、治理工具或正式实验输出。`main/` 不得依赖任何外层目录。

## `paper_artifact_rebuild_package`

该包用于在普通 GPU 服务器执行实验并从 governed records 重建 tables、figures、reports 与 manifests。默认包含：

```text
main/
runtime/
evaluation/
external_baseline/
configs/
experiments/
workflows/
scripts/
必要的 docs/
README.md
pyproject.toml
```

该包明确排除：

```text
.codex/
tools/harness/
paper_workflow/
audit_reports/
outputs/
tests/constraints/
tests/integration/
tests/functional/
私有密钥、私有数据与本地绝对路径配置
```

`paper_workflow/` 只是 Colab / Notebook 最外层入口，服务器复现不依赖它。测试与
`tools/harness/` 属于开发仓库的抽离前治理层，同样不进入服务器重建包。开发仓库必须先运行
`pytest -q` 和 `python tools/harness/run_all_audits.py`，再执行抽离。抽离清单会将
`package_execution_mode` 标记为 `paper_artifact_rebuild_package`；服务器 workflow 读取该标记后，
不会在重建包内重复访问已经排除的测试和 harness 路径。

## 正式输出边界

正式实验输出必须写入独立本地或远程运行目录，由 records 和 manifests 驱动重建。仓库内不得提交真实大规模结果、最终论文表图、私有认证密钥或未经治理的临时输出。

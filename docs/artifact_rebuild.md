# Artifact 重建治理

正式 artifacts 必须满足：

1. records 是事实来源。
2. tables 由 records 或受治理中间表重建。
3. figures 由 records 或 tables 重建。
4. reports 由 records、tables、figures 和 manifests 生成。
5. manifests 记录输入、输出、代码版本、配置摘要和重建命令。
6. supported claims 只能引用 governed artifacts。

## 与核心方法的边界

Artifact 重建属于实验与论文产物层，不属于最小核心方法层。当前依赖方向为：

```text
main <- runtime <- evaluation <- experiments <- workflows <- scripts <- paper_workflow
                    ^
                    |
             external_baseline
```

`evaluation/metrics/`、`evaluation/statistics/` 和 `evaluation/protocol/` 可以依赖 `main/` 与 `runtime/`；`main/` 不得反向依赖这些目录。artifact builders 位于 `experiments/` 或 `scripts/`，Notebook 只调用 `workflows/`，不得在单元格中手写正式 records、tables、figures 或 reports。

这一边界保证 `minimal_method_package` 可以只抽离 `main/methods/` 与方法配置，同时服务器重建包仍保留完整实验和证据生产能力。

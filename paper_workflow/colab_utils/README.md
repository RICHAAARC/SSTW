# Colab helper 目录

该目录只保存 Colab 或跨 Notebook 共享的 Python helper。Notebook 入口文件统一放在 `paper_workflow/colab_notebooks/`。

目录边界:

- `.py` helper 可以被 Notebook、测试和 workflow 配置复用。
- 不在本目录新增 `.ipynb` 文件。
- Notebook 不直接手写正式 records、tables、figures、reports 或 manifests, 只调用仓库模块。

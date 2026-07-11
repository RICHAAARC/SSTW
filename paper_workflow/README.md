# Paper Workflow

此目录保存论文相关 Notebook 入口和 Colab 专用挂载、鉴权包装。Notebook 只能调用
`scripts/run_generative_video_server_workflow.py`, 不得直接调用实验 runner、方法、检测、
统计、门禁或产物 writer。普通 GPU 服务器与 Colab 因而共享同一 workflow 和同一
fail-closed 环境预检。

Colab 冷启动执行顺序、workflow profile 切换方式、Google Drive 落盘路径和 package 检查方法见 `paper_workflow/colab_notebooks/README.md`。

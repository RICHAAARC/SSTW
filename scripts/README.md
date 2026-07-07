# Scripts

此目录保存命令行辅助脚本。

## GPU 服务器无 Notebook 运行入口

`run_generative_video_server_workflow.py` 用于在普通 GPU 服务器上执行与 Colab
Notebook 等价的 workflow。Notebook 只作为 Colab 入口; 该脚本复用
`configs/paper_workflow/generative_video_notebook_workflows.json` 中的 stage plan
和 `paper_workflow/notebook_utils/generative_video_model_probe_workflow.py` 中的命令构造逻辑。

典型 validation_scale 全流程:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile validation_scale \
  --pipeline paper_protocol_complete
```

只检查将要执行的阶段, 不运行 GPU:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --pipeline formal_comparison_scoring \
  --dry-run
```

只重跑 formal comparison scoring:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile validation_scale \
  --pipeline formal_comparison_scoring
```

只重跑 paper evidence postprocess:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile validation_scale \
  --pipeline paper_evidence_postprocess
```

只重跑 paper gate/package:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile validation_scale \
  --pipeline paper_gate_and_package
```

`--project-root` 是服务器上的结果根目录, 等价于 Colab 中的 Google Drive
项目根。阶段 zip、manifest、records、tables、figures 和 reports 都会落在该根目录
下的 profile-specific 子目录中。

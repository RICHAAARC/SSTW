# Scripts

此目录保存命令行辅助脚本。

## GPU 服务器无 Notebook 运行入口

`run_generative_video_server_workflow.py` 用于在普通 GPU 服务器上执行与 Colab
Notebook 等价的 workflow。Notebook 只作为 Colab 入口; 该脚本复用
`configs/paper_workflow/generative_video_notebook_workflows.json` 中的 stage plan
和 `workflows/generative_video_paper.py` 中的命令构造逻辑。

典型 probe_paper 全流程:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile probe_paper \
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
  --workflow-profile probe_paper \
  --pipeline formal_comparison_scoring
```

只重跑 paper evidence postprocess:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile probe_paper \
  --pipeline paper_evidence_postprocess
```

只重跑 paper gate/package:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile probe_paper \
  --pipeline paper_gate_and_package
```

`--project-root` 是服务器上的结果根目录, 等价于 Colab 中的 Google Drive
项目根。阶段 zip、manifest、records、tables、figures 和 reports 都会落在该根目录
下的 profile-specific 子目录中。

## 受锁定的 GPU 运行环境

服务器与 Colab 必须先安装同一份精确依赖锁:

```bash
python -m pip install --requirement requirements/paper_runtime_lock.txt
```

非 `--dry-run` 执行会自动读取
`requirements/paper_runtime_environment_lock.json`, 并在任何正式阶段前验证:

1. Python、PyTorch、CUDA 与所有公共 distribution 的精确版本;
2. GPU 显存和 compute capability 是否达到锁文件下限;
3. 开发仓库是否绑定40位 commit 且工作树干净, 或抽离包是否记录了干净源 commit;
4. Wan2.1 与 LTX-Video revision 是否已解析为不可变的40位 Hugging Face commit。

任一条件不满足时, CLI 返回非零状态并阻断 workflow。只运行预检可使用:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --pipeline runtime_environment_preflight
```

为了复跑既有实验, 应从上一次服务器 decision 中读取模型 commit 并显式传入:

```bash
python scripts/run_generative_video_server_workflow.py \
  --project-root /data/SSTW \
  --workflow-profile probe_paper \
  --pipeline paper_protocol_complete \
  --model-revision <40位主模型commit> \
  --cross-model-revision <40位跨模型commit>
```

空 revision 只允许在首次运行时通过 Hugging Face 元数据解析; CLI 会先冻结 commit,
再把相同 commit 传入生成 workflow。Notebook 不维护第二套运行逻辑, 只构造并调用
上述同一条服务器命令。

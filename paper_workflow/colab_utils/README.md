# Colab helper 目录

该目录只保存 Colab 或跨 Notebook 共享的 Python helper。Notebook 入口文件统一放在 `paper_workflow/colab_notebooks/`。

目录边界:

- `.py` helper 可以被 Notebook、测试和 workflow 配置复用。
- 不在本目录新增 `.ipynb` 文件。
- Notebook 不直接手写正式 records、tables、figures、reports 或 manifests, 只调用仓库模块。

## 轨迹认证加载器

`trajectory_authentication.py` 从 Google Drive 项目根目录下的私有 JSON 文件加载
`SSTW_TRAJECTORY_AUTHENTICATION_KEY` 和
`SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID`。它只向当前 Python 进程环境写入密钥,
返回值不包含密钥本体, 因而可以由各个拆分式 paper profile Notebook 复用。

该 helper 属于通用的 secret bootstrap 写法。项目特定部分包括默认 Drive 路径、
32字节最低熵要求、key ID 格式以及同一 runtime 禁止静默更换认证身份。

## Hugging Face 认证引导

`huggingface_authentication.py` 解决 VSCode 直接连接 Colab kernel 时本机环境变量
不会自动进入远端进程的问题。它优先使用远端已有的 `HF_TOKEN`, 否则从 Google
Drive 项目根目录的 `.sstw_private/hf_token.txt` 读取 token, 完成登录并把 token
写入当前远端进程环境, 以便服务器 CLI 子进程继承。返回摘要只包含认证状态、来源
和账号名, 不包含 token 本体。不同 Drive 布局可通过 `SSTW_HF_TOKEN_FILE` 覆盖路径。

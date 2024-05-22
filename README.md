大模型推理/训练时的小工具

- [1. 下载 huggingface 上的模型](#1-下载-huggingface-上的模型)
- [2. huggingface 模型的layer层 input/output 的导出](#2-huggingface-模型的layer层-inputoutput-的导出)


# 1. 下载 huggingface 上的模型 
- llm_tools/download_model_hf.py
  - 需要 access token, 通过注册hugging face账号可以获得
  - `pip install huggingface_hub`
  - 复制 huggingface 上模型的名字到 repo_id 即可

# 2. huggingface 模型的layer层 input/output 的导出

- llm_tools/layer_hook_hf.py
  - 
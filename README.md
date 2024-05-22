大模型推理/训练时的小工具

- [1. 下载 huggingface 上的模型](#1-下载-huggingface-上的模型)
- [2. huggingface 模型的layer层 input/output 的导出](#2-huggingface-模型的layer层-inputoutput-的导出)


# 1. 下载 huggingface 上的模型 
- llm_tools/download_model_hf.py
  - 需要 access token, 通过注册hugging face账号可以获得
  - `pip install huggingface_hub`
  - 复制 huggingface 上模型的名字到 repo_id 即可

# 2. huggingface 模型的layer层 input/output 的导出

- llm_tools/llm_infer/layer_hook_hf.py
  - 如果我们使用 tensorrt-llm/vllm 等大模型推理引擎时，会需要进行精度对齐，那么 transformers 的官方实现，可以作为对齐的基准
  - 该小工具可以一层一层的提取 transformers 端的输入和输出


```
Layer model.layers.0.input_layernorm input - mean: 3.612041473388672e-05, sum: 1.03515625, shape: torch.Size([1, 7, 4096])
Layer model.layers.0.input_layernorm output - mean: -0.00041937828063964844, sum: -12.0234375, shape: torch.Size([7, 4096])
Layer model.layers.0.self_attn.q_proj input - mean: -0.00041937828063964844, sum: -12.0234375, shape: torch.Size([1, 7, 4096])
Layer model.layers.0.self_attn.q_proj output - mean: 0.001178741455078125, sum: 33.8125, shape: torch.Size([7, 4096])
Layer model.layers.0.self_attn.k_proj input - mean: -0.00041937828063964844, sum: -12.0234375, shape: torch.Size([1, 7, 4096])
Layer model.layers.0.self_attn.k_proj output - mean: -0.0028285980224609375, sum: -81.125, shape: torch.Size([7, 4096])
Layer model.layers.0.self_attn.v_proj input - mean: -0.00041937828063964844, sum: -12.0234375, shape: torch.Size([1, 7, 4096])
Layer model.layers.0.self_attn.v_proj output - mean: 0.0008749961853027344, sum: 25.09375, shape: torch.Size([7, 4096])
Layer model.layers.0.self_attn.rotary_emb input - mean: 0.0008749961853027344, sum: 25.09375, shape: torch.Size([1, 32, 7, 128])
Layer model.layers.0.self_attn.rotary_emb output - mean: 0.8447265625, sum: 757.0, shape: torch.Size([7, 128])
Layer model.layers.0.self_attn.o_proj input - mean: 0.0003902912139892578, sum: 11.1953125, shape: torch.Size([1, 7, 4096])
Layer model.layers.0.self_attn.o_proj output - mean: -0.0016489028930664062, sum: -47.28125, shape: torch.Size([7, 4096])
Layer model.layers.0.self_attn input is None, empty or not a tensor.
```
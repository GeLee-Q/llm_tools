
- [LLM 工具集使用指南](#llm-工具集使用指南)
  - [1. 下载 Hugging Face 模型](#1-下载-hugging-face-模型)
  - [2. Hugging Face 模型结构与权重数据](#2-hugging-face-模型结构与权重数据)
  - [3. Hugging Face 模型每层输入输出](#3-hugging-face-模型每层输入输出)
  - [4. Tensorrt-LLM 权重转换阶段与 Hugging Face 模型对齐](#4-tensorrt-llm-权重转换阶段与-hugging-face-模型对齐)
  - [5. Tensorrt-LLM 模型构建阶段与 Hugging Face 模型对齐](#5-tensorrt-llm-模型构建阶段与-hugging-face-模型对齐)
  - [6. Tensorrt-LLM 推理阶段与 Hugging Face 模型对齐](#6-tensorrt-llm-推理阶段与-hugging-face-模型对齐)


# LLM 工具集使用指南

本指南介绍如何使用一系列工具来下载、分析和对齐 Hugging Face 上的大语言模型，并将其与 TensorRT 推理引擎集成。

## 1. 下载 Hugging Face 模型

**工具路径:** `llm_tools/download_model_hf.py`

**步骤:**

1. **获取 Access Token:** 访问 Hugging Face 网站 ([https://huggingface.co/](https://huggingface.co/)) 并注册账号以获取访问令牌 (Access Token)。
2. **安装 Hugging Face Hub 库:** 使用 pip 安装 `huggingface_hub` 库：
   ```bash
   pip install huggingface_hub
   ```
3. **复制模型名称:** 在 Hugging Face 模型页面找到目标模型，复制其仓库名称 (repo_id)，例如 "google/flan-t5-xl"。
4. **运行下载脚本:** 执行 `download_model_hf.py` 脚本，并将 Access Token 和模型名称作为参数传入。

```bash
python llm_tools/download_model_hf.py 
```

## 2. Hugging Face 模型结构与权重数据

**状态:** 待完成


## 3. Hugging Face 模型每层输入输出

**工具路径:** `llm_tools/llm_infer/torch_hf_precision/layer_hook_hf.py`

**目的:**

- 当使用 TensorRT-LLM 或 VLLM 等大模型推理引擎时，需要进行精度对齐。
- 使用 Transformers 官方实现作为基准，逐层提取输入输出数据，用于与其他推理引擎的结果进行比较。

**使用方法:**

- 该脚本利用 PyTorch 钩子机制，在模型推理过程中记录每层的输入和输出张量。
- 用户需要指定目标模型和输入数据，脚本会返回每层的输入输出数据列表。

## 4. Tensorrt-LLM 权重转换阶段与 Hugging Face 模型对齐

**工具路径:** `llm_tools/llm_infer/tensorrt_llm_precision/build_intern.py`

**TensorRT-LLM 版本:** 0.7.1 (0.9.0 版本需要修改其他部分)

**Tensorrt-LLM 组网步骤:**

1. **权重处理:** 将 Hugging Face 模型的权重转换为 TensorRT-LLM 兼容的格式。
   -  TensorRT-LLM 定义了一系列模型结构，位于 `tensorrt_llm/models` 目录下。
   -  需要根据目标模型选择对应的 TensorRT-LLM 模型结构，并将 Hugging Face 权重映射到该结构中。
2. **模型构建:** 使用 TensorRT API 构建推理引擎。
   -  利用 TensorRT Builder 对象，将网络定义转换为可优化的推理引擎。

## 5. Tensorrt-LLM 模型构建阶段与 Hugging Face 模型对齐

**状态:** 待完成



## 6. Tensorrt-LLM 推理阶段与 Hugging Face 模型对齐

**状态:** 待完成




# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import math
import os
import time
from pathlib import Path

# isort: off
import torch
import torch.multiprocessing as mp
import tensorrt as trt
# isort: on
from transformers import AutoConfig, AutoModelForCausalLM
from weight import (get_scaling_factors, load_from_awq_internlm,
                    load_from_binary, load_from_gptq_internlm,
                    load_from_hf_internlm, load_from_meta_internlm)

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers.attention import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

from weight import parse_ft_config  # isort:skip

MODEL_NAME = "internlm"

# 2 routines: get_engine_name, serialize_engine
# are direct copy from gpt example, TODO: put in utils?

import onnx
import tensorrt as trt
from onnx import TensorProto, helper



def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    dtype = str_dtype_to_trt(args.dtype)
    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    assert args.n_layer % args.pp_size == 0, \
        f"num_layers {args.n_layer} must be a multiple of pipeline parallelism size {args.pp_size}"
    # Initialize Module
    tensorrt_llm_internlm = tensorrt_llm.models.LLaMAForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        num_kv_heads=args.n_kv_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=dtype,
        mlp_hidden_size=args.inter_size,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        mapping=mapping,
        rotary_base=args.rotary_base,
        rotary_scaling=args.rotary_scaling,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        use_fused_mlp=False,
        attn_bias=args.attn_bias,
        quant_mode=args.quant_mode,
        rms_norm_eps=args.rms_norm_eps)
    if args.use_smooth_quant:
        tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                               args.quant_mode)
    elif args.use_weight_only:
        if args.weight_only_precision == 'int8':
            tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                                   args.quant_mode)
        elif args.weight_only_precision == 'int4':
            tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                                   args.quant_mode)
        elif args.weight_only_precision == 'int4_awq':
            tensorrt_llm_internlm = quantize_model(model=tensorrt_llm_internlm,
                                                   quant_mode=args.quant_mode,
                                                   group_size=args.group_size,
                                                   zero=False,
                                                   pre_quant_scale=True,
                                                   exclude_modules=[])
        elif args.weight_only_precision == 'int4_gptq':
            tensorrt_llm_internlm = quantize_model(model=tensorrt_llm_internlm,
                                                   quant_mode=args.quant_mode,
                                                   group_size=args.group_size,
                                                   zero=True,
                                                   pre_quant_scale=False)
    elif args.enable_fp8 or args.fp8_kv_cache:
        logger.info(f'Loading scaling factors from '
                    f'{args.quantized_fp8_model_path}')
        quant_scales = get_scaling_factors(args.quantized_fp8_model_path,
                                           num_layers=args.n_layer,
                                           quant_mode=args.quant_mode)
        tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                               quant_mode=args.quant_mode,
                                               quant_scales=quant_scales)
    if args.per_group:
        load_func = load_from_awq_internlm if args.weight_only_precision == 'int4_awq' else load_from_gptq_internlm
        load_func(tensorrt_llm_internlm=tensorrt_llm_internlm,
                  quant_ckpt_path=args.quant_ckpt_path,
                  mapping=mapping,
                  dtype=args.dtype)
    elif args.meta_ckpt_dir is not None:
        load_from_meta_internlm(tensorrt_llm_internlm, args.meta_ckpt_dir,
                                mapping, args.dtype)
    elif args.model_dir is not None:
        logger.info(f'Loading HF InternLM ... from {args.model_dir}')
        tik = time.time()
        hf_internlm = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map={
                "model": "cpu",
                "lm_head": "cpu"
            },  # Load to CPU memory
            torch_dtype="auto",
            trust_remote_code=True)
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF InternLM loaded. Total time: {t}')
        load_from_hf_internlm(tensorrt_llm_internlm,
                              hf_internlm,
                              mapping=mapping,
                              dtype=args.dtype)
        
        # 打印模型信息
        print("\n>>>>>>> 模型参数信息:")
        for name, param in tensorrt_llm_internlm.named_parameters():
            print(f"{name}")
        
        print("\n>>>>>>> 模型模块信息:")
        for name, module in tensorrt_llm_internlm.named_modules():
            print(f"{name}: {module}")

        del hf_internlm
    elif args.ft_model_dir is not None:
        load_from_binary(tensorrt_llm_internlm,
                         args.ft_model_dir,
                         mapping,
                         fp16=(args.dtype == 'float16'),
                         multi_query_mode=(args.n_kv_head != args.n_head))

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_rmsnorm_plugin:
        network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)

    # Quantization plugins.
    if args.use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
        network.plugin_config.set_rmsnorm_quantization_plugin(dtype=args.dtype)
        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.multi_block_mode:
        network.plugin_config.enable_mmha_multi_block_mode()
    if args.use_weight_only:
        if args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype='float16')
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype,
                                              args.use_custom_all_reduce)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_internlm.named_parameters())

        # Forward
        inputs = tensorrt_llm_internlm.prepare_inputs(args.max_batch_size,
                                                      args.max_input_len,
                                                      args.max_output_len, True,
                                                      args.max_beam_width,
                                                      args.max_num_tokens)
        tensorrt_llm_internlm(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_internlm.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = dtype
        if args.visualize:
            model_path = os.path.join(args.output_dir, 'test.onnx')
            to_onnx(network.trt_network, model_path)

    tensorrt_llm.graph_rewriting.optimize(network)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine


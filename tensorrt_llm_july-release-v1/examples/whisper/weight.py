import configparser
import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import (pad_vocab_size, str_dtype_to_np,
                                 str_dtype_to_torch)
from tensorrt_llm.functional import is_gated_activation
from tensorrt_llm.models import WhisperEncoder
from tensorrt_llm.quantization import QuantMode

def load_weight(tensorrt_llm_whisper: WhisperEncoder,
                model_params : dict):
    tensorrt_llm.logger.info('Loading weights from PT...')
    tik = time.time()
    
    tensorrt_llm_whisper.conv1.weight.value = torch.unsqueeze(model_params['encoder.conv1.weight'], -1).numpy()
    tensorrt_llm_whisper.conv1.bias.value = model_params['encoder.conv1.bias'].numpy()
    tensorrt_llm_whisper.conv2.weight.value = torch.unsqueeze(model_params['encoder.conv2.weight'], -1).numpy()
    tensorrt_llm_whisper.conv2.bias.value = model_params['encoder.conv2.bias'].numpy()
    
    tensorrt_llm_whisper.temp_block.attn_ln.weight.value = model_params['encoder.blocks.0.attn_ln.weight'].numpy()
    tensorrt_llm_whisper.temp_block.attn_ln.bias.value = model_params['encoder.blocks.0.attn_ln.bias'].numpy()
    
    fused_weight = torch.cat([
        model_params['encoder.blocks.0.attn.query.weight'],
        model_params['encoder.blocks.0.attn.key.weight'],
        model_params['encoder.blocks.0.attn.value.weight']
    ],  dim=-1).transpose(0, 1).numpy()
    tensorrt_llm_whisper.temp_block.attention.qkv.weight.value =  fused_weight
    
    bias_shape = model_params['encoder.blocks.0.attn.query.bias'].shape
    fused_bias = torch.cat([
        model_params['encoder.blocks.0.attn.query.bias'],
        torch.zeros([*bias_shape]),
        model_params['encoder.blocks.0.attn.value.bias']
    ],  dim=-1).numpy()
    tensorrt_llm_whisper.temp_block.attention.qkv.bias.value =  fused_bias
    
    tensorrt_llm_whisper.temp_block.attention.dense.weight.value = model_params['encoder.blocks.0.attn.out.weight'].numpy()
    tensorrt_llm_whisper.temp_block.attention.dense.bias.value = model_params['encoder.blocks.0.attn.out.bias'].numpy()
    
    # tensorrt_llm_whisper.temp_block.mlp_ln.weight.value = model_params['encoder.blocks.0.mlp_ln.weight'].numpy()
    # tensorrt_llm_whisper.temp_block.mlp_ln.bias.value = model_params['encoder.blocks.0.mlp_ln.bias'].numpy()
    
    # tensorrt_llm_whisper.temp_block.mlp1.weight.value = model_params['encoder.blocks.0.mlp.0.weight'].numpy()
    # tensorrt_llm_whisper.temp_block.mlp1.bias.value = model_params['encoder.blocks.0.mlp.0.bias'].numpy()
    
    # tensorrt_llm_whisper.temp_block.mlp2.weight.value = model_params['encoder.blocks.0.mlp.2.weight'].numpy()
    # tensorrt_llm_whisper.temp_block.mlp2.bias.value = model_params['encoder.blocks.0.mlp.2.bias'].numpy()
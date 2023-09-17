import configparser
import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import (pad_vocab_size, str_dtype_to_np,
                                 str_dtype_to_torch)
from tensorrt_llm.functional import is_gated_activation
from tensorrt_llm.models import WhisperEncoder, WhisperDecoder, CrossAttn_KV
from tensorrt_llm.quantization import QuantMode

def fromfile(dir_path, name, shape=None, dtype=None):
    p = dir_path + '/' + name
    if Path(p).exists():
        t = np.fromfile(p, dtype=dtype)
        if shape is not None:
            t = t.reshape(shape)
        return t
    return None

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def trans_weight(weight):
    return np.ascontiguousarray(weight.numpy())

def load_encoder_weight(tensorrt_llm_whisper: WhisperEncoder,
                model_metadata : dict,
                model_params : dict,
                n_layer : int):
    tensorrt_llm.logger.info('Loading encoder weights from PT...')
    tensorrt_llm_whisper.positional_embedding.value = sinusoids(model_metadata['n_audio_ctx'], model_metadata['n_audio_state']).numpy()

    tensorrt_llm_whisper.conv1.weight.value = torch.unsqueeze(model_params['encoder.conv1.weight'], -1).numpy()
    tensorrt_llm_whisper.conv1.bias.value = model_params['encoder.conv1.bias'].numpy()
    tensorrt_llm_whisper.conv2.weight.value = torch.unsqueeze(model_params['encoder.conv2.weight'], -1).numpy()
    tensorrt_llm_whisper.conv2.bias.value = model_params['encoder.conv2.bias'].numpy()
    
    for i in range(n_layer):
    
        tensorrt_llm_whisper.blocks[i].attn_ln.weight.value = model_params['encoder.blocks.'+str(i)+'.attn_ln.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].attn_ln.bias.value = model_params['encoder.blocks.'+str(i)+'.attn_ln.bias'].numpy()
    
        fused_weight = torch.cat([
            model_params['encoder.blocks.'+str(i)+'.attn.query.weight'],
            model_params['encoder.blocks.'+str(i)+'.attn.key.weight'],
            model_params['encoder.blocks.'+str(i)+'.attn.value.weight']
        ],  dim=0).numpy()
        tensorrt_llm_whisper.blocks[i].attn.qkv.weight.value =  fused_weight
    
        bias_shape = model_params['encoder.blocks.'+str(i)+'.attn.query.bias'].shape
        fused_bias = torch.cat([
            model_params['encoder.blocks.'+str(i)+'.attn.query.bias'],
            torch.zeros([*bias_shape], dtype=torch.float16),
            model_params['encoder.blocks.'+str(i)+'.attn.value.bias']
        ],  dim=0).numpy()
        tensorrt_llm_whisper.blocks[i].attn.qkv.bias.value = fused_bias
    
        tensorrt_llm_whisper.blocks[i].attn.dense.weight.value = model_params['encoder.blocks.'+str(i)+'.attn.out.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].attn.dense.bias.value = model_params['encoder.blocks.'+str(i)+'.attn.out.bias'].numpy()
        tensorrt_llm_whisper.blocks[i].attn.dense.matmul_trans_weight = False
    
        tensorrt_llm_whisper.blocks[i].mlp_ln.weight.value = model_params['encoder.blocks.'+str(i)+'.mlp_ln.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].mlp_ln.bias.value = model_params['encoder.blocks.'+str(i)+'.mlp_ln.bias'].numpy()
    
        tensorrt_llm_whisper.blocks[i].mlp1.weight.value = trans_weight(model_params['encoder.blocks.'+str(i)+'.mlp.0.weight'])
        tensorrt_llm_whisper.blocks[i].mlp1.bias.value = trans_weight(model_params['encoder.blocks.'+str(i)+'.mlp.0.bias'])
        # tensorrt_llm_whisper.blocks[0].mlp1.matmul_trans_weight = False

        tensorrt_llm_whisper.blocks[i].mlp2.weight.value = trans_weight(model_params['encoder.blocks.'+str(i)+'.mlp.2.weight'])
        tensorrt_llm_whisper.blocks[i].mlp2.bias.value = trans_weight(model_params['encoder.blocks.'+str(i)+'.mlp.2.bias'])

    tensorrt_llm_whisper.ln_post.weight.value = model_params['encoder.ln_post.weight'].numpy()
    tensorrt_llm_whisper.ln_post.bias.value = model_params['encoder.ln_post.bias'].numpy()

def load_decoder_weight(
                tensorrt_llm_whisper: WhisperDecoder,
                model_params : dict,
                n_layer : int,
                quantize_dir : str,
                ):
    tensorrt_llm.logger.info('Loading decoder weights from PT...')
    # tensorrt_llm_whisper.positional_embedding.value = model_params['decoder.positional_embedding'].numpy()
    quant_mode = getattr(tensorrt_llm_whisper, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_whisper, "quant_mode", QuantMode(0))

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()
    
    tensorrt_llm_whisper.token_embedding.weight.value = model_params['decoder.token_embedding.weight'].numpy()
    
    for i in range(n_layer):

        tensorrt_llm_whisper.blocks[i].attn_ln.weight.value = model_params['decoder.blocks.'+str(i)+'.attn_ln.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].attn_ln.bias.value = model_params['decoder.blocks.'+str(i)+'.attn_ln.bias'].numpy()
    
        fused_weight = torch.cat([
            model_params['decoder.blocks.'+str(i)+'.attn.query.weight'],
            model_params['decoder.blocks.'+str(i)+'.attn.key.weight'],
            model_params['decoder.blocks.'+str(i)+'.attn.value.weight']
        ],  dim=0).numpy()
        tensorrt_llm_whisper.blocks[i].attn.qkv.weight.value =  fused_weight
    
        bias_shape = model_params['decoder.blocks.'+str(i)+'.attn.query.bias'].shape
        fused_bias = torch.cat([
            model_params['decoder.blocks.'+str(i)+'.attn.query.bias'],
            torch.zeros([*bias_shape], dtype=torch.float16),
            model_params['decoder.blocks.'+str(i)+'.attn.value.bias']
        ],  dim=0).numpy()
        tensorrt_llm_whisper.blocks[i].attn.qkv.bias.value = fused_bias
    
        tensorrt_llm_whisper.blocks[i].attn.dense.weight.value = model_params['decoder.blocks.'+str(i)+'.attn.out.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].attn.dense.bias.value = model_params['decoder.blocks.'+str(i)+'.attn.out.bias'].numpy()
        tensorrt_llm_whisper.blocks[i].attn.dense.matmul_trans_weight = False

        if use_int8_kv_cache:
            t = fromfile(
                quantize_dir, 
                'model.decoder.blocks.'+str(i)+'.attn.query_key_value.scale_y_quant_orig.bin',
                [1],
                np.float32)
            tensorrt_llm_whisper.blocks[i].attn.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_whisper.blocks[i].attn.kv_quant_orig_scale.value = t

        tensorrt_llm_whisper.blocks[i].cross_attn_ln.weight.value = model_params['decoder.blocks.'+str(i)+'.cross_attn_ln.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].cross_attn_ln.bias.value = model_params['decoder.blocks.'+str(i)+'.cross_attn_ln.bias'].numpy()

        tensorrt_llm_whisper.blocks[i].cross_attn.q_linear.weight.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.cross_attn.query.weight'])
        tensorrt_llm_whisper.blocks[i].cross_attn.q_linear.bias.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.cross_attn.query.bias'])

        tensorrt_llm_whisper.blocks[i].cross_attn.dense.weight.value = model_params['decoder.blocks.'+str(i)+'.cross_attn.out.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].cross_attn.dense.bias.value = model_params['decoder.blocks.'+str(i)+'.cross_attn.out.bias'].numpy()
        tensorrt_llm_whisper.blocks[i].cross_attn.dense.matmul_trans_weight = False

        tensorrt_llm_whisper.blocks[i].mlp_ln.weight.value = model_params['decoder.blocks.'+str(i)+'.mlp_ln.weight'].numpy()
        tensorrt_llm_whisper.blocks[i].mlp_ln.bias.value = model_params['decoder.blocks.'+str(i)+'.mlp_ln.bias'].numpy()
    
        tensorrt_llm_whisper.blocks[i].mlp1.weight.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.mlp.0.weight'])
        tensorrt_llm_whisper.blocks[i].mlp1.bias.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.mlp.0.bias'])
        # tensorrt_llm_whisper.blocks[0].mlp1.matmul_trans_weight = False

        tensorrt_llm_whisper.blocks[i].mlp2.weight.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.mlp.2.weight'])
        tensorrt_llm_whisper.blocks[i].mlp2.bias.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.mlp.2.bias'])

    tensorrt_llm_whisper.ln.weight.value = model_params['decoder.ln.weight'].numpy()
    tensorrt_llm_whisper.ln.bias.value = model_params['decoder.ln.bias'].numpy()

    tensorrt_llm_whisper.token_embedding_weight.value = model_params['decoder.token_embedding.weight'].numpy()

def load_crossattn_linear_weight(tensorrt_llm_whisper: CrossAttn_KV,
                model_params : dict,
                n_layer : int):
    tensorrt_llm.logger.info('Loading CrossAttn weights from PT...')
    
    for i in range(n_layer):
        tensorrt_llm_whisper.blocks[i].k_linear.weight.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.cross_attn.key.weight'])

        tensorrt_llm_whisper.blocks[i].v_linear.weight.value = trans_weight(model_params['decoder.blocks.'+str(i)+'.cross_attn.value.weight'])
        tensorrt_llm_whisper.blocks[i].v_linear.weight.bias = trans_weight(model_params['decoder.blocks.'+str(i)+'.cross_attn.value.bias'])

# model = torch.load("large-v2.pt")

# print(model['dims'])
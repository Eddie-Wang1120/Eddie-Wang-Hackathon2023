import os
import time
import argparse

import tensorrt as trt

import tensorrt_llm

from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.functional import Tensor, RaggedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.models import weight_only_quantize

from tensorrt_llm import str_dtype_to_trt

import numpy as np
import torch

from weight import load_encoder_weight, load_decoder_weight, load_crossattn_linear_weight

from collections import OrderedDict

from tensorrt_llm.quantization import QuantMode

MODEL_ENCODER_NAME = "whisper_encoder"
MODEL_DECODER_NAME = "whisper_decoder"
MODEL_CROSSATTN_NAME = "whsiper_crossattn"

def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)

def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default="large-v2.pt")
    parser.add_argument('--quantize_dir', type=str, default="quantize/1-gpu")
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='whisper_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )

    args = parser.parse_args(args)
    logger.set_level(args.log_level)

    plugins_args = [
        'use_gemm_plugin', 'use_layernorm_plugin', 'use_gpt_attention_plugin'
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"plugin_arg is None, setting it as {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    if args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    return args

def build_encoder(model, args):
    model_metadata = model['dims']
    model_params = model['model_state_dict']

    builder = Builder()

    max_batch_size = args.max_batch_size
    hidden_states = model_metadata['n_audio_state']
    num_heads = model_metadata['n_audio_head']
    num_layers = model_metadata['n_audio_layer']

    builder_config = builder.create_builder_config(
        name = MODEL_ENCODER_NAME,
        precision = 'float16',
        tensor_parallel=1,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_states,
        max_batch_size=max_batch_size,
        int8=args.quant_mode.has_act_and_weight_quant(),
    )
    
    tensorrt_llm_whisper_encoder = tensorrt_llm.models.WhisperEncoder(
        model_metadata['n_mels'],
        model_metadata['n_audio_ctx'],
        model_metadata['n_audio_state'],
        model_metadata['n_audio_head'],
        model_metadata['n_audio_layer'],
        str_dtype_to_trt("float16")
    )

    if args.use_weight_only:
        tensorrt_llm_whisper_encoder = weight_only_quantize(tensorrt_llm_whisper_encoder,
                                                args.quant_mode)

    load_encoder_weight(tensorrt_llm_whisper_encoder, model_metadata, model_params, model_metadata['n_audio_layer'])
    
    network = builder.create_network()

    if args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)

    with net_guard(network):

        inputs = tensorrt_llm_whisper_encoder.prepare_inputs()
        
        tensorrt_llm_whisper_encoder(inputs)
        
        for k, v in tensorrt_llm_whisper_encoder.named_network_outputs():
            network._mark_output(v, k,
                             str_dtype_to_trt('float16'))

    engine = None
    engine_name = get_engine_name(MODEL_ENCODER_NAME, 'float16', 1, 0)

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'encoder_config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))

def build_decoder(model, args):
    model_metadata = model['dims']
    model_params = model['model_state_dict']
    vocab_size = model_metadata['n_vocab']
    hidden_states = model_metadata['n_text_state']
    num_heads = model_metadata['n_text_head']
    num_layers = model_metadata['n_text_layer']
    num_text_ctx = model_metadata['n_text_ctx']
    num_audio = 1
    num_audio_ctx = model_metadata['n_audio_ctx']

    positional_embedding = model['model_state_dict']['decoder.positional_embedding']
    positional_embedding = positional_embedding.numpy()
    np.save(os.path.join(args.output_dir, 'positional_embedding.npy'), positional_embedding)

    builder = Builder()

    builder_config = builder.create_builder_config(
        name = MODEL_DECODER_NAME,
        precision=args.dtype,
        tensor_parallel=1,
        num_layers=num_layers,
        num_heads=num_heads,
        num_audio=num_audio,
        num_audio_ctx=num_audio_ctx,
        num_text_ctx=num_text_ctx,
        hidden_size=hidden_states,
        vocab_size=vocab_size,
        max_batch_size=args.max_batch_size,
        use_int8_kv_cache=args.quant_mode.has_int8_kv_cache(),
        int8=(args.quant_mode.has_act_and_weight_quant()
                  or args.quant_mode.has_int8_kv_cache()),
    )
    
    tensorrt_llm_whisper_decoder = tensorrt_llm.models.WhisperDecoder(
        model_metadata['n_vocab'],
        model_metadata['n_text_ctx'],
        model_metadata['n_text_state'],
        model_metadata['n_text_head'],
        model_metadata['n_text_layer'],
        str_dtype_to_trt(args.dtype),
        quant_mode=args.quant_mode,
    )

    if args.use_weight_only:
        tensorrt_llm_whisper_decoder = weight_only_quantize(tensorrt_llm_whisper_decoder,
                                                args.quant_mode)

    load_decoder_weight(
        tensorrt_llm_whisper_decoder, 
        model_params, 
        model_metadata['n_text_layer'],
        args.quantize_dir)
    
    network = builder.create_network()

    max_batch_size = args.max_batch_size
    max_input_len = args.max_input_len
    max_new_tokens = args.max_output_len
    max_beam_width = args.max_beam_width

    # if args.use_gemm_plugin:
    #     network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    # if args.use_layernorm_plugin:
    #     network.plugin_config.set_layernorm_plugin(
    #         dtype=args.use_layernorm_plugin)
    # if args.use_gpt_attention_plugin:
    #     network.plugin_config.set_gpt_attention_plugin(
    #         dtype=args.use_gpt_attention_plugin)
        
    if args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)

    with net_guard(network):

        inputs = tensorrt_llm_whisper_decoder.prepare_inputs(
            max_batch_size,
            max_input_len,
            max_new_tokens,
            max_beam_width,
            args.use_gpt_attention_plugin
        )
        
        tensorrt_llm_whisper_decoder(*inputs)
    
        for k, v in tensorrt_llm_whisper_decoder.named_network_outputs():
            network._mark_output(v, k,
                             str_dtype_to_trt(args.dtype))

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)

    engine = None
    engine_name = get_engine_name(MODEL_DECODER_NAME, args.dtype, 1, 0)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'decoder_config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))

def build_crossattn_kv_linear(model, args):
    model_metadata = model['dims']
    model_params = model['model_state_dict']
    num_heads = model_metadata['n_text_head']
    num_layers = model_metadata['n_text_layer']

    builder = Builder()

    builder_config = builder.create_builder_config(
        name = MODEL_CROSSATTN_NAME,
        precision = 'float16',
        num_layers=num_layers,
        num_heads=num_heads,
        int8=args.quant_mode.has_act_and_weight_quant(),
    )
    
    tensorrt_llm_whisper_crossattn = tensorrt_llm.models.CrossAttn_KV(
        model_metadata['n_text_state'],
        model_metadata['n_text_head'],
        model_metadata['n_text_layer'],
        str_dtype_to_trt('float16'),
    )

    if args.use_weight_only:
        tensorrt_llm_whisper_crossattn = weight_only_quantize(tensorrt_llm_whisper_crossattn,
                                                args.quant_mode)

    load_crossattn_linear_weight(tensorrt_llm_whisper_crossattn, model_params, model_metadata['n_text_layer'])

    network = builder.create_network()

    if args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)

    with net_guard(network):
        # print(tensorrt_llm_whisper.named_parameters())
        # network.set_named_parameters(tensorrt_llm_whisper_decoder.named_parameters())

        inputs = tensorrt_llm_whisper_crossattn.prepare_inputs()
        
        tensorrt_llm_whisper_crossattn(inputs)
    
        for k, v in tensorrt_llm_whisper_crossattn.named_network_outputs():
            network._mark_output(v, k,
                             str_dtype_to_trt('float16'))

    engine = None
    engine_name = get_engine_name(MODEL_CROSSATTN_NAME, 'float16', 1, 0)
    
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'cross_attn_config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))

def run_build(args=None):
    args = parse_arguments(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = torch.load(args.model_dir)
    build_encoder(model, args)
    build_decoder(model, args)
    build_crossattn_kv_linear(model, args)

if __name__ == '__main__':
    run_build()
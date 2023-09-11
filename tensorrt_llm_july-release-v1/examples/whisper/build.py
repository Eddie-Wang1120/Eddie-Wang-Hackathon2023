import os
import time

import tensorrt as trt

import tensorrt_llm

from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.functional import Tensor, RaggedTensor
from tensorrt_llm.logger import logger

from tensorrt_llm import str_dtype_to_trt

import numpy as np
import torch

from weight import load_encoder_weight, load_decoder_weight, load_crossattn_linear_weight

from collections import OrderedDict

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

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def build_encoder(model):
    model_metadata = model['dims']
    model_params = model['model_state_dict']

    # debug
    logger.set_level('verbose')

    builder = Builder()

    builder_config = builder.create_builder_config(
        name = MODEL_ENCODER_NAME,
        precision = 'float16'
    )

    tensorrt_llm_whisper_encoder = tensorrt_llm.models.WhisperEncoder(
        model_metadata['n_mels'],
        model_metadata['n_audio_ctx'],
        model_metadata['n_audio_state'],
        model_metadata['n_audio_head'],
        model_metadata['n_audio_layer'],
        sinusoids(model_metadata['n_audio_ctx'], model_metadata['n_audio_state']).numpy()
    )

    load_encoder_weight(tensorrt_llm_whisper_encoder, model_params, model_metadata['n_audio_layer'])
    
    network = builder.create_network()
    
    with net_guard(network):
        # print(tensorrt_llm_whisper.named_parameters())
        # network.set_named_parameters(tensorrt_llm_whisper.named_parameters())
        x = Tensor(
            name='x',
            dtype=trt.float16,
            shape=[1, 80, 3000],
            is_network_input=True
        )

        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[1])

        max_input_length = Tensor(name='max_input_length',
                                  dtype=trt.int32,
                                  shape=[1])

        x_ragged = RaggedTensor.from_row_lengths(
            x,
            input_lengths,
            max_input_length
        )

        tensorrt_llm_whisper_encoder(x_ragged)
    
        for k, v in tensorrt_llm_whisper_encoder.named_network_outputs():
            network._mark_output(v, k,
                             str_dtype_to_trt('float16'))

    engine = None
    engine_name = get_engine_name(MODEL_ENCODER_NAME, 'float16', 1, 0)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join('./', 'config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join('./', engine_name))

def build_decoder(model):
    model_metadata = model['dims']
    model_params = model['model_state_dict']
    
    # debug
    logger.set_level('verbose')

    builder = Builder()

    builder_config = builder.create_builder_config(
        name = MODEL_ENCODER_NAME,
        precision = 'float16'
    )
    
    tensorrt_llm_whisper_decoder = tensorrt_llm.models.WhisperDecoder(
        model_metadata['n_vocab'],
        model_metadata['n_text_ctx'],
        model_metadata['n_text_state'],
        model_metadata['n_text_head'],
        model_metadata['n_text_layer'],
        str_dtype_to_trt('float16')
    )

    load_decoder_weight(tensorrt_llm_whisper_decoder, model_params, model_metadata['n_text_layer'])
    
    network = builder.create_network()
    
    with net_guard(network):
        # print(tensorrt_llm_whisper.named_parameters())
        # network.set_named_parameters(tensorrt_llm_whisper_decoder.named_parameters())

        inputs = tensorrt_llm_whisper_decoder.prepare_inputs(256,
                                                 200,
                                                 200, True,
                                                 1)
        
        tensorrt_llm_whisper_decoder(*inputs)
    
        for k, v in tensorrt_llm_whisper_decoder.named_network_outputs():
            network._mark_output(v, k,
                             str_dtype_to_trt('float16'))

    engine = None
    engine_name = get_engine_name(MODEL_DECODER_NAME, 'float16', 1, 0)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join('./', 'config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join('./', engine_name))

def build_crossattn_kv_linear(model):
    model_metadata = model['dims']
    model_params = model['model_state_dict']
    
    # debug
    logger.set_level('verbose')

    builder = Builder()

    builder_config = builder.create_builder_config(
        name = MODEL_ENCODER_NAME,
        precision = 'float16'
    )
    
    tensorrt_llm_whisper_crossattn = tensorrt_llm.models.CrossAttn_KV(
        model_metadata['n_text_state'],
        model_metadata['n_text_head'],
        model_metadata['n_text_layer'],
        str_dtype_to_trt('float16')
    )

    load_crossattn_linear_weight(tensorrt_llm_whisper_crossattn, model_params, model_metadata['n_text_layer'])

    network = builder.create_network()
    
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

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join('./', 'config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join('./', engine_name))

if __name__ == '__main__':
    model = torch.load("large-v2.pt")
    build_encoder(model)
    build_decoder(model)
    build_crossattn_kv_linear(model)
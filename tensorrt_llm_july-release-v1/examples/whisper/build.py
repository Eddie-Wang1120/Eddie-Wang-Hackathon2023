import os
import time

import tensorrt as trt

import tensorrt_llm

from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.functional import Tensor
from tensorrt_llm.logger import logger

import numpy as np
import torch

from weight import load_weight

MODEL_NAME = "whisper"

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


model = torch.load("large-v2.pt")
model_metadata = model['dims']
model_params = model['model_state_dict']

builder = Builder()

builder_config = builder.create_builder_config(
    name = MODEL_NAME,
    precision = 'float16'
)

tensorrt_llm_whisper = tensorrt_llm.models.WhisperEncoder(
    model_metadata['n_mels'],
    model_metadata['n_audio_ctx'],
    model_metadata['n_audio_state'],
    model_metadata['n_audio_head'],
    model_metadata['n_audio_layer'],
    sinusoids(model_metadata['n_audio_ctx'], model_metadata['n_audio_state']).numpy()
)

load_weight(tensorrt_llm_whisper, model_params)

network = builder.create_network()

# tensorrt_llm_whisper.linear1.weight.value = np.random.randn(16, 32)

with net_guard(network):
    network.set_named_parameters(tensorrt_llm_whisper.named_parameters())
    x = Tensor(
        name='x',
        dtype=trt.float16,
        shape=[1, 80, 3000],
        is_network_input=True
    )
    tensorrt_llm_whisper(x)

# print(tensorrt_llm_whisper.linear1)
# print(tensorrt_llm_whisper.linear2.weight.value.shape)

engine = None
engine_name = get_engine_name(MODEL_NAME, 'float16', 1, 0)

engine = builder.build_engine(network, builder_config)

config_path = os.path.join('./', 'config.json')
builder.save_config(builder_config, config_path)

serialize_engine(engine, os.path.join('./', engine_name))
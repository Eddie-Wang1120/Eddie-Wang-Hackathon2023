import argparse
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

import tensorrt_llm

from build import get_engine_name
from whisper_utils import load_audio, pad_or_trim, log_mel_spectrogram

from encoding import WhisperEncoding
from decoding import WhisperDecoding

from tensorrt_llm.runtime.session import Session, TensorInfo

import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--input_file',
                        type=str,
                        default='test.m4a')
    
    return parser.parse_args()

def generate(
    log_level: str = 'error',
    engine_dir: str = '',
    input_file: str = 'test.m4a',
    model_file: str = 'large-v2.pt'
):
    tensorrt_llm.logger.set_level(log_level)
    
    model = torch.load(model_file)
    model_metadata = OrderedDict(model['dims'])
    positional_embedding = model['model_state_dict']['decoder.positional_embedding']
    del model
    
    engine_dir = Path('./')
    config_path = engine_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    # with open(engine_dir / get_engine_name('whisper_decoder', dtype, world_size, runtime_rank), 'rb') as f:
    #     session = Session.from_serialized_engine(f.read())
    
    # print(session._print_io_info())
    
    audio = load_audio(input_file)
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to('cuda').type(torch.float16)
    mel = mel.unsqueeze(0)
    
    model_metadata.update({'n_audio': mel.shape[0]})

    whisper_encoding = WhisperEncoding(
        engine_dir / get_engine_name('whisper_encoder', dtype, world_size, runtime_rank),
        'float16'
    )
    
    whisper_decoding = WhisperDecoding(
        engine_dir / get_engine_name('whisper_decoder', dtype, world_size, runtime_rank),
        engine_dir / get_engine_name('whsiper_crossattn', dtype, world_size, runtime_rank),
        'float16',
        True,
        'en',
        'transcribe',
        model_metadata,
        positional_embedding
    )
    
    begin_time = time.time()
    
    audio_features = whisper_encoding.get_audio_features(mel)
    
    languages, language_probs = whisper_decoding.detect_language(audio_features)

    tokens, sum_logprobs, no_speech_probs = whisper_decoding.main_loop(audio_features)

    result = whisper_decoding.post_process(tokens, sum_logprobs, no_speech_probs, audio_features, languages)
    
    print("transcribe time " + str(time.time() - begin_time))
    
    result = result[0]
    print(result.text)

if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
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
    parser.add_argument('--engine_dir', type=str, default='whisper_outputs')
    parser.add_argument('--input_file',
                        type=str,
                        default='test.m4a')
    
    return parser.parse_args()

def generate(
    log_level: str = 'error',
    engine_dir: str = 'whisper_outputs',
    input_file: str = 'test.m4a',
):
    tensorrt_llm.logger.set_level(log_level)

    world_size = 1
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
    
    audio = load_audio(input_file)
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to('cuda').type(torch.float16)
    mel = mel.unsqueeze(0)
    
    engine_dir = Path(engine_dir)
    # serialize_path = engine_dir / get_engine_name('whisper_decoder', 'float16', 1, 0)
        
    # with open(serialize_path, 'rb') as f:
    #     decoder_session = Session.from_serialized_engine(f.read())

    # print(decoder_session._print_io_info())
    # asdasdasfsa

    whisper_encoding = WhisperEncoding(
        engine_dir,
    )
    
    whisper_decoding = WhisperDecoding(
        engine_dir,
    )
    
    #begin_time = time.time()
    
    audio_features = whisper_encoding.get_audio_features(mel)
    
    languages, language_probs = whisper_decoding.detect_language(audio_features)

    begin_time = time.time()
    for _ in range(10):
        tokens, sum_logprobs, no_speech_probs = whisper_decoding.main_loop(audio_features)

    print("transcribe time " + str(time.time() - begin_time))
    result = whisper_decoding.post_process(tokens, sum_logprobs, no_speech_probs, audio_features, languages)
    
    #print("transcribe time " + str(time.time() - begin_time))
    
    result = result[0]
    print(result.text)

if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))

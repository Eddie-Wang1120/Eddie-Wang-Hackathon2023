import argparse
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger

from build import get_engine_name  # isort:skip
from whisper_utils import load_audio, pad_or_trim, log_mel_spectrogram

from encoding import WhisperEncoding
from decoding import WhisperDecoding

from torch_model import ModelDimensions, Whisper

from datasets import load_dataset, load_metric

def eval_tensorrt_llm(whisper_encoding, whisper_decoding, mel):
    audio_features = whisper_encoding.get_audio_features(mel)
    
    languages, language_probs = whisper_decoding.detect_language(audio_features)

    tokens, sum_logprobs, no_speech_probs = whisper_decoding.main_loop(audio_features)

    result = whisper_decoding.post_process(tokens, sum_logprobs, no_speech_probs, audio_features, languages)
    
    result = result[0]
    return result

def eval_torch(whisper_encoding, whisper_decoding, mel, model):

    audio_features = whisper_encoding.torch_get_audio_features(model, mel)
    
    languages, language_probs = whisper_decoding.torch_detect_language(model, audio_features)

    tokens, sum_logprobs, no_speech_probs = whisper_decoding.torch_main_loop(model, audio_features)

    result = whisper_decoding.post_process(tokens, sum_logprobs, no_speech_probs, audio_features, languages)
    
    result = result[0]
    return result

def main(args):
    tensorrt_llm.logger.set_level(args.log_level)
    
    test_trt_llm = args.test_trt_llm
    test_torch = args.test_torch
    
    checkpoint = torch.load(args.checkpoint_file)
    model_metadata = OrderedDict(checkpoint['dims'])
    positional_embedding = checkpoint['model_state_dict']['decoder.positional_embedding']

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to('cuda')
    
    del checkpoint

    engine_dir = Path(args.engine_dir)
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
    
    audio = load_audio(args.input_file)
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
    
    data_label = []
    data_label.append("This audio is for the test of NVIDIA Inference Competition 2023. The main goal of the test is to check the accuracy and the performance of the optimized Whisper model.")

    data_idx = 0
    if test_torch:    
        profiler.start('torch')
        result = eval_torch(whisper_encoding, whisper_decoding, mel, model)
        profiler.stop('torch')
        output_torch = []
        output_torch.append(result.text)
        logger.info(
            "---------------------------------------------------------")
        logger.info("Torch Generated : ")
        logger.info(f" Input : {args.input_file}")
        logger.info(f"\n Reference : {data_label[data_idx]}")
        logger.info(f"\n Output : {output_torch[data_idx]}")
        logger.info(
            "---------------------------------------------------------")

        
    if test_trt_llm:    
        profiler.start('tensorrt_llm')
        result = eval_tensorrt_llm(whisper_encoding, whisper_decoding, mel)
        profiler.stop('tensorrt_llm')
        output_tensorrt_llm = []
        output_tensorrt_llm.append(result.text)
        logger.info(
            "---------------------------------------------------------")
        logger.info("TensorRT-LLM Generated : ")
        logger.info(f" Input : {args.input_file}")
        logger.info(f"\n Reference : {data_label[data_idx]}")
        logger.info(f"\n Output : {output_tensorrt_llm[data_idx]}")
        logger.info(
            "---------------------------------------------------------")

    metric_tensorrt_llm = load_metric("rouge")
    metric_torch = load_metric("rouge")

    if test_torch:
        metric_torch.add_batch(
            predictions=[
                output_torch[data_idx]
            ],
            references=[
                data_label[data_idx]
            ]
        )

    if test_trt_llm:
        metric_tensorrt_llm.add_batch(
            predictions=[
                output_tensorrt_llm[data_idx]
            ],
            references=[
                data_label[data_idx]
        ]
            )

    if test_torch:
        np.random.seed(0)  # rouge score use sampling to compute the score
        logger.info(
            f'Torch (total latency: {profiler.elapsed_time_in_sec("torch")} sec)'
        )
        logger.info(f"Torch beam 0 result")
        computed_metrics_torch = metric_torch.compute()
        for key in computed_metrics_torch.keys():
            logger.info(
                f'  {key} : {computed_metrics_torch[key].mid[2]*100}')

    if test_trt_llm:
        np.random.seed(0)  # rouge score use sampling to compute the score
        logger.info(
            f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
        )
        logger.info(f"TensorRT-LLM beam 0 result")
        computed_metrics_tensorrt_llm = metric_tensorrt_llm.compute()
        for key in computed_metrics_tensorrt_llm.keys():
            logger.info(
                f'  {key} : {computed_metrics_tensorrt_llm[key].mid[2]*100}')



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_torch', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp16'],
                        default='fp16')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='./')
    parser.add_argument('--input_file', type=str, default='./test.m4a')
    parser.add_argument('--checkpoint_file', type=str, default='./large-v2.pt')
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    args = parser.parse_args()
    main(args)

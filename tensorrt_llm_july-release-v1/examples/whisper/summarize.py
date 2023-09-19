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

from normalizers import EnglishTextNormalizer

from tqdm.notebook import tqdm

import pandas as pd
import jiwer

import string
import re

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

def load_dataset(dataset_dir):
    label_file = None
    audio_file = []
    for file in dataset_dir.iterdir():
        if str(file).endswith('txt'):
            label_file = file
        else:
            audio_file.append(file)
    
    references = []
    with open(label_file, 'r') as f:
        for line in f:
            references.append((str(line).split(' ', 1))[1].replace('\n', ''))
    
    return audio_file, references

def main(args):
    tensorrt_llm.logger.set_level(args.log_level)
    
    test_trt_llm = args.test_trt_llm
    test_torch = args.test_torch
    
    checkpoint = torch.load(args.checkpoint_file)
    model_metadata = OrderedDict(checkpoint['dims'])

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to('cuda')
    
    del checkpoint

    world_size = 1
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    model_metadata.update({'n_audio': 1})

    engine_dir = Path(args.engine_dir)

    whisper_encoding = WhisperEncoding(
        engine_dir,
    )

    whisper_decoding = WhisperDecoding(
        engine_dir,
    )
    output_torch = []
    output_tensorrt_llm = []
    output_reference = []

    dataset_dir = Path(args.dataset_dir)
    for dir in dataset_dir.iterdir():
        for child_dir in dir.iterdir():
            
            audio_files, references = load_dataset(child_dir)
            audio_files = sorted(audio_files)
            
            for i, audio_file in enumerate(audio_files):
            
                audio = load_audio(audio_file)
                if audio.shape[-1] > 480000:
                    continue
                audio = pad_or_trim(audio)
                mel = log_mel_spectrogram(audio).to('cuda').type(torch.float16)
                mel = mel.unsqueeze(0)
                output_reference.append(references[i])
                
                if test_torch:    
                    profiler.start('torch')
                    result = eval_torch(whisper_encoding, whisper_decoding, mel, model)
                    profiler.stop('torch')
                    punctuations = re.findall(r'[.,!?]', result.text) 
                    table = str.maketrans({p: '' for p in punctuations})
                    result = result.text.translate(table).upper()
                    output_torch.append(result)
                    logger.info(
                        "---------------------------------------------------------")
                    logger.info("Torch Generated : ")
                    logger.info(f" Input : {audio_file}")
                    logger.info(f"\n Reference : {references[i]}")
                    logger.info(f"\n Output : {result}")
                    logger.info(
                        "---------------------------------------------------------")

                if test_trt_llm:    
                    profiler.start('tensorrt_llm')
                    result = eval_tensorrt_llm(whisper_encoding, whisper_decoding, mel)
                    profiler.stop('tensorrt_llm')
                    punctuations = re.findall(r'[.,!?]', result.text) 
                    table = str.maketrans({p: '' for p in punctuations})
                    result = result.text.translate(table).upper()
                    output_tensorrt_llm.append(result)
                    logger.info(
                        "---------------------------------------------------------")
                    logger.info("TensorRT-LLM Generated : ")
                    logger.info(f" Input : {audio_file}")
                    logger.info(f"\n Reference : {references[i]}")
                    logger.info(f"\n Output : {result}")
                    logger.info(
                    "---------------------------------------------------------")
    
    normalizer = EnglishTextNormalizer()
    
    if test_torch:
        data = pd.DataFrame(dict(hypothesis=output_torch, reference=output_reference))
        data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
        data["reference_clean"] = [normalizer(text) for text in data["reference"]]
        wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
        logger.info(
            f'Torch (total latency: {profiler.elapsed_time_in_sec("torch")} sec)'
        )
        logger.info(f"Torch beam 0 result")
        logger.info(f"\nWER: {wer * 100:.2f} %")

    if test_trt_llm:
        data = pd.DataFrame(dict(hypothesis=output_tensorrt_llm, reference=output_reference))
        data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
        data["reference_clean"] = [normalizer(text) for text in data["reference"]]
        wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
        logger.info(
            f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
        )
        logger.info(f"TensorRT-LLM beam 0 result")
        logger.info(f"\nWER: {wer * 100:.2f} %")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_torch', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp16'],
                        default='fp16')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='whisper_outputs')
    parser.add_argument('--dataset_dir', type=str, default='./LibriSpeech/test-clean')
    parser.add_argument('--checkpoint_file', type=str, default='./large-v2.pt')
    args = parser.parse_args()
    main(args)

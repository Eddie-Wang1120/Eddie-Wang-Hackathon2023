'''
Utilities for SmoothQuant models
'''

import functools
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from decoding import WhisperDecoding
from encoding import WhisperEncoding

from pathlib import Path

from whisper_utils import load_audio, pad_or_trim, log_mel_spectrogram

@torch.no_grad()
def apply_smoothing(scales,
                    gemm_weights,
                    layernorm_weights=None,
                    layernorm_bias=None,
                    dtype=torch.float32,
                    layernorm_1p=False):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha=0.5,
                weight_scales=None):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                    orig_dtype)

    return scales

def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
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
            references.append(line[15:-1])
    
    return audio_file, references


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)

    if ln is not None:
        ln.weight.div_(scales)
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def capture_activation_range(model,
                             engine_dir,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=0)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    whisper_encoding = WhisperEncoding(engine_dir, only_torch=True)
    whisper_decoding = WhisperDecoding(engine_dir, only_torch=True)

    audio_files, _ = load_dataset(dataset)

    for audio_file in tqdm(audio_files, desc="calibrating model"):
        audio = load_audio(audio_file)
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to('cuda').type(torch.float16)
        mel = mel.unsqueeze(0)
        
        audio_features = whisper_encoding.torch_get_audio_features(model, mel)
    
        languages, language_probs = whisper_decoding.torch_detect_language(model, audio_features)

        tokens, sum_logprobs, no_speech_probs = whisper_decoding.torch_main_loop(model, audio_features)


    for h in hooks:
        h.remove()

    return act_scales

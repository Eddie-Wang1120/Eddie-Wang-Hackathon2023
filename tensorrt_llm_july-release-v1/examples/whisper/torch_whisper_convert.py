import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

from torch_model import ModelDimensions, Whisper

from utils.convert import split_and_save_weight

from collections import OrderedDict

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

@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "gpt"
    storage_type: str = "fp32"
    dataset_dir: str = None
    engine_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="whisper",
            type=str,
            help="Specify GPT variants to convert checkpoints correctly",
            choices=["whisper"])
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float16",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset_dir",
                            type=str,
                            default='./LibriSpeech/test',
                            help="cache dir to load the hugging face dataset")
        parser.add_argument("--engine_dir",
                            type=str,
                            default='whisper_outputs',
                            help="cache dir to load the hugging face dataset")
        return ProgArgs(**vars(parser.parse_args(args)))

def concat_qkv_weight_bias(q, hf_key, hf_model):
    bias_shape = q.shape
    if 'key.bias' in hf_key.replace("attn.query", "attn.key"):
        k = torch.zeros([*bias_shape], dtype=torch.float16).to('cuda')
    else:
        k = hf_model.state_dict()[hf_key.replace("attn.query", "attn.key")]
    v = hf_model.state_dict()[hf_key.replace("attn.query", "attn.value")]
    return torch.cat([q, k, v], dim=0)

@torch.no_grad()
def torch_whisper_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = True if args.model in ["santacoder", "starcoder"
                                              ] else False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.in_file)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to('cuda')

    engine_dir = Path(args.engine_dir)
    # # load position_embedding from rank 0
    # model = AutoModelForCausalLM.from_pretrained(args.in_file,
    #                                              device_map="auto",
    #                                              trust_remote_code=True)
    
    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        act_range = capture_activation_range(
            model, engine_dir, args.dataset_dir)
        # if args.smoothquant is not None:
        #     smooth_gpt_model(model, act_range, args.smoothquant)
    
    
    pop_list = []
    add_list = []
    for key in act_range.keys():
        if 'attn.query' in key or 'cross_attn.query' in key:
            ft_key = key.replace("query", "query_key_value")
            q_key = key
            k_key = key.replace("query", "key")
            v_key = key.replace("query", "value")
            q_act_range = act_range.get(q_key)
            k_act_range = act_range.get(k_key)
            v_act_range = act_range.get(v_key)
            qkv_act_range = {'x': torch.cat((q_act_range['x'], k_act_range['x'], v_act_range['x']), dim=0),
                             'y': torch.cat((q_act_range['y'], k_act_range['y'], v_act_range['y']), dim=0),
                             'w': torch.cat((q_act_range['w'], k_act_range['w'], v_act_range['w']), dim=0)
                             }
            # act_range[ft_key] = qkv_act_range
            add_list.append({ft_key : qkv_act_range})
            pop_list.append(q_key)
            pop_list.append(k_key)
            pop_list.append(v_key)
            # act_range.pop(q_key)
            # act_range.pop(k_key)
            # act_range.pop(v_key)
    for add_item in add_list:
        act_range.update(add_item)
    for pop_name in pop_list:
        act_range.pop(pop_name)
        
    config = configparser.ConfigParser()
    config["whisper"] = {}
    for key in vars(args):
        config["whisper"][key] = f"{vars(args)[key]}"
    # for k, v in vars(model.config).items():
    #     config["whisper"][k] = f"{v}"
    config["whisper"]["storage_dtype"] = args.storage_type
    config["whisper"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    global_ft_weights = [
        "model.wpe", "model.wte", "model.final_layernorm.bias",
        "model.final_layernorm.weight", "model.lm_head.weight"
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        ft_name = name

        if 'attn.query' in name or 'cross_attn.query' in name:
            param = concat_qkv_weight_bias(param, name, model)
            ft_name = ft_name.replace("query", "query_key_value")
            # q_name = name.replace(".weight", "")
            # k_name = name.replace("query", "key").replace(".weight", "")
            # v_name = name.replace("query", "value").replace(".weight", "")
            # q_act_range = act_range.get(q_name)
            # k_act_range = act_range.get(k_name)
            # v_act_range = act_range.get(v_name)
            # print(k_name)
            # print(k_act_range)
            
            # qkv_act_range = {'x': torch.cat((q_act_range['x'], k_act_range['x'], v_act_range['x']), dim=0),
            #                  'y': torch.cat((q_act_range['y'], k_act_range['y'], v_act_range['y']), dim=0),
            #                  'w': torch.cat((q_act_range['w'], k_act_range['w'], v_act_range['w']), dim=0)
            #                  }
            # print(qkv_act_range)
        
        starmap_args.append(
                (0, saved_dir, infer_tp, ft_name, param.to(storage_type),
                 storage_type, act_range.get(ft_name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": multi_query_mode,
                     "local_dim": None
                 }))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    torch_whisper_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())

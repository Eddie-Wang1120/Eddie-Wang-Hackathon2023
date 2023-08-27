from pathlib import Path
import json
import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.logger import logger

import torch

from build import get_engine_name  # isort:skip

logger.set_level('verbose')
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

engine_name = get_engine_name('whisper', dtype, world_size, runtime_rank)
serialize_path = engine_dir / engine_name
with open(serialize_path, 'rb') as f:
    session = Session.from_serialized_engine(f.read())

torch_dtype = str_dtype_to_torch(dtype) if isinstance(dtype, str) else dtype

shape = (1, 80, 3000)
x = torch.ones([1, 80, 3000], dtype=torch.float16, device='cuda')
inputs = {'x' : x}

output_info = session.infer_shapes([
        TensorInfo(name, str_dtype_to_trt(dtype), tensor.shape)
        for name, tensor in inputs.items()
    ])

logger.debug(f'output info {output_info}')
outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
        for t in output_info
    }

# Execute model inference
stream = torch.cuda.current_stream()
ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
assert ok, 'Engine execution failed'
stream.synchronize()
print(outputs['add_output'])
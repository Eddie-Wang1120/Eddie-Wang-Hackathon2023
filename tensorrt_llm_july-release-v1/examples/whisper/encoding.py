from collections import OrderedDict
import torch

from tensorrt_llm.runtime.session import Session, TensorInfo
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from build import get_engine_name
import json

class WhisperEncoding:
    def __init__(
        self, 
        engine_dir,
        only_torch : bool = False
        ):
        
        if not only_torch:
            self.session = self.get_session(engine_dir)
        self.dtype = 'float16'

    def get_session(self, engine_dir):
        config_path = engine_dir / 'encoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
        dtype = config['builder_config']['precision']
        world_size = config['builder_config']['tensor_parallel']
        num_heads = config['builder_config']['num_heads'] // world_size
        hidden_size = config['builder_config']['hidden_size'] // world_size
        num_layers = config['builder_config']['num_layers']
        
        self.dtype = dtype
        
        serialize_path = engine_dir / get_engine_name('whisper_encoder', self.dtype, world_size, 0)
        
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        
        return session 
    
    def torch_get_audio_features(self, model, mel):
        with torch.no_grad():
            audio_features = model.encoder(mel)
        return audio_features
    
    def get_audio_features(self, mel):
        inputs = OrderedDict()
        output_list = []
        
        inputs.update({'x':mel})
        output_list.append(TensorInfo('x', str_dtype_to_trt("float16"), mel.shape))
        
        input_lengths = torch.tensor((1,), dtype=torch.int32, device='cuda')
        inputs.update({'input_lengths':input_lengths})
        output_list.append(TensorInfo('input_lengths', str_dtype_to_trt("int32"), input_lengths.shape))
    
        max_input_length = torch.tensor((1,), dtype=torch.int32, device='cuda')
        inputs.update({'max_input_length':max_input_length})
        output_list.append(TensorInfo('max_input_length', str_dtype_to_trt("int32"), max_input_length.shape))
        
        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
                t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
                for t in output_info
            }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['output']
        return audio_features
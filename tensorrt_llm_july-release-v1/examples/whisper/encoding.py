from collections import OrderedDict
import torch

from tensorrt_llm.runtime.session import Session, TensorInfo
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)

class WhisperEncoding:
    def __init__(
        self, 
        serialize_path, 
        dtype
        ):
        
        self.session = self.get_session(serialize_path)
        self.dtype = dtype

    def get_session(self, serialize_path):
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        # print(session._print_io_info())
        return session 
    
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
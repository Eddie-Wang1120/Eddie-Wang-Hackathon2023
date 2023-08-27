from typing import Dict, Iterable, Optional

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt

from ...functional import gelu, transpose, constant, Tensor, RaggedTensor
from ...layers import Linear, Conv2d, Conv1d, LayerNorm, Attention
from ...module import Module, ModuleList

class ResidualAttentionBlock(Module):
    def __init__(self,
                 n_state: int,
                 n_head : int,
                 cross_attention: bool = False
                 ):
        super().__init__()
        self.attn_ln = LayerNorm(n_state, dtype=str_dtype_to_trt("float16"))
        self.cross_attn = (
            Attention(
                n_state,
                n_head,
                1280,
                bias=True,
                dtype=str_dtype_to_trt("float16")
            ) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.attention = Attention(
            n_state,
            n_head,
            1280,
            bias=True,
            dtype=str_dtype_to_trt("float16"),
        )
        # n_mlp = n_state*4
        # self.mlp_ln = LayerNorm(n_state)
        # self.mlp1 = Linear(n_state, n_mlp)
        # self.mlp2 = Linear(n_mlp, n_state)
    
    def forward(self,
                x: Tensor
                ):
        residual1 = x
        x = self.attn_ln(x)
        hidden_states = RaggedTensor.from_row_lengths(x,
                                          1280,
                                          1280)
        x = (self.attention(hidden_states)).data
        # x = (self.attention(hidden_states)).data + residual1
        # residual2 = x
        # # if self.cross_attn: TODO
        # x = self.mlp_ln(x)
        # x = self.mlp1(x)
        # x = gelu(x)
        # x = self.mlp2(x)
        # x = x + residual2
        return x

class WhisperEncoder(Module):
    def __init__(self, 
                 n_mels: int, 
                 n_ctx: int, 
                 n_state: int, 
                 n_head: int, 
                 n_layer: int,
                 positional_embedding,
                 ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.gelu = gelu
        self.permute = transpose
        self.positional_embedding = positional_embedding

        # self.blocks = ModuleList([
        #     ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)
        # ])
    
        self.temp_block = ResidualAttentionBlock(n_state, n_head)
    
    def forward(self, x):
        positional_embedding_buffer = constant(self.positional_embedding)
        
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x) # minor miss
        x = self.permute(x, 2, 1)
        x = x + positional_embedding_buffer
        
        x = self.temp_block(x)
        
        x.mark_output('add_output', str_dtype_to_trt("float16"))
        return x
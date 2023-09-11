from typing import Dict, Iterable, Optional

from ..._common import default_net, precision
from ..._utils import pad_vocab_size, str_dtype_to_trt

from ...functional import clip, concat, view, shape, permute, cast, matmul, gelu, transpose, constant, slice, Tensor, RaggedTensor
from ...layers import Linear, Conv2d, Conv1d, LayerNorm, Attention, Gelu, Embedding
from ...module import Module, ModuleList
from ...parameter import Parameter

import tensorrt as trt
from collections import OrderedDict

class ResidualAttentionBlock(Module):
    def __init__(self,
                 n_state: int,
                 n_head : int,
                 n_ctx : int,
                 cross_attention: bool = False
                 ):
        super().__init__()
        self.attn_ln = LayerNorm(n_state, dtype=str_dtype_to_trt("float16"))

        self.attn = Attention(
            n_state,
            n_head,
            n_ctx,
            bias=True,
            dtype=str_dtype_to_trt("float16"),
            
        )

        self.cross_attn = (
            Attention(
                n_state,
                n_head,
                n_ctx,
                cross_attention=True,
                bias=True,
                dtype=str_dtype_to_trt("float16")
            ) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state*4
        self.mlp_ln = LayerNorm(n_state)

        self.mlp1 = Linear(n_state, n_mlp, bias=True, dtype=str_dtype_to_trt("float16"))
        self.mlp2 = Linear(n_mlp, n_state, bias=True, dtype=str_dtype_to_trt("float16"))
    
    def forward(self,
                x: RaggedTensor,
                mask: Optional[Tensor] = None,
                multi_kv_cache: Optional[Tensor] = None,
                cross_kv_cache: Optional[Tensor] = None,
                use_cache: Optional[bool] = False
                ):
        row_length = x.row_lengths
        max_row_length = x.max_row_length
        residual1 = x.data
        hidden_states = self.attn_ln(x.data)
        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                          row_length,
                                          max_row_length)
        
        # x = (self.attn(hidden_states, 
        #                attention_mask=mask,
        #                past_key_value=multi_kv_cache,
        #                use_cache = use_cache
        #                )).data + residual1

        # self.register_network_output("before_mul_attn", hidden_states.data)
        # self.register_network_output("mask", mask)
        # self.register_network_output("kv_cache", multi_kv_cache)

        attention_output = self.attn(hidden_states, 
                       attention_mask=mask,
                       past_key_value=multi_kv_cache,
                       use_cache=use_cache
                       )

        if use_cache:
            attention_output, presents = attention_output
        
        # self.register_network_output("after_mul_attn", attention_output.data)
        
        x = attention_output.data + residual1
        
        # self.register_network_output("after_mul_attn_add", x)

        if self.cross_attn:
            cross_residual = x
            x = self.cross_attn_ln(x)
            cross_hidden_states = RaggedTensor.from_row_lengths(
                x,
                row_length,
                max_row_length
            )
            # x = cross_residual + self.cross_attn(cross_hidden_states, xa=xa).data
            x = cross_residual + self.cross_attn(
                cross_hidden_states, 
                cross_key_value=cross_kv_cache).data
            
        # self.register_network_output("after_mul_attn_add", x)
            
        residual2 = x
        x = self.mlp_ln(x)
        x = self.mlp1(x)
        x = gelu(x)
        x = self.mlp2(x)
        x = x + residual2
        x = RaggedTensor.from_row_lengths(x,
                                          row_length,
                                          max_row_length)
        
        if use_cache:
            return (x, presents)
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

        self.blocks = ModuleList([
            ResidualAttentionBlock(n_state, n_head, n_ctx) for _ in range(n_layer)
        ])
    
        self.ln_post = LayerNorm(n_state)
    
    def forward(self, x: RaggedTensor):
        positional_embedding_buffer = constant(self.positional_embedding)
        row_length = x.row_lengths
        max_row_length = x.max_row_length
        x = x.data
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x) # minor miss
        x = self.permute(x, 2, 1)
        x = x + positional_embedding_buffer
        
        hidden_states = RaggedTensor.from_row_lengths(
            x,
            row_length,
            max_row_length
        )
        
        for block in self.blocks:
            hidden_states = block(hidden_states)
        x = hidden_states.data
        x = self.ln_post(x)
        x.mark_output('output', str_dtype_to_trt("float16"))
        return x
    
class WhisperDecoder(Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dtype):
        super().__init__()
        
        self.token_embedding = Embedding(n_vocab, n_state, dtype=dtype)
        # self.positional_embedding = Parameter(shape=(n_ctx, n_state), dtype=dtype)
        self.n_state = n_state
        self.n_layer = n_layer
        self.blocks = ModuleList([
            ResidualAttentionBlock(n_state, n_head, n_ctx, cross_attention=True) for _ in range(n_layer)
        ])
        
        # self.block = ResidualAttentionBlock(n_state, n_head, cross_attention=True)
        
        self.ln = LayerNorm(n_state)
        
        self.token_embedding_weight = Parameter(shape=(n_vocab, n_state), dtype=dtype)
    
    def forward(self, 
                x: RaggedTensor,
                mask: Tensor, 
                positional_embedding: Tensor, 
                multi_kv_cache: list = None,
                cross_kv_cache: list = None,
                use_cache: bool = False):
        
        if use_cache:
            presents = []
        
        # offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        embed = self.token_embedding(x.data)
        # self.register_network_output('token', embed)
        # hidden_states = embed + slice(self.positional_embedding.value, starts=[offset, 0], sizes=[x.data.shape[-1], self.n_state])
        hidden_states = embed + positional_embedding
        # self.register_network_output('after_add', hidden_states)
        hidden_states = RaggedTensor.from_row_lengths(
            hidden_states,
            x.row_lengths,
            x.max_row_length
        )
        
        # self.register_network_output('before_block', hidden_states.data)
        
        for i, block in enumerate(self.blocks):
            kv_cache = multi_kv_cache[i]
            c_kv_cache = cross_kv_cache[i]
            hidden_states = block(hidden_states,
                                   mask=mask, 
                                   multi_kv_cache=kv_cache,
                                   cross_kv_cache=c_kv_cache,
                                   use_cache=use_cache)
            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        x = hidden_states.data

        x = self.ln(x)
        # self.register_network_output("x_ln", x)
        x = matmul(x, self.token_embedding_weight.value, transb=True)

        x.mark_output('output', str_dtype_to_trt("float16"))
        
        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', str_dtype_to_trt("float16"))
            return (x, presents)
        
        return x
    
    def prepare_inputs(self, max_batch_size, max_input_len, max_new_tokens,
                       use_cache, max_beam_width):
        
        max_output_len = max_new_tokens
        
        max_len = max_input_len + max_output_len
        
        bb_range = [1, (max_batch_size * max_beam_width + 1) // 2, max_batch_size * max_beam_width]
        mask_len_range = [1, 1, max_len + 1]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2 + 1, max_len + 1]
        
        x = Tensor(
            name='x',
            dtype=trt.int32,
            shape=[1, -1],
            is_network_input=True,
            dim_range=OrderedDict([
                ('batch_size', [1]),
                ('input_len', [inlen_range])
            ])
        )

        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[1],
                               dim_range=OrderedDict([
                                   ('batch_size', [1])
                               ]))

        max_input_length = Tensor(name='max_input_length',
                                  dtype=trt.int32,
                                  shape=[-1],
                                  dim_range=OrderedDict([
                                      ('max_input_len', [inlen_range])
                                  ]))

        x_ragged = RaggedTensor.from_row_lengths(
            x,
            input_lengths,
            max_input_length
        )

        mask = Tensor(
            name='mask',
            dtype=trt.float32,
            shape=[-1, -1],
            is_network_input=True,
            dim_range=OrderedDict([
                ('batch_size', [bb_range]),
                ('mask_len', [mask_len_range]),
            ]))
        
        positional_embedding = Tensor(
            name='positional_embedding',
            dtype=trt.float16,
            shape=[-1, 1280],
            is_network_input=True,
            dim_range=OrderedDict([
                ('input_len', [inlen_range]),
                ('hidden_states', [1280])
            ])
        )
        
        past_key_value = []
        kv_dim_range = OrderedDict([
                    ('batch_size', [1]),
                    ('kv', [2]),
                    ('num_heads', [20]),
                    ('past_key_len', [max_len_range]),
                    ('head_size', [64]),
                ])
        
        for i in range(self.n_layer):
            kv = Tensor(name=f'past_key_value_'+str(i),
                    dtype=trt.float16,
                    shape=[1, 2, 20, -1, 64],
                    is_network_input=True,
                    dim_range=kv_dim_range)
            past_key_value.append(kv)
        
        cross_past_key_value = []
        cross_kv_dim_range = OrderedDict([
                    ('batch_size', [1]),
                    ('kv', [2]),
                    ('num_heads', [20]),
                    ('past_key_len', [1500]),
                    ('head_size', [64]),
                ])
        
        for i in range(self.n_layer):
            kv = Tensor(name=f'cross_past_key_value_'+str(i),
                    dtype=trt.float16,
                    shape=[1, 2, 20, 1500, 64],
                    is_network_input=True,
                    dim_range=cross_kv_dim_range)
            cross_past_key_value.append(kv)
        
        return (x_ragged, mask, positional_embedding, past_key_value, cross_past_key_value, True)

class KVLinearBlock(Module):
    def __init__(self,
                 n_state: int,
                 n_head,
                 dtype
                 ):
        super().__init__()
        
        self.k_linear = Linear(n_state,
                                n_state,
                                bias=False,
                                dtype=dtype,
                                gather_output=False)

        self.v_linear = Linear(n_state,
                                n_state,
                                bias=True,
                                dtype=dtype,
                                gather_output=False)
        self.n_head = n_head
        self.n_state = n_state
        self.n_head_size = n_state // n_head
        
    def forward(self,
                xa: Tensor,
                ):
        key = self.k_linear(xa)
        value = self.v_linear(xa)

        def transpose_for_scores(x, is_kv: bool = False):
            _num_attention_heads = self.n_head
            new_x_shape = concat([
                shape(x, 0),
                shape(x, 1), _num_attention_heads, self.n_head_size
            ])
            return x.view(new_x_shape).permute([0, 2, 1, 3])

        key = transpose_for_scores(key, is_kv=True)
        value = transpose_for_scores(value, is_kv=True)
        
        key_inflated_shape = concat([
            shape(key, 0), 1,
            shape(key, 1),
            shape(key, 2),
            shape(key, 3)
        ])
        inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
        inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
        cross_past_key_value = concat([inflated_key, inflated_value], dim=1)

        # if self.use_int8_kv_cache:

        #     def quantize_tensor(x, scale):
        #         scaled = x * scale
        #         rounded = round(scaled)
        #         clipped = clip(rounded, -128, 127)
        #         quantized = cast(clipped, 'int8')
        #         return quantized

        #     cross_past_key_value = quantize_tensor(
        #         cross_past_key_value, self.kv_orig_quant_scale.value)

        return cross_past_key_value

class CrossAttn_KV(Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int, dtype):
        super().__init__()
        self.blocks = ModuleList([
            KVLinearBlock(n_state, n_head, dtype) for _ in range(n_layer)
        ])
        
    def forward(self, 
                xa: Tensor):
        presents = []
        for i, block in enumerate(self.blocks):
            cross_past_key_value = block(xa)
            presents.append(cross_past_key_value)
            
        for i, present in enumerate(presents):
            present.mark_output(f'cross_present_key_value_{i}', str_dtype_to_trt("float16"))
        return presents
    
    def prepare_inputs(self,):
        xa = Tensor(
            name='xa',
            dtype=trt.float16,
            shape=[1, 1500, 1280],
            is_network_input=True,
            dim_range=OrderedDict([
                ('batch_size', [1]),
                ('position_len', [1500]),
                ('hidden_states', [1280])
            ])
        )
        
        return (xa)
    
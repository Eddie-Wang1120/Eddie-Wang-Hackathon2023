import torch
from torch.nn import Conv1d, Conv2d, LayerNorm
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Dict, Iterable, Optional
import numpy as np
model = torch.load("large-v2.pt")
print(model['dims'])
# print(model['model_state_dict']['encoder.blocks.0.attn.query.weight'].shape)
# print(model['model_state_dict']['encoder.conv1.weight'].shape)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


temp = torch.ones([1, 80, 3000]).type(torch.float16).to("cuda")
conv1_weight = model['model_state_dict']['encoder.conv1.weight'].to("cuda")
conv1_bias = model['model_state_dict']['encoder.conv1.bias'].to("cuda")
conv2_weight = model['model_state_dict']['encoder.conv2.weight'].to("cuda")
conv2_bias = model['model_state_dict']['encoder.conv2.bias'].to("cuda")

conv1 = Conv1d(80, 1280, kernel_size=3, padding=1).to("cuda")
conv1.half()
conv2 = Conv1d(1280, 1280, kernel_size=3, stride=2, padding=1).to("cuda")
conv2.half()
conv1.weight.data = conv1_weight
conv1.bias.data = conv1_bias
conv2.weight.data = conv2_weight
conv2.bias.data = conv2_bias

model_params = model['model_state_dict']
n_state = model['dims']['n_audio_state']
n_head = model['dims']['n_audio_head']
n_mlp = n_state*4

position = sinusoids(model['dims']['n_audio_ctx'], n_state).to("cuda")

x = conv1(temp)
x = F.gelu(x)
x = conv2(x)
x = F.gelu(x)
print(x.shape)
x = torch.transpose(x, 2, 1)
x = x + position
print(x.shape)

attn_ln = LayerNorm(n_state).to("cuda")
attention = MultiHeadAttention(n_state, n_head).to("cuda")
mlp_ln = LayerNorm(n_state).to("cuda")
mlp1 = Linear(n_state, n_mlp).to("cuda")
mlp2 = Linear(n_mlp, n_state).to("cuda")

attn_ln.weight.data = model_params['encoder.blocks.0.attn_ln.weight'].to("cuda").float()
attn_ln.bias.data = model_params['encoder.blocks.0.attn_ln.bias'].to("cuda").float()
    
attention.query.weight.value =  model_params['encoder.blocks.0.attn.query.weight']
attention.query.bias.value =  model_params['encoder.blocks.0.attn.query.bias']
attention.key.weight.value =  model_params['encoder.blocks.0.attn.key.weight']
attention.value.weight.value =  model_params['encoder.blocks.0.attn.value.weight']
attention.value.bias.value =  model_params['encoder.blocks.0.attn.value.bias']
    
attention.out.weight.value = model_params['encoder.blocks.0.attn.out.weight'].numpy()
attention.out.bias.value = model_params['encoder.blocks.0.attn.out.bias'].numpy()
    
mlp_ln.weight.value = model_params['encoder.blocks.0.mlp_ln.weight'].numpy()
mlp_ln.bias.value = model_params['encoder.blocks.0.mlp_ln.bias'].numpy()
    
mlp1.weight.value = model_params['encoder.blocks.0.mlp.0.weight'].numpy()
mlp1.bias.value = model_params['encoder.blocks.0.mlp.0.bias'].numpy()
    
mlp2.weight.value = model_params['encoder.blocks.0.mlp.2.weight'].numpy()
mlp2.bias.value = model_params['encoder.blocks.0.mlp.2.bias'].numpy()

residual1 = x
x = attn_ln(x)
x = attention(x)[0]
# x = residual1 + attention(x)[0]
# residual = x
# x = mlp_ln(x)
# x = mlp1(x)
# x = F.gelu(x)
# x = mlp2(x)
# x = residual + x
print(x)

# temp2 = conv1(temp)
# print(temp2.shape)

# temp_1d = torch.unsqueeze(temp, -1)
# weight_1d = torch.unsqueeze(weight, -1)

# # print(temp_1d.shape)
# # print(weight_1d.shape)
# # print(bias.shape)

# conv2 = Conv2d(80, 1280, kernel_size=(3, 1), padding=(1, 0)).to("cuda")
# conv2.half()
# conv2.weight.data = weight_1d
# conv2.bias.data = bias
# temp2_1d = conv2(temp_1d)
# temp2_1d = torch.squeeze(temp2_1d, dim=-1)

# print(temp2_1d)

# # print(temp2.equal(temp2_1d))
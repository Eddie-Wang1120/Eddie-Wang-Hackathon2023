from ..functional import softplus, tanh, gelu
from ..module import Module


class Mish(Module):

    def forward(self, input):
        return input * tanh(softplus(input, beta=1.0, threshold=20.0))
    
class Gelu(Module):
    def forward(self, input):
        return gelu(input)

import torch
from torch import nn
from einops import einsum, reduce
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.empty(self.d_model))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1
        nn.init.trunc_normal_(self.weights, mean = 0, std = std, a = -3 * std, b = 3 * std)
    
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_in = x
        rms = reduce(x_in ** 2, "... d_model -> ... 1", "sum") / self.d_model
        x = x_in / torch.sqrt(rms + self.eps)

        result = einsum(x, self.weights, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)
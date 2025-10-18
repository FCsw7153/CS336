import torch
from torch import nn
from einops import rearrange, einsum
import einx
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


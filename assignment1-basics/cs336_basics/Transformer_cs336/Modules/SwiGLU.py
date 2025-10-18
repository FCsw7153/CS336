import torch
from torch import nn
from einops import einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.w2_weight = nn.Parameter(torch.empty(self.d_model, self.d_ff))
        self.w3_weight = nn.Parameter(torch.empty(self.d_ff, self.d_model))

    def forward(self, x: torch.Tensor):
        W1 = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3 = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = W1 * torch.sigmoid(W1)
        ss = silu * W3
        swiglu = einsum(self.w2_weight, ss, "d_model d_ff, ... d_ff -> ... d_model")
        return swiglu
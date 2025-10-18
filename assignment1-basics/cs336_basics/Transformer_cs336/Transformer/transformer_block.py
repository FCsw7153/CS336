import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch import nn
from einops import rearrange, repeat, einsum, reduce
import math
from jaxtyping import Float, Int
from cs336_basics.Transformer_cs336.Modules import Linear
from cs336_basics.Transformer_cs336.Modules import RoPE
from cs336_basics.Transformer_cs336.Modules import RMSNorm, SwiGLU
from cs336_basics.Transformer_cs336.Attentions import MultiheadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.RMSNorm1 = RMSNorm(self.d_model)
        self.RMSNorm2 = RMSNorm(self.d_model)

        self.SwiGLU_FFN = SwiGLU(self.d_model, self.d_ff)

        self.MultiheadSelfAttention = MultiheadSelfAttention(self.d_model, self.num_heads, self.max_seq_len, self.theta)

    
    def forward(self, x, token_positions):
        x_r = self.RMSNorm1(x)
        x_multi = self.MultiheadSelfAttention(x_r, token_positions)
        y = x + x_multi

        output = self.RMSNorm2(y)
        output = self.SwiGLU_FFN(output) + y

        return output
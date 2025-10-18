import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math
from jaxtyping import Float, Int
from cs336_basics.Transformer_cs336.Modules import Linear
from cs336_basics.Transformer_cs336.Modules import RoPE
from .attention_fn import *

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int = None, theta:float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = self.d_model // self.num_heads

        self.q_weight = Linear(self.d_model, self.d_k * self.num_heads)
        self.k_weight = Linear(self.d_model, self.d_k * self.num_heads)
        self.v_weight = Linear(self.d_model, self.d_k * self.num_heads)
        self.o_weight = Linear(self.d_k * self.num_heads, self.d_model)

    def forward(self, x, token_positions: Tensor = None):
        B, S, d_model = x.shape
        Q_x = self.q_weight(x)
        K_x = self.k_weight(x)
        V_x = self.v_weight(x)

        Q_x = Q_x.reshape(B, S, self.num_heads, self.d_k)
        K_x = K_x.reshape(B, S, self.num_heads, self.d_k)
        V_x = V_x.reshape(B, S, self.num_heads, self.d_k)
        Q_x = Q_x.transpose(-3, -2)
        K_x = K_x.transpose(-3, -2)
        V_x = V_x.transpose(-3, -2)

        mask = torch.tril(torch.ones(S, S))

        if token_positions is not None:
            rope = RoPE(self.theta, self.d_k, self.max_seq_len)
            Q_x = rope(Q_x, token_positions)
            K_x = rope(K_x, token_positions)

        output = scaled_dot_product_attention(Q_x, K_x, V_x, mask)

        B, H, S, D = output.shape
        output = output.transpose(-3, -2)
        output = output.reshape(B, S, d_model)

        return self.o_weight(output)
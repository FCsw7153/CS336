import torch
from torch import nn
from einops import einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.sin, self.cos = self.create_positions()

    def create_positions(self):
        inv_freq = 1 / (self.theta ** (torch.arange(0, self.d_k, 2) / self.d_k))
        positions = torch.arange(0, self.max_seq_len)
        angles = positions[:, None] * inv_freq[None, :]
        sin = angles.sin()
        cos = angles.cos()
        return sin, cos

    def forward(self, x, y):
        s_sin = self.sin[y]
        s_cos = self.cos[y]
        
        new_sin = torch.repeat_interleave(s_sin, repeats=2, dim=-1)
        new_cos = torch.repeat_interleave(s_cos, repeats=2, dim=-1)

        prefix_shape = x.shape[:-1] # b, h, s
        new_shape = prefix_shape + (-1, 2) # b, h, s, d_k/2, 2
        x_pairs = x.reshape(new_shape)
        
        col1, col2 = x_pairs[..., 0], x_pairs[..., 1]
        rotated_pairs = torch.stack([-col2, col1], dim=-1)
        
        prefix_shape = rotated_pairs.shape[:-2]
        x_rotated = rotated_pairs.reshape(*prefix_shape, -1)
        
        result = x * new_cos + x_rotated * new_sin
        return result

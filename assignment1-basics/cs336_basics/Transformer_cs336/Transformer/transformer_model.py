import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch import nn
from einops import rearrange, repeat, einsum, reduce
import math
from jaxtyping import Float, Int
from cs336_basics.Transformer_cs336.Modules import Embedding
from cs336_basics.Transformer_cs336.Modules import Linear
from cs336_basics.Transformer_cs336.Modules import RoPE
from cs336_basics.Transformer_cs336.Modules import RMSNorm, SwiGLU
from cs336_basics.Transformer_cs336.Attentions import MultiheadSelfAttention
from .transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.Embedding = Embedding(self.vocab_size, self.d_model)

        self.TransformerList = nn.ModuleList(
            TransformerBlock(
                self.d_model,
                self.num_heads,
                self.d_ff,
                self.context_length,
                self.rope_theta
            ) for _ in range(self.num_layers)
        )

        self.find_norm = RMSNorm(self.d_model)

        self.output_FFN = Linear(self.d_model, self.vocab_size)
    
    def forward(self, x):
        b, s = x.shape
        x = self.Embedding(x)

        pos_ids = torch.arange(0, s)
        pos_ids = repeat(pos_ids, "s->b s", b = b)

        for layer in self.TransformerList:
            x = layer(x, pos_ids)
        
        x = self.find_norm(x)
        x = self.output_FFN(x)

        return x
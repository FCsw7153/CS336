import torch
from torch import nn
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Int
import math

def softmax(
    x: torch.Tensor,
    i: int
):
    x_max = torch.max(x, dim=i, keepdim=True).values
    x_exp = torch.exp(x-x_max)
    x_sum = torch.sum(x_exp, dim=i, keepdim=True)

    return x_exp/x_sum

def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_v"],
    mask: Float[Tensor, "... seq_len seq_len"] | None=None
) -> Float[Tensor, "... seq_len d_v"]:
    d_k = Q.shape[-1]
    QK = torch.matmul(Q, K.transpose(-2, -1))
    scores = QK / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.inf)
    
    scores = softmax(scores, i = -1)

    output = torch.matmul(scores, V)

    return output
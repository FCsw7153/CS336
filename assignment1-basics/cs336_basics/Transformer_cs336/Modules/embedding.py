import torch
from torch import nn
import math
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1
        nn.init.trunc_normal_(self.weights, mean = 0.0, std = 1, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor):
        return self.weights[token_ids]
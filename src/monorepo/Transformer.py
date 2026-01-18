"""
Author: e-zorzi
License: Apache 2.0
"""

import torch
import torch.nn as nn
from math import sqrt

"""TODO
    - Make heads QKV fused
    - Handle batch dimension
"""

_N_FEATURES = 16
_HIDDEN_SIZE = 512
_KQ_EMBEDDING_SIZE = 744


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int = 3,
        in_features: int = _N_FEATURES,
        kq_embedding_size: int = _KQ_EMBEDDING_SIZE,
    ):
        super().__init__()

        if in_features % n_heads != 0:
            raise ValueError(
                f"The number of heads must divide exactly the number of features, "
                f"but you have {in_features} features and {n_heads} heads"
            )

        self.n_heads = n_heads
        self.in_features = in_features
        self.kq_embedding_size = kq_embedding_size
        self.head_dim = in_features // n_heads

        self.Q_proj = nn.ModuleList([
            nn.Linear(in_features, kq_embedding_size) for _ in range(n_heads)
        ])
        self.K_proj = nn.ModuleList([
            nn.Linear(in_features, kq_embedding_size) for _ in range(n_heads)
        ])
        self.V_proj = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(n_heads)
        ])

        self.attn_projection = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        results = torch.zeros_like(x)
        attentions = []

        for i in range(self.n_heads):
            Q = self.Q_proj[i](x)
            K = self.K_proj[i](x)
            V = self.V_proj[i](x)

            logits = (Q @ K.T) / sqrt(self.kq_embedding_size)
            attn = torch.softmax(logits, dim=-1)

            start = i * self.head_dim
            end = (i + 1) * self.head_dim
            results[:, start:end] = attn @ V

            attentions.append(attn)

        out = self.attn_projection(results)

        if return_weights:
            return out, attentions
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 3,
        in_features: int = _N_FEATURES,
        hidden_size: int = _HIDDEN_SIZE,
        n_layers: int = 3,
        kq_embedding_size: int = 32,
    ):
        super().__init__()

        if n_layers < 3:
            raise ValueError("Must set n_layers to at least 3")

        self.multiheadattention = MultiHeadAttention(
            n_heads=n_heads,
            in_features=in_features,
            kq_embedding_size=kq_embedding_size,
        )

        self.norm1 = nn.RMSNorm(in_features)
        self.norm2 = nn.RMSNorm(in_features)

        layers = [
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
        ]
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            ])
        layers.extend([
            nn.Linear(hidden_size, in_features),
            nn.SiLU(),
        ])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.forward_pre_norm(x)

    def forward_pre_norm(self, x: torch.Tensor):
        # Pre-norm
        y = self.multiheadattention(self.norm1(x))
        z = x + y
        projected = self.mlp(self.norm2(z))
        return projected + z

    def forward_post_norm(self, x: torch.tensor):
        # Post-norm
        y = self.multiheadattention(x)
        z = self.norm1(x + y)
        projected = self.mlp(z)
        return self.norm2(projected + z)


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        n_heads: int = 3,
        in_features: int = _N_FEATURES,
        out_features: int = 2,
        hidden_size_per_block: int = _HIDDEN_SIZE,
        n_layers_per_block: int = 3,
        kq_embedding_size: int = 32,
        n_blocks: int = 8,
        return_logits: bool = True,
    ):
        super().__init__()

        self.return_logits = return_logits

        self.stack = nn.Sequential(*[
            TransformerBlock(
                n_heads=n_heads,
                in_features=in_features,
                hidden_size=hidden_size_per_block,
                n_layers=n_layers_per_block,
                kq_embedding_size=kq_embedding_size,
            )
            for _ in range(n_blocks)
        ])

        self.norm = nn.RMSNorm(in_features)

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, out_features),
        )

        if not return_logits:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = self.head(self.norm(self.stack(x)))
        if not self.return_logits:
            return self.softmax(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        in_features: int = _N_FEATURES,
        out_features: int = _N_FEATURES,
    ):
        super().__init__()
        self.embedder = nn.Embedding(2**16, out_features)

    def forward(self, x: torch.Tensor):
        return self.embedder(x)

"""
Author: e-zorzi
License: Apache 2.0
"""

"""TODO
- QKV in single linear layer
- Correct multihead implementation
- Full transformer block
"""
import torch
from math import sqrt
from attrs import define, field

_N_FEATURES = 16
_HIDDEN_SIZE = 512


@define(auto_attribs=True, kw_only=True)
class MultiHeadAttention(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    _kq_embedding_size: int = field(default=32)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        # TODO add correct multihead logic
        self.Q_proj = torch.nn.Linear(self.in_features, self._kq_embedding_size)
        self.K_proj = torch.nn.Linear(self.in_features, self._kq_embedding_size)
        self.V_proj = torch.nn.Linear(self.in_features, self.in_features)

    def __call__(self, x: torch.tensor, return_weights=False):
        # TODO make it work with arbitrary batch size
        # assert x.shape[0] == _n_features
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        logits = torch.matmul(Q, K.T) / sqrt(self.in_features)

        attention_weights = torch.softmax(logits, axis=-1)
        if return_weights:
            return (torch.matmul(attention_weights, V), attention_weights)
        else:
            return torch.matmul(attention_weights, V)


@define(auto_attribs=True, kw_only=True)
class TransformerBlock(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    hidden_size: int = field(default=_HIDDEN_SIZE)
    n_layers: int = field(default=_HIDDEN_SIZE)
    _kq_embedding_size: int = field(default=32)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.multiheadattention = MultiHeadAttention(
            n_heads=self.n_heads,
            _kq_embedding_size=self._kq_embedding_size,
            n_features=self.in_features,
        )
        self.norm1 = torch.nn.RMSNorm(self.in_features)
        self.norm2 = torch.nn.RMSNorm(self.in_features)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_size, self.in_features),
        )

    def __call__(self, x: torch.tensor):
        y = self.multiheadattention(x)
        z = self.norm1(x + y)
        projected = self.mlp(z)
        return self.norm2(projected + z)


@define(auto_attribs=True, kw_only=True)
class Transformer(torch.nn.Module):
    n_heads: int = field(default=3)
    n_blocks: int = field(default=8)
    in_features: int = field(default=_N_FEATURES)
    n_layers_block: int = field(default=2)
    hidden_size_block: int = field(default=_HIDDEN_SIZE)
    _kq_embedding_size: int = field(default=32)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        # TODO implement full Transformer using the blocks and arbitrary MLP
        raise NotImplementedError("Must be implemented.")

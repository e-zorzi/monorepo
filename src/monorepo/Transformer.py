"""
Author: e-zorzi
License: Apache 2.0
"""

import torch
from math import sqrt
from attrs import define, field


"""TODO
- QKV in single linear layer
- Correct multihead implementation
"""

_N_FEATURES = 16
_HIDDEN_SIZE = 512
_KQ_EMBEDDING_SIZE = 744


@define(auto_attribs=True, kw_only=True, eq=False)
class MultiHeadAttention(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    _kq_embedding_size: int = field(default=_KQ_EMBEDDING_SIZE)

    def __attrs_pre_init__(self):
        # Necessary to init the class (being a subclass of Torch.nn.Module)
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

    def parameters(self, recurse=True):
        return super().parameters(recurse)


@define(auto_attribs=True, kw_only=True, eq=False)
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
            kq_embedding_size=self._kq_embedding_size,
            in_features=self.in_features,
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

    def parameters(self, recurse=True):
        return super().parameters(recurse)


@define(auto_attribs=True, kw_only=True, eq=False)
class EncoderTransformer(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    out_features: int = field(default=2)
    hidden_size_per_block: int = field(default=_HIDDEN_SIZE)
    n_layers_per_block: int = field(default=_HIDDEN_SIZE)
    _kq_embedding_size: int = field(default=32)
    n_blocks: int = field(default=8)
    return_logits: bool = field(default=True)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.stack = torch.nn.Sequential(*[
            TransformerBlock(
                n_heads=self.n_heads,
                in_features=self.in_features,
                hidden_size=self.hidden_size_per_block,
                n_layers=self.n_layers_per_block,
                kq_embedding_size=self._kq_embedding_size,
            )
            for _ in range(self.n_blocks)
        ])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.in_features, out_features=256),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=256, out_features=self.out_features),
        )
        self.norm = torch.nn.RMSNorm(self.in_features)
        if not self.return_logits:
            self.softmax = torch.nn.Softmax(dim=0)

    def __call__(self, x: torch.Tensor):
        x = self.head(self.norm(self.stack(x)))
        if not self.return_logits:
            return self.softmax(x)
        return x


@define(auto_attribs=True, kw_only=True, eq=False)
class Embedder(torch.nn.Module):
    in_features: int = field(default=_N_FEATURES)
    out_features: int = field(default=_N_FEATURES)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.embedder = torch.nn.Embedding(2**16, embedding_dim=self.out_features)

    def __call__(self, x: torch.Tensor):
        return self.embedder(x)

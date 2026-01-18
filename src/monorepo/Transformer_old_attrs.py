"""
Author: e-zorzi
License: Apache 2.0
"""

import torch
from math import sqrt
from attrs import define, field


"""TODO
    - Make heads QKV fused
    - Handle batch dimension
"""

_N_FEATURES = 16
_HIDDEN_SIZE = 512
_KQ_EMBEDDING_SIZE = 744


@define(auto_attribs=True, kw_only=True, eq=False)
class MultiHeadAttention(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    _kq_embedding_size: int = field(default=_KQ_EMBEDDING_SIZE)

    @n_heads.validator
    def _n_heads_validator(self, attr, val):
        if (self.in_features / val) % 1 != 0:
            raise ValueError(
                f"The number of heads must divide exactly the number of features, but you have {self.in_features} features and {self.n_heads} heads"
            )

    def __attrs_pre_init__(self):
        # Necessary to init the class (being a subclass of Torch.nn.Module)
        super().__init__()

    def __attrs_post_init__(self):
        self.add_module(
            "attn_projection", torch.nn.Linear(self.in_features, self.in_features)
        )
        for i in range(self.n_heads):
            self.add_module(
                f"Q_proj_{i}",
                torch.nn.Linear(self.in_features, self._kq_embedding_size),
            )
            self.add_module(
                f"K_proj_{i}",
                torch.nn.Linear(self.in_features, self._kq_embedding_size),
            )
            self.add_module(
                f"V_proj_{i}",
                torch.nn.Linear(self.in_features, int(self.in_features / self.n_heads)),
            )

    def forward(self, x: torch.tensor, return_weights=False):
        for name, p in self.attn_projection.named_parameters():
            assert p.device == x.device, (
                f"attn_projection.{name} on {p.device}, x on {x.device}"
            )

        # Naive implemenation of multihead attn with concatenations
        # TODO improve (although I guess in prod a real net will have the number of heads and this code fixed)

        results = torch.zeros_like(x)
        attentions = []
        for i in range(self.n_heads):
            Q = self._modules[f"Q_proj_{i}"](x)
            K = self._modules[f"K_proj_{i}"](x)
            V = self._modules[f"V_proj_{i}"](x)
            logits = torch.matmul(Q, K.T) / sqrt(self._kq_embedding_size)

            attention_weights = torch.softmax(logits, dim=-1)
            results[:, i * V.shape[1] : ((i + 1) * V.shape[1])] = torch.matmul(
                attention_weights, V
            )
            attentions.append(attention_weights)
        if return_weights:
            return self.attn_projection(results), attentions
        else:
            return self.attn_projection(results)


@define(auto_attribs=True, kw_only=True, eq=False)
class TransformerBlock(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    hidden_size: int = field(default=_HIDDEN_SIZE)
    n_layers: int = field(default=3)
    _kq_embedding_size: int = field(default=32)

    @n_layers.validator
    def _n_layers_validator(self, attr, val):
        if val < 3:
            raise ValueError("Must set n_layers to at least 3")

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
        layers = [
            torch.nn.Linear(self.in_features, self.hidden_size),
            torch.nn.SiLU(),
        ]
        for _ in range(self.n_layers - 2):
            layers.extend([
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.SiLU(),
            ])
        layers.extend([
            torch.nn.Linear(self.hidden_size, self.in_features),
            torch.nn.SiLU(),
        ])
        self.mlp = torch.nn.Sequential(*layers)

    def forward_post_norm_old(self, x: torch.tensor):
        # Post-norm
        y = self.multiheadattention(x)
        z = self.norm1(x + y)
        projected = self.mlp(z)
        return self.norm2(projected + z)

    def forward(self, x: torch.tensor):
        # Pre-norm
        y = self.multiheadattention(self.norm1(x))
        z = x + y
        projected = self.mlp(self.norm2(z))
        return projected + z


@define(auto_attribs=True, kw_only=True, eq=False)
class EncoderTransformer(torch.nn.Module):
    n_heads: int = field(default=3)
    in_features: int = field(default=_N_FEATURES)
    out_features: int = field(default=2)
    hidden_size_per_block: int = field(default=_HIDDEN_SIZE)
    n_layers_per_block: int = field(default=3)
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
            self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        return self.embedder(x)

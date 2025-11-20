import torch
from math import sqrt

# _MAX_CONTEXT = 16
_N_FEATURES = 17


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads=3, kq_embedding_size=4, _n_features=_N_FEATURES):
        # TODO add correct multihead logic
        super().__init__()
        self._n_features = _n_features
        self.Q_proj = torch.nn.Linear(_n_features, kq_embedding_size)
        self.K_proj = torch.nn.Linear(_n_features, kq_embedding_size)
        self.V_proj = torch.nn.Linear(_n_features, _n_features)

    def __call__(self, x: torch.tensor, return_weights=False):
        # TODO make it work with arbitrary batch size
        # assert x.shape[0] == _n_features
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        logits = torch.matmul(Q, K.T) / sqrt(self._n_features)

        attention_weights = torch.softmax(logits, axis=-1)
        if return_weights:
            return (torch.matmul(attention_weights, V), attention_weights)
        else:
            return torch.matmul(attention_weights, V)


class TransformerBlock(torch.nn.Module):
    def __init__(self, n_heads=3, kq_embedding_size=4, hidden_size=1024):
        super().__init__()
        self.multiheadattention = MultiHeadAttention(
            n_heads=n_heads, kq_embedding_size=kq_embedding_size
        )
        self.norm1 = torch.nn.RMSNorm(_N_FEATURES)
        self.norm2 = torch.nn.RMSNorm(_N_FEATURES)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(_N_FEATURES, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, _N_FEATURES),
        )

    def __call__(self, x: torch.tensor):
        y = self.multiheadattention(x)
        z = self.norm1(x + y)
        projected = self.mlp(z)
        return self.norm2(projected + z)


class Transformer(torch.nn.Module):
    def __init__(self):
        # TODO implement full Transformer using the blocks and arbitrary MLP
        raise NotImplementedError("Must be implemented.")

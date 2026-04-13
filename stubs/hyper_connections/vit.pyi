from _typeshed import Incomplete
from hyper_connections.mHCv2 import mHC as mHC
from torch.nn import Module

def pair(t): ...

class FeedForward(Module):
    net: Incomplete
    def __init__(self, dim, hidden_dim, dropout: float = 0.0) -> None: ...
    def forward(self, x): ...

class Attention(Module):
    heads: Incomplete
    scale: Incomplete
    norm: Incomplete
    attend: Incomplete
    dropout: Incomplete
    to_qkv: Incomplete
    to_out: Incomplete
    def __init__(self, dim, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None: ...
    def forward(self, x): ...

class Transformer(Module):
    norm: Incomplete
    layers: Incomplete
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout: float = 0.0, num_residual_streams: int = 4, mhc_kwargs=...) -> None: ...
    def forward(self, x): ...

class ViT(Module):
    to_patch_embedding: Incomplete
    cls_token: Incomplete
    pos_embedding: Incomplete
    dropout: Incomplete
    transformer: Incomplete
    pool: Incomplete
    to_latent: Incomplete
    mlp_head: Incomplete
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool: str = 'cls', channels: int = 3, dim_head: int = 64, dropout: float = 0.0, emb_dropout: float = 0.0, num_residual_streams: int = 4, mhc_kwargs=...) -> None: ...
    def forward(self, img): ...

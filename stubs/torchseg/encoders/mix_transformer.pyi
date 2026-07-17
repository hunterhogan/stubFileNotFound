import torch
import torch.nn as nn
from _typeshed import Incomplete

class Mlp(nn.Module):
    fc1: Incomplete
    dwconv: Incomplete
    act: Incomplete
    fc2: Incomplete
    drop: Incomplete
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=..., drop: float = 0.0) -> None: ...
    def forward(self, x, H, W): ...

class Attention(nn.Module):
    dim: Incomplete
    num_heads: Incomplete
    scale: Incomplete
    q: Incomplete
    kv: Incomplete
    attn_drop: Incomplete
    proj: Incomplete
    proj_drop: Incomplete
    sr_ratio: Incomplete
    sr: Incomplete
    norm: Incomplete
    def __init__(self, dim, num_heads: int = 8, qkv_bias: bool = False, qk_scale=None, attn_drop: float = 0.0, proj_drop: float = 0.0, sr_ratio: int = 1) -> None: ...
    def forward(self, x, H, W): ...

class Block(nn.Module):
    norm1: Incomplete
    attn: Incomplete
    drop_path: Incomplete
    norm2: Incomplete
    mlp: Incomplete
    def __init__(self, dim, num_heads, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_scale=None, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, act_layer=..., norm_layer=..., sr_ratio: int = 1) -> None: ...
    def forward(self, x, H, W): ...

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    img_size: Incomplete
    patch_size: Incomplete
    num_patches: Incomplete
    proj: Incomplete
    norm: Incomplete
    def __init__(self, img_size: int = 224, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768) -> None: ...
    def forward(self, x): ...

class MixVisionTransformer(nn.Module):
    num_classes: Incomplete
    depths: Incomplete
    patch_embed1: Incomplete
    patch_embed2: Incomplete
    patch_embed3: Incomplete
    patch_embed4: Incomplete
    block1: Incomplete
    norm1: Incomplete
    block2: Incomplete
    norm2: Incomplete
    block3: Incomplete
    norm3: Incomplete
    block4: Incomplete
    norm4: Incomplete
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, num_classes: int = 1000, embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias: bool = False, qk_scale=None, drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0, norm_layer=..., depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)) -> None: ...
    def init_weights(self, pretrained=None) -> None: ...
    def reset_drop_path(self, drop_path_rate) -> None: ...
    def freeze_patch_emb(self) -> None: ...
    @torch.jit.ignore
    def no_weight_decay(self): ...
    def get_classifier(self): ...
    head: Incomplete
    def reset_classifier(self, num_classes, global_pool: str = '') -> None: ...
    def forward_features(self, x): ...
    def forward(self, x): ...

class DWConv(nn.Module):
    dwconv: Incomplete
    def __init__(self, dim: int = 768) -> None: ...
    def forward(self, x, H, W): ...

class MixVisionTransformerEncoder(MixVisionTransformer):
    output_stride: int
    depth: Incomplete
    in_channels: int
    out_channels: Incomplete
    def __init__(self, out_channels, depth: int = 5, **kwargs) -> None: ...
    def forward(self, x): ...
    def load_state_dict(self, state_dict): ...

def get_pretrained_cfg(name): ...

mix_transformer_encoders: Incomplete

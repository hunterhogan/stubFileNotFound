from _typeshed import Incomplete
from einops import rearrange as rearrange
from PoPE_pytorch.pope import apply_pope_to_qk as apply_pope_to_qk, PolarEmbedReturn as PolarEmbedReturn
from torch import is_tensor as is_tensor, Tensor as Tensor
from torch.nn import Module

def exists(v): ...
def default(v, d): ...

class AxialPoPE(Module):
    apply_pope_to_qk: Incomplete
    dim: Incomplete
    heads: Incomplete
    axial_dims: Incomplete
    inv_freqs: Incomplete
    bias: Incomplete
    def __init__(self, dim, *, heads, axial_dims: tuple[int, ...] | None = None, theta: int = 10000, bias_uniform_init: bool = False) -> None: ...
    @property
    def device(self): ...
    @staticmethod
    def get_grid_positions(*dims, device=None): ...
    def forward(self, pos_or_dims: Tensor | tuple[int, ...]): ...

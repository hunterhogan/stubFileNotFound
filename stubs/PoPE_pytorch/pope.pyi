from _typeshed import Incomplete
from torch import stack as stack, Tensor as Tensor
from torch.nn import Module
from typing import NamedTuple

class PolarEmbedReturn(NamedTuple):
    freqs: Incomplete
    bias: Incomplete

def exists(v): ...
def default(v, d): ...
def apply_pope_to_qk(pope: PolarEmbedReturn, q, k, to_magnitude=..., return_complex: bool = False): ...

class PoPE(Module):
    apply_pope_to_qk: Incomplete
    bias: Incomplete
    def __init__(self, dim, *, heads, theta: int = 10000, bias_uniform_init: bool = False, inv_freqs: Tensor | list[float] | None = None) -> None: ...
    @property
    def device(self): ...
    def forward(self, pos_or_seq_len: Tensor | int, offset: int = 0): ...

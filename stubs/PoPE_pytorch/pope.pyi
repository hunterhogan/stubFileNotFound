from collections.abc import Callable
from torch import Tensor
from typing import NamedTuple
import torch
from torch.nn import Module
import torch.nn.functional as F

class PolarEmbedReturn(NamedTuple):
	freqs: Tensor
	bias: Tensor

def apply_pope_to_qk(
	pope: tuple[Tensor, Tensor], q: Tensor, k: Tensor, to_magnitude: Callable[..., Tensor] = F.softplus, return_complex: bool = False
) -> tuple[Tensor, Tensor]: ...

class PoPE(Module):
	@staticmethod
	def apply_pope_to_qk(
		pope: tuple[Tensor, Tensor], q: Tensor, k: Tensor, to_magnitude: Callable[..., Tensor] = F.softplus, *, return_complex: bool = False
	) -> tuple[Tensor, Tensor]: ...

	def __init__(
		self, dim: int, *, heads: int, theta: float = 10000, bias_uniform_init: bool = False, inv_freqs: Tensor | list[float] | None = None
	) -> None: ...

	@property
	def device(self) -> torch.device: ...

	def forward(self, pos_or_seq_len: Tensor | int, offset: int = 0) -> PolarEmbedReturn: ...

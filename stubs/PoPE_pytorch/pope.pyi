from collections.abc import Callable
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

class PoPE(Module):
	@staticmethod
	def apply_pope_to_qk(
		pope: tuple[Tensor, Tensor], q: Tensor, k: Tensor, to_magnitude: Callable[..., Tensor] = F.softplus, *, return_complex: bool = False
	) -> tuple[Tensor, Tensor]: ...

	def __init__(
		self, dim: int, *, heads: int, theta: float = 10000, bias_uniform_init: bool = False, inv_freqs: Tensor | None = None
	) -> None: ...

	def forward(self, pos_or_seq_len: Tensor | int, offset: int = 0) -> tuple[Tensor, Tensor]: ...

from collections.abc import Callable
from PoPE_pytorch.pope import PolarEmbedReturn
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.nn import Module
from torch.types import Number
import torch
import torch.nn.functional as F

class AxialPoPE(Module):
	@staticmethod
	def apply_pope_to_qk(pope: tuple[Tensor, Tensor], q: Tensor, k: Tensor, to_magnitude: Callable[..., Tensor] = F.softplus, *, return_complex: bool = False) -> tuple[Tensor, Tensor]: ...
	dim: int
	heads: int
	axial_dims: tuple[int, ...]
	def __init__(
		self, dim: int, *, heads: int, axial_dims: tuple[int, ...] | None = None, theta: float = 10000, bias_uniform_init: bool = False
	) -> None: ...
	@property
	def device(self) -> torch.device: ...
	def forward(self, pos_or_dims: Tensor | tuple[int, ...]) -> PolarEmbedReturn: ...
	@staticmethod
	def get_grid_positions(*dims: Number, device: DeviceLikeType | None = None) -> Tensor: ...

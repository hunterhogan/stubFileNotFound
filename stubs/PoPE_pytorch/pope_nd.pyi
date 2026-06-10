from collections.abc import Callable
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.nn import Module
from torch.types import Number
import torch.nn.functional as F

class AxialPoPE(Module):
	@staticmethod
	def apply_pope_to_qk(pope: tuple[Tensor, Tensor], q: Tensor, k: Tensor, to_magnitude: Callable[..., Tensor] = F.softplus, *, return_complex: bool = False) -> tuple[Tensor, Tensor]: ...
	def forward(self, pos_or_dims: Tensor | tuple[int, ...]) -> tuple[Tensor, Tensor]: ...
	@staticmethod
	def get_grid_positions(*dims: Number, device: DeviceLikeType | None = None) -> Tensor: ...

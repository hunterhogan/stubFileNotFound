from torch import Tensor
from typing import Literal
import torch.nn.modules.module

def apply_rotary_emb(
	freqs: Tensor, t: Tensor, start_index: int = 0, scale: Tensor | float = 1.0, seq_dim: int = -2, freqs_seq_dim: int | None = None
) -> Tensor: ...
def apply_learned_rotations(rotations: Tensor, t: Tensor, start_index: int = 0, freq_ranges: Tensor | None = None) -> Tensor: ...

class RotaryEmbedding(torch.nn.modules.module.Module):
	def __init__(
		self,
		dim: int,
		custom_freqs: Tensor | None = None,
		freqs_for: Literal["lang", "pixel", "constant"] = "lang",
		theta: int | float = 10000,
		max_freq: int | float = 10,
		num_freqs: int = 1,
		*,
		learned_freq: bool = False,
		use_xpos: bool = False,
		xpos_scale_base: int | float = 512,
		interpolate_factor: float = 1.0,
		theta_rescale_factor: float = 1.0,
		seq_before_head_dim: bool = False,
		cache_if_possible: bool = True,
		cache_max_seq_len: int = 8192,
	) -> None: ...
	def forward(self, t: Tensor, seq_len: int | None = None, offset: int = 0) -> Tensor: ...
	def get_seq_pos(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None, offset: int = 0) -> Tensor: ...

	def rotate_queries_or_keys(self, t: Tensor, seq_dim: int | None = None, offset: int = 0, scale: Tensor | float | None = None) -> Tensor: ...

	def rotate_queries_with_cached_keys(self, q: Tensor, k: Tensor, seq_dim: int | None = None, offset: int = 0) -> tuple[Tensor, Tensor]: ...

	def rotate_queries_and_keys(self, q: Tensor, k: Tensor, seq_dim: int | None = None) -> tuple[Tensor, Tensor]: ...

	def get_scale(self, t: Tensor, seq_len: int | None = None, offset: int = 0) -> Tensor: ...

	def get_axial_freqs(self, *dims: int, offsets: (tuple[int | float, ...] | Tensor | None) = None) -> Tensor: ...

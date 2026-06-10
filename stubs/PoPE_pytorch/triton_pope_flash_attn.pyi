from collections.abc import Callable, Sequence
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from triton.runtime.autotuner import Autotuner
from typing import Any
import triton

def cache_by_id(fn: Callable[..., Autotuner]) -> Callable[..., Autotuner]: ...

@cache_by_id
def get_autotuned_kernel(kernel_fn: triton.JITFunction[Any], configs_fn: Callable[[], list[triton.Config]], keys: Sequence[str], blk_d: int, elem_bytes: int, device_idx: int = 0) -> Autotuner: ...

def flash_attn_forward(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	freqs: Tensor | None = None,
	pope_bias: Tensor | None = None,
	mask: Tensor | None = None,
	*,
	causal: bool = False,
	softmax_scale: float | None = None,
	dropout: float = 0.0,
	drop_seed: int = 0,
) -> tuple[Tensor, Tensor]: ...

def flash_attn_backward(
	do: Tensor,
	q: Tensor,
	k: Tensor,
	v: Tensor,
	o: Tensor,
	lse: Tensor,
	dq: Tensor,
	dk: Tensor,
	dv: Tensor,
	dfreqs: Tensor | None = None,
	dpope_bias: Tensor | None = None,
	freqs: Tensor | None = None,
	pope_bias: Tensor | None = None,
	mask: Tensor | None = None,
	softmax_scale: float | None = None,
	dropout: float = 0.0,
	drop_seed: int = 0,
	*,
	causal: bool = False,
) -> None: ...
class FlashAttnFunction(Function):
	@staticmethod
	def forward(
		ctx: FunctionCtx,
		q: Tensor,
		k: Tensor,
		v: Tensor,
		freqs: Tensor | None = None,
		pope_bias: Tensor | None = None,
		mask: Tensor | None = None,
		causal: bool = False,
		softmax_scale: float | None = None,
		dropout: float = 0.0,
	) -> Tensor: ...

	@staticmethod
	def backward(ctx: FunctionCtx, do: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, None, None, None, None]: ...

def flash_attn(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	freqs: Tensor | None = None,
	pope_bias: Tensor | None = None,
	mask: Tensor | None = None,
	causal: bool = False,
	softmax_scale: float | None = None,
	dropout: float = 0.0,
) -> Tensor: ...

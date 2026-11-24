from torch import Tensor

def compute_attn_similarity_non_fused(q: Tensor, k: Tensor, pope: tuple[Tensor, Tensor], pope_pos_emb_indices: Tensor | None = None, *, head_dimension_at_first: bool = True) -> Tensor: ...
def compute_attn_similarity(q: Tensor, k: Tensor, pope: tuple[Tensor, Tensor], pope_pos_emb_indices: Tensor | None = None, *, allow_tf32: bool = True, head_dimension_at_first: bool = True) -> Tensor: ...
def flash_attn_with_pope(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	pos_emb: tuple[Tensor, Tensor],
	pope_pos_emb_indices: Tensor | None = None,
	*,
	mask: Tensor | None = None,
	causal: bool = False,
	softmax_scale: float | None = None,
	fused: bool | None = None,
	head_dimension_at_first: bool = True,
	dropout: float = 0.0,
) -> Tensor: ...

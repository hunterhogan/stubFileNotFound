from .registry import register_rewrite as register_rewrite, Rewrite as Rewrite, rewrite_registry as rewrite_registry
from numba.core.rewrites import (
	ir_print as ir_print, static_binop as static_binop, static_getitem as static_getitem, static_raise as static_raise)

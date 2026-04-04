# ruff: noqa: SLF001
from . import core
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')  # noqa: PYI001

def _check_dtype(dtypes: list[str]) -> Callable[[T], T]: ...
def _add_math_1arg_docstr(name: str) -> Callable[[T], T]: ...
def _add_math_2arg_docstr(name: str) -> Callable[[T], T]: ...
def _add_math_3arg_docstr(name: str) -> Callable[[T], T]: ...

@core.builtin
@_check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x: core.tensor, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential (base 2)")
@core._tensor_member_fn
def exp2(x: core.tensor, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x: core.tensor, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x: core.tensor, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("cosine")
@core._tensor_member_fn
def cos(x: core.tensor, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("sine")
@core._tensor_member_fn
def sin(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr(...)
@core._tensor_member_fn
def sqrt_rn(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _semantic=...) -> core.tensor: ...

@core._tensor_member_fn
@core.builtin
@_add_math_1arg_docstr("absolute value")
def abs(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_add_math_2arg_docstr("fast division")
def fdiv(x, y, ieee_rounding=..., _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_2arg_docstr(...)
def div_rn(x, y, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("floor")
@core._tensor_member_fn
def floor(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("ceil")
@core._tensor_member_fn
def ceil(x, _semantic=...) -> core.tensor: ...

@core.builtin
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _semantic=...) -> core.tensor: ...


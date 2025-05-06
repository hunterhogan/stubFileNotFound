# pyright: reportDeprecated = false
from numpy.typing import NDArray
from typing import Any, Union, TypeVar
import numpy as np

T = TypeVar('T', bound=np.number[Any])

__all__ = ['atomic_add', 'atomic_sub', 'atomic_max', 'atomic_min', 'atomic_xchg']

def atomic_add(ary: NDArray[T], i: int | tuple[int,...], v: T) -> T:
    """
    Atomically, perform `ary[i] += v` and return the previous v of `ary[i]`.

    i must be a simple i for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """

def atomic_sub(ary: NDArray[T], i: int | tuple[int,...], v: T) -> T:
    """
    Atomically, perform `ary[i] -= v` and return the previous v of `ary[i]`.

    i must be a simple i for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """

def atomic_max(ary: NDArray[T], i: int | tuple[int,...], v: T) -> T:
    """
    Atomically, perform `ary[i] = max(ary[i], v)` and return the previous v of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple i for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """

def atomic_min(ary: NDArray[T], i: int | tuple[int,...], v: T) -> T:
    """
    Atomically, perform `ary[i] = min(ary[i], v)` and return the previous v of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple i for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """

def atomic_xchg(ary: NDArray[T], i: int | tuple[int,...], v: T) -> T:
    """
    Atomically, perform `ary[i] = v` and return the previous v of `ary[i]`.

    i must be a simple i for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """

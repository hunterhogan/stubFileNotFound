from typing import Any
from torch import Tensor
import numpy
def from_numpy(ndarray: numpy.ndarray[Any, numpy.dtype[numpy.float64 | numpy.float32 | numpy.float16 | numpy.complex64 | numpy.complex128 | numpy.int64 | numpy.int32 | numpy.int16 | numpy.int8 | numpy.uint8 | numpy.bool_]]) -> Tensor:
    r"""
    from_numpy(ndarray) -> Tensor

    Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

    The returned tensor and :attr:`ndarray` share the same memory. Modifications to
    the tensor will be reflected in the :attr:`ndarray` and vice versa. The returned
    tensor is not resizable.

    It currently accepts :attr:`ndarray` with dtypes of ``numpy.float64``,
    ``numpy.float32``, ``numpy.float16``, ``numpy.complex64``, ``numpy.complex128``,
    ``numpy.int64``, ``numpy.int32``, ``numpy.int16``, ``numpy.int8``, ``numpy.uint8``,
    and ``bool``.

    .. warning::
        Writing to a tensor created from a read-only NumPy array is not supported and will result in undefined behavior.

    Example::

        >>> a = numpy.array([1, 2, 3])
        >>> t = torch.from_numpy(a)
        >>> t
        tensor([ 1,  2,  3])
        >>> t[0] = -1
        >>> a
        array([-1,  2,  3])
    """
    ...

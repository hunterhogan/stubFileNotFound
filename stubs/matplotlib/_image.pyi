import numpy
from typing import ClassVar

BESSEL: _InterpolationType
BICUBIC: _InterpolationType
BILINEAR: _InterpolationType
BLACKMAN: _InterpolationType
CATROM: _InterpolationType
GAUSSIAN: _InterpolationType
HAMMING: _InterpolationType
HANNING: _InterpolationType
HERMITE: _InterpolationType
KAISER: _InterpolationType
LANCZOS: _InterpolationType
MITCHELL: _InterpolationType
NEAREST: _InterpolationType
QUADRIC: _InterpolationType
SINC: _InterpolationType
SPLINE16: _InterpolationType
SPLINE36: _InterpolationType

class _InterpolationType:
    __members__: ClassVar[dict] = ...  # read-only
    BESSEL: ClassVar[_InterpolationType] = ...
    BICUBIC: ClassVar[_InterpolationType] = ...
    BILINEAR: ClassVar[_InterpolationType] = ...
    BLACKMAN: ClassVar[_InterpolationType] = ...
    CATROM: ClassVar[_InterpolationType] = ...
    GAUSSIAN: ClassVar[_InterpolationType] = ...
    HAMMING: ClassVar[_InterpolationType] = ...
    HANNING: ClassVar[_InterpolationType] = ...
    HERMITE: ClassVar[_InterpolationType] = ...
    KAISER: ClassVar[_InterpolationType] = ...
    LANCZOS: ClassVar[_InterpolationType] = ...
    MITCHELL: ClassVar[_InterpolationType] = ...
    NEAREST: ClassVar[_InterpolationType] = ...
    QUADRIC: ClassVar[_InterpolationType] = ...
    SINC: ClassVar[_InterpolationType] = ...
    SPLINE16: ClassVar[_InterpolationType] = ...
    SPLINE36: ClassVar[_InterpolationType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: matplotlib._image._InterpolationType, value: int) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: matplotlib._image._InterpolationType) -> int"""
    def __int__(self) -> int:
        """__int__(self: matplotlib._image._InterpolationType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def resample(input_array: numpy.ndarray, output_array: numpy.ndarray, transform: object, interpolation: _InterpolationType = ..., resample: bool = ..., alpha: float = ..., norm: bool = ..., radius: float = ...) -> None:
    """resample(input_array: numpy.ndarray, output_array: numpy.ndarray, transform: object, interpolation: matplotlib._image._InterpolationType = <_InterpolationType.NEAREST: 0>, resample: bool = False, alpha: float = 1, norm: bool = False, radius: float = 1) -> None

    Resample input_array, blending it in-place into output_array, using an affine transform.

    Parameters
    ----------
    input_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`
        If 2-d, the image is grayscale.  If 3-d, the image must be of size 4 in the last
        dimension and represents RGBA data.

    output_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`
        The dtype and number of dimensions must match `input_array`.

    transform : matplotlib.transforms.Transform instance
        The transformation from the input array to the output array.

    interpolation : int, default: NEAREST
        The interpolation method.  Must be one of the following constants defined in this
        module:

          NEAREST, BILINEAR, BICUBIC, SPLINE16, SPLINE36, HANNING, HAMMING, HERMITE, KAISER,
          QUADRIC, CATROM, GAUSSIAN, BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN

    resample : bool, optional
        When `True`, use a full resampling method.  When `False`, only resample when the
        output image is larger than the input image.

    alpha : float, default: 1
        The transparency level, from 0 (transparent) to 1 (opaque).

    norm : bool, default: False
        Whether to norm the interpolation function.

    radius: float, default: 1
        The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.

    """

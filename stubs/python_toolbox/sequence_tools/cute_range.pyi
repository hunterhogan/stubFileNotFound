from .misc import CuteSequence as CuteSequence
from _typeshed import Incomplete
from python_toolbox import caching as caching
from typing import Any

infinity: Incomplete
infinities: Incomplete
NoneType: Incomplete

def parse_range_args(*args: Any) -> Any: ...
def _is_integral_or_none(thing: Any) -> Any: ...

class CuteRange(CuteSequence):
    """
    Improved version of Python's `range` that has extra features.

    `CuteRange` is like Python's built-in `range`, except (1) it's cute and (2)
    it's completely different. LOL, just kidding.

    `CuteRange` takes `start`, `stop` and `step` arguments just like `range`,
    but it allows you to use floating-point numbers (or decimals), and it
    allows you to use infinite numbers to produce infinite ranges.

    Obviously, `CuteRange` allows iteration, index access, searching for a
    number's index number, checking whether a number is in the range or not,
    and slicing.

    Examples
    --------
        `CuteRange(float('inf'))` is an infinite range starting at zero and
        never ending.

        `CuteRange(7, float('inf'))` is an infinite range starting at 7 and
        never ending. (Like `itertools.count(7)` except it has all the
        amenities of a sequence, you can get items using list notation, you can
        slice it, you can get index numbers of items, etc.)

        `CuteRange(-1.6, 7.3)` is the finite range of numbers `(-1.6, -0.6,
        0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4)`.

        `CuteRange(10.4, -float('inf'), -7.1)` is the infinite range of numbers
        `(10.4, 3.3, -3.8, -10.9, -18.0, -25.1, ... )`.

    """

    def __init__(self, *args: Any) -> None: ...
    _reduced: Incomplete
    __hash__: Incomplete
    __eq__: Incomplete
    distance_to_cover: Incomplete
    @caching.CachedProperty
    def length(self) -> Any:
        """
        The length of the `CuteRange`.

        We're using a property `.length` rather than the built-in `__len__`
        because `__len__` can't handle infinite values or floats.
        """
    __repr__: Incomplete
    @caching.CachedProperty
    def _repr(self) -> Any: ...
    @caching.CachedProperty
    def short_repr(self) -> Any:
        """
        A shorter representation of the `CuteRange`.

        This is different than `repr(cute_range)` only in cases where `step=1`.
        In these cases, while `repr(cute_range)` would be something like
        `CuteRange(7, 20)`, `cute_range.short_repr` would be `7..20`.
        """
    def __getitem__(self, i: Any, allow_out_of_range: bool = False) -> Any: ...
    def __len__(self) -> int: ...
    def index(self, i: Any, start: Any=..., stop: Any=...) -> Any:
        """Get the index number of `i` in this `CuteRange`."""
    is_infinite: Incomplete




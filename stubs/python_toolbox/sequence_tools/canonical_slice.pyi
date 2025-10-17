from _typeshed import Incomplete
from typing import Any

infinity: Incomplete
infinities: Incomplete

class CanonicalSlice:
    """
    A canonical representation of a `slice` with `start`, `stop`, and `step`.

    This is helpful because `slice`'s own `.start`, `.stop` and `.step` are
    sometimes specified as `None` for convenience, so Python will infer them
    automatically. Here we make them explicit. If we're given an iterable (or
    the length of one) in `iterable_or_length`, we'll give a canoncial slice
    for that length, otherwise we'll do a generic one, which is rarely usable
    for actual slicing because it often has `infinity` in it, so it's useful
    only for canonalization. (e.g. checking whether two different slices are
    actually equal.)

    When doing a generic canonical slice (without giving an iterable or
    length):

      - If `start` is `None`, it will be set to `0` (if the `step` is positive)
        or `infinity` (if the `step` is negative.)

      - If `stop` is `None`, it will be set to `infinity` (if the `step` is
        positive) or `0` (if the `step` is negative.)

      - If `step` is `None`, it will be changed to the default `1`.

    """

    given_slice: Incomplete
    length: Incomplete
    offset: Incomplete
    step: int
    start: Incomplete
    stop: Incomplete
    slice_: Incomplete
    def __init__(self, slice_: Any, iterable_or_length: Any=None, offset: int = 0) -> None: ...
    __iter__: Incomplete
    __repr__: Incomplete
    _reduced: Incomplete
    __hash__: Incomplete
    __eq__: Incomplete
    __contains__: Incomplete




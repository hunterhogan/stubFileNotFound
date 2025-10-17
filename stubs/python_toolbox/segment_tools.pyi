
from typing import Any

def crop_segment(segment: Any, base_segment: Any) -> Any:
    """
    Crop `segment` to fit inside `base_segment`.

    This means that if it was partially outside of `base_segment`, that portion
    would be cut off and you'll get only the intersection of `segment` and
    `base_segment`.

    Example:

        >>> crop_segment((7, 17), (10, 20))
        (10, 17)

    """
def merge_segments(segments: Any) -> Any:
    """
    "Clean" a bunch of segments by removing any shared portions.

    This function takes an iterable of segments and returns a cleaned one in
    which any duplicated portions were removed. Some segments which were
    contained in others would be removed completely, while other segments that
    touched each other would be merged.

    Example:

        >>> merge_segments((0, 10), (4, 16), (16, 17), (30, 40))
        ((0, 17), (30, 40))

    """




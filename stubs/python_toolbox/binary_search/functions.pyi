from .roundings import (
	BOTH as BOTH, CLOSEST as CLOSEST, CLOSEST_IF_BOTH as CLOSEST_IF_BOTH, EXACT as EXACT, HIGH as HIGH,
	HIGH_IF_BOTH as HIGH_IF_BOTH, HIGH_OTHERWISE_LOW as HIGH_OTHERWISE_LOW, LOW as LOW, LOW_IF_BOTH as LOW_IF_BOTH,
	LOW_OTHERWISE_HIGH as LOW_OTHERWISE_HIGH, Rounding as Rounding, roundings as roundings)
from typing import Any

def binary_search_by_index(sequence: Any, value: Any, function: Any=..., rounding: Any=...) -> Any:
    """
    Do a binary search, returning answer as index number.

    For all rounding options, a return value of None is returned if no matching
    item is found. (In the case of `rounding=BOTH`, either of the items in the
    tuple may be `None`)

    You may optionally pass a key function as `function`, so instead of the
    objects in `sequence` being compared, their outputs from `function` will be
    compared. If you do pass in a function, it's assumed that it's strictly
    rising.

    Note: This function uses `None` to express its inability to find any
    matches; therefore, you better not use it on sequences in which None is a
    possible item.

    Similiar to `binary_search` (refer to its documentation for more info). The
    difference is that instead of returning a result in terms of sequence
    items, it returns the indexes of these items in the sequence.

    For documentation of rounding options, check `binary_search.roundings`.
    """
def _binary_search_both(sequence: Any, value: Any, function: Any=...) -> Any:
    """
    Do a binary search through a sequence with the `BOTH` rounding.

    You may optionally pass a key function as `function`, so instead of the
    objects in `sequence` being compared, their outputs from `function` will be
    compared. If you do pass in a function, it's assumed that it's strictly
    rising.

    Note: This function uses `None` to express its inability to find any
    matches; therefore, you better not use it on sequences in which `None` is a
    possible item.
    """
def binary_search(sequence: Any, value: Any, function: Any=..., rounding: Any=...) -> Any:
    """
    Do a binary search through a sequence.

    For all rounding options, a return value of None is returned if no matching
    item is found. (In the case of `rounding=BOTH`, either of the items in the
    tuple may be `None`)

    You may optionally pass a key function as `function`, so instead of the
    objects in `sequence` being compared, their outputs from `function` will be
    compared. If you do pass in a function, it's assumed that it's strictly
    rising.

    Note: This function uses `None` to express its inability to find any
    matches; therefore, you better not use it on sequences in which None is a
    possible item.

    For documentation of rounding options, check `binary_search.roundings`.
    """
def make_both_data_into_preferred_rounding(both: Any, value: Any, function: Any=..., rounding: Any=...) -> Any:
    """
    Convert results gotten using `BOTH` to a different rounding option.

    This function takes the return value from `binary_search` (or other such
    functions) with `rounding=BOTH` as the parameter `both`. It then gives the
    data with a different rounding, specified with the parameter `rounding`.
    """




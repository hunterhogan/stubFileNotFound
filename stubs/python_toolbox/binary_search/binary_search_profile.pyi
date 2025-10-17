from .functions import (
	_binary_search_both as _binary_search_both, binary_search as binary_search,
	binary_search_by_index as binary_search_by_index,
	make_both_data_into_preferred_rounding as make_both_data_into_preferred_rounding)
from .roundings import (
	BOTH as BOTH, CLOSEST as CLOSEST, CLOSEST_IF_BOTH as CLOSEST_IF_BOTH, EXACT as EXACT, HIGH as HIGH,
	HIGH_IF_BOTH as HIGH_IF_BOTH, HIGH_OTHERWISE_LOW as HIGH_OTHERWISE_LOW, LOW as LOW, LOW_IF_BOTH as LOW_IF_BOTH,
	LOW_OTHERWISE_HIGH as LOW_OTHERWISE_HIGH, Rounding as Rounding, roundings as roundings)
from _typeshed import Incomplete
from typing import Any

class BinarySearchProfile:
    """
    A profile of binary search results.

    A binary search profile allows to access all kinds of aspects of the
    results of a binary search, while not having to execute the search more
    than one time.
    """

    results: Incomplete
    all_empty: Incomplete
    one_side_empty: Incomplete
    is_surrounded: Incomplete
    had_to_compromise: Incomplete
    got_none_because_no_item_on_other_side: Incomplete
    def __init__(self, sequence: Any, value: Any, function: Any=..., *, both: Any=None) -> None:
        """
        Construct a `BinarySearchProfile`.

        `sequence` is the sequence through which the search is made. `value` is
        the wanted value.

        You may optionally pass a key function as `function`, so instead of the
        objects in `sequence` being compared, their outputs from `function`
        will be compared. If you do pass in a function, it's assumed that it's
        strictly rising.

        In the `both` argument you may put binary search results (with the BOTH
        rounding option.) This will prevent the constructor from performing the
        search itself. It will use the results you provided when giving its
        analysis.
        """




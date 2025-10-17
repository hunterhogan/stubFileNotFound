from _typeshed import Incomplete
from typing import Any

_length_of_recurrent_perm_space_cache: Incomplete

def calculate_length_of_recurrent_perm_space(k: Any, fbb: Any) -> Any:
    """
    Calculate the length of a recurrent `PermSpace`.

    `k` is the `n_elements` of the space, i.e. the length of each perm. `fbb`
    is the space's `FrozenBagBag`, meaning a bag where each key is the number
    of recurrences of an item and each count is the number of different items
    that have this number of recurrences. (See documentation of `FrozenBagBag`
    for more info.)

    It's assumed that the space is not a `CombSpace`, it's not fixed, not
    degreed and not sliced.
    """

_length_of_recurrent_comb_space_cache: Incomplete

def calculate_length_of_recurrent_comb_space(k: Any, fbb: Any) -> Any:
    """
    Calculate the length of a recurrent `CombSpace`.

    `k` is the `n_elements` of the space, i.e. the length of each perm. `fbb`
    is the space's `FrozenBagBag`, meaning a bag where each key is the number
    of recurrences of an item and each count is the number of different items
    that have this number of recurrences. (See documentation of `FrozenBagBag`
    for more info.)

    It's assumed that the space is not fixed, not degreed and not sliced.
    """




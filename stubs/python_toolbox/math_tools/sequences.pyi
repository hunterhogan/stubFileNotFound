from _typeshed import Incomplete
from typing import Any

infinity: Incomplete
_stirling_caches: Incomplete
_n_highest_cache_completed: int

def stirling(n: Any, k: Any, skip_calculation: bool = False) -> Any:
    """
    Calculate Stirling number of the second kind of `n` and `k`.

    More information about these numbers:
    https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind

    Example:

        >>> stirling(3, 2)
        -3

    """
def abs_stirling(n: Any, k: Any) -> Any:
    """
    Calculate Stirling number of the first kind of `n` and `k`.

    More information about these numbers:
    https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind

    Example:

        >>> abs_stirling(3, 2)
        3

    """




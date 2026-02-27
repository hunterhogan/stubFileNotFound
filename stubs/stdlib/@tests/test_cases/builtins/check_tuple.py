from __future__ import annotations

from typing import assert_type, Tuple

# Empty tuples, see #8275
class TupleSub(tuple[int, ...]):
    pass


assert_type(TupleSub(), TupleSub)
assert_type(TupleSub([1, 2, 3]), TupleSub)

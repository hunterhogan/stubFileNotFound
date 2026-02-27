from __future__ import annotations

from collections.abc import Iterator
from typing import assert_type

class OldStyleIter:
    def __getitem__(self, index: int) -> str:
        return str(index)


for x in iter(OldStyleIter()):
    assert_type(x, str)

assert_type(iter(OldStyleIter()), Iterator[str])
assert_type(next(iter(OldStyleIter())), str)

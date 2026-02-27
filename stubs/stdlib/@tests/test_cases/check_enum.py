from __future__ import annotations

from typing import assert_type, Literal, Type
import enum
import sys

A = enum.Enum("A", "spam eggs bacon")
B = enum.Enum("B", ["spam", "eggs", "bacon"])
C = enum.Enum("C", [("spam", 1), ("eggs", 2), ("bacon", 3)])
D = enum.Enum("D", {"spam": 1, "eggs": 2})

assert_type(A, type[A])
assert_type(B, type[B])
assert_type(C, type[C])
assert_type(D, type[D])


class EnumOfTuples(enum.Enum):
    X = 1, 2, 3
    Y = 4, 5, 6


assert_type(EnumOfTuples((1, 2, 3)), EnumOfTuples)

# TODO: ideally this test would pass:
#
# if sys.version_info >= (3, 12):
#     assert_type(EnumOfTuples(1, 2, 3), EnumOfTuples)


if sys.version_info >= (3, 11):

    class Foo(enum.StrEnum):
        X = enum.auto()

    assert_type(Foo.X, Literal[Foo.X])
    assert_type(Foo.X.value, str)

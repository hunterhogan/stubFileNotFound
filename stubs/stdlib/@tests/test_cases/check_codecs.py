from __future__ import annotations

from typing import assert_type
import codecs

assert_type(codecs.decode("x", "unicode-escape"), str)
assert_type(codecs.decode(b"x", "unicode-escape"), str)

assert_type(codecs.decode(b"x", "utf-8"), str)
codecs.decode("x", "utf-8")  # type: ignore

assert_type(codecs.decode("ab", "hex"), bytes)
assert_type(codecs.decode(b"ab", "hex"), bytes)

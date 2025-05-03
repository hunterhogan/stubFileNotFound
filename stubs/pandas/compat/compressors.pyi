import _abc
import bz2
import lzma
from pickle import PickleBuffer
from typing import ClassVar

PY310: bool
has_bz2: bool
has_lzma: bool
def flatten_buffer(b: bytes | bytearray | memoryview | PickleBuffer) -> bytes | bytearray | memoryview:
    """
    Return some 1-D `uint8` typed buffer.

    Coerces anything that does not match that description to one that does
    without copying if possible (otherwise will copy).
    """

class BZ2File(bz2.BZ2File):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...

class LZMAFile(lzma.LZMAFile):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...

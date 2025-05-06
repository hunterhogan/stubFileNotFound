import bz2
import lzma
from pickle import PickleBuffer

has_bz2: bool
has_lzma: bool

def flatten_buffer(b: bytes | bytearray | memoryview | PickleBuffer) -> bytes | bytearray | memoryview:
    """
    Return some 1-D `uint8` typed buffer.

    Coerces anything that does not match that description to one that does
    without copying if possible (otherwise will copy).
    """

class BZ2File(bz2.BZ2File):
    def write(self, b) -> int: ...

class LZMAFile(lzma.LZMAFile):
    def write(self, b) -> int: ...

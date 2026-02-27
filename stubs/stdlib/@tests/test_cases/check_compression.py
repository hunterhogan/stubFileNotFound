from __future__ import annotations

from _typeshed import ReadableBuffer
from bz2 import BZ2Decompressor
from lzma import LZMADecompressor
from typing import assert_type, cast
from zlib import decompressobj
import io
import sys

if sys.version_info >= (3, 14):
    from compression._common._streams import _Decompressor, _Reader, DecompressReader
    from compression.zstd import ZstdDecompressor
else:
    from _compression import _Decompressor, _Reader, DecompressReader

###
# Tests for DecompressReader/_Decompressor
###


class CustomDecompressor:
    def decompress(self, data: ReadableBuffer, max_length: int = -1) -> bytes:
        return b""

    @property
    def unused_data(self) -> bytes:
        return b""

    @property
    def eof(self) -> bool:
        return False

    @property
    def needs_input(self) -> bool:
        return False


def accept_decompressor(d: _Decompressor) -> None:
    d.decompress(b"random bytes", 0)
    assert_type(d.eof, bool)
    assert_type(d.unused_data, bytes)


fp = cast(_Reader, io.BytesIO(b"hello world"))
DecompressReader(fp, decompressobj)
DecompressReader(fp, BZ2Decompressor)
DecompressReader(fp, LZMADecompressor)
DecompressReader(fp, CustomDecompressor)
accept_decompressor(decompressobj())
accept_decompressor(BZ2Decompressor())
accept_decompressor(LZMADecompressor())
accept_decompressor(CustomDecompressor())

if sys.version_info >= (3, 14):
    DecompressReader(fp, ZstdDecompressor)
    accept_decompressor(ZstdDecompressor())

from _io import (  # used elsewhere in typeshed
	_BufferedIOBase, _IOBase, _RawIOBase, _TextIOBase, _WrappedBuffer as _WrappedBuffer,
	BlockingIOError as BlockingIOError, BufferedRandom as BufferedRandom, BufferedReader as BufferedReader,
	BufferedRWPair as BufferedRWPair, BufferedWriter as BufferedWriter, BytesIO as BytesIO,
	DEFAULT_BUFFER_SIZE as DEFAULT_BUFFER_SIZE, FileIO as FileIO, IncrementalNewlineDecoder as IncrementalNewlineDecoder,
	open as open, open_code as open_code, StringIO as StringIO, TextIOWrapper as TextIOWrapper)
from typing import Final, Protocol, TypeVar
import abc
import sys

__all__ = [
    "SEEK_CUR",
    "SEEK_END",
    "SEEK_SET",
    "BlockingIOError",
    "BufferedIOBase",
    "BufferedRWPair",
    "BufferedRandom",
    "BufferedReader",
    "BufferedWriter",
    "BytesIO",
    "FileIO",
    "IOBase",
    "RawIOBase",
    "StringIO",
    "TextIOBase",
    "TextIOWrapper",
    "UnsupportedOperation",
    "open",
    "open_code",
]

if sys.version_info >= (3, 14):
    __all__ += ["Reader", "Writer"]

if sys.version_info >= (3, 11):
    from _io import text_encoding as text_encoding

    __all__ += ["DEFAULT_BUFFER_SIZE", "IncrementalNewlineDecoder", "text_encoding"]

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)

SEEK_SET: Final = 0
SEEK_CUR: Final = 1
SEEK_END: Final = 2

class UnsupportedOperation(OSError, ValueError): ...
class IOBase(_IOBase, metaclass=abc.ABCMeta): ...
class RawIOBase(_RawIOBase, IOBase): ...
class BufferedIOBase(_BufferedIOBase, IOBase): ...
class TextIOBase(_TextIOBase, IOBase): ...

if sys.version_info >= (3, 14):
    class Reader(Protocol[_T_co]):
        __slots__ = ()
        def read(self, size: int = ..., /) -> _T_co: ...

    class Writer(Protocol[_T_contra]):
        __slots__ = ()
        def write(self, data: _T_contra, /) -> int: ...

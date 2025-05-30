import os
from .exceptions import *
from .ext import ExtType, Timestamp
from .fallback import Packer, Unpacker, unpackb
from ._cmsgpack import Packer, Unpacker, unpackb

version = ...
__version__ = ...
if os.environ.get("MSGPACK_PUREPYTHON"):
    ...
else:
    ...
def pack(o, stream, **kwargs): # -> None:
    """
    Pack object `o` and write it to `stream`

    See :class:`Packer` for options.
    """
    ...

def packb(o, **kwargs): # -> None:
    """
    Pack object `o` and return packed bytes

    See :class:`Packer` for options.
    """
    ...

def unpack(stream, **kwargs): # -> int | Any | list[Any] | tuple[Any, ...] | dict[Any, Any] | bytes | str | float | datetime | Timestamp | ExtType | bytearray | bool | None:
    """
    Unpack an object from `stream`.

    Raises `ExtraData` when `stream` contains extra bytes.
    See :class:`Unpacker` for options.
    """
    ...

load = ...
loads = ...
dump = ...
dumps = ...

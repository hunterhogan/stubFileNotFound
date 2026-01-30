from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from collections.abc import Mapping
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import bytechr as bytechr, byteord as byteord, strjoin as strjoin

hdmxHeaderFormat: str

class _GlyphnamedList(Mapping):
    _array: Incomplete
    _map: Incomplete
    def __init__(self, reverseGlyphOrder, data) -> None: ...
    def __getitem__(self, k): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def keys(self): ...

class table__h_d_m_x(DefaultTable.DefaultTable):
    """Horizontal Device Metrics table

    The ``hdmx`` table is an optional table that stores advance widths for
    glyph outlines at specified pixel sizes.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/hdmx
    """
    hdmx: Incomplete
    def decompile(self, data, ttFont) -> None: ...
    version: int
    recordSize: Incomplete
    numRecords: Incomplete
    def compile(self, ttFont): ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...

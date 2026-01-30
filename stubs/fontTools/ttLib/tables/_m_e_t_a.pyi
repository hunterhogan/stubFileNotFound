from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import bytesjoin as bytesjoin, readHex as readHex, strjoin as strjoin
from fontTools.ttLib import TTLibError as TTLibError

META_HEADER_FORMAT: str
DATA_MAP_FORMAT: str

class table__m_e_t_a(DefaultTable.DefaultTable):
    """Metadata table

    The ``meta`` table contains various metadata values for the font. Each
    category of metadata in the table is identified by a four-character tag.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/meta
    """
    data: Incomplete
    def __init__(self, tag=None) -> None: ...
    def decompile(self, data, ttFont) -> None: ...
    def compile(self, ttFont): ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...

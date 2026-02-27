from . import DefaultTable as DefaultTable, ttProgram as ttProgram
from _typeshed import Incomplete
from fontTools.ttLib import TTFont

class table__f_p_g_m(DefaultTable.DefaultTable):
    """Font Program table

    The ``fpgm`` table typically contains function defintions that are
    used by font instructions. This Font Program is similar to the Control
    Value Program that is stored in the ``prep`` table, but
    the ``fpgm`` table is only executed one time, when the font is first
    used.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/fpgm
    """

    program: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    def __bool__(self) -> bool:
        """
        >>> fpgm = table__f_p_g_m()
        >>> bool(fpgm)
        False
        >>> p = ttProgram.Program()
        >>> fpgm.program = p
        >>> bool(fpgm)
        False
        >>> bc = bytearray([0])
        >>> p.fromBytecode(bc)
        >>> bool(fpgm)
        True
        >>> p.bytecode.pop()
        0
        >>> bool(fpgm)
        False
        """
    __nonzero__ = __bool__

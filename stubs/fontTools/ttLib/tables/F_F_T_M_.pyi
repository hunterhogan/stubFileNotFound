from . import DefaultTable as DefaultTable
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.misc.timeTools import timestampFromString as timestampFromString, timestampToString as timestampToString
from fontTools.ttLib import TTFont

FFTMFormat: str

class table_F_F_T_M_(DefaultTable.DefaultTable):
    """FontForge Time Stamp table

    The ``FFTM`` table is used by the free-software font editor
    FontForge to record timestamps for the creation and modification
    of font source (.sfd) files and a timestamp for FontForge's
    own source code.

    See also https://fontforge.org/docs/techref/non-standard.html
    """

    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

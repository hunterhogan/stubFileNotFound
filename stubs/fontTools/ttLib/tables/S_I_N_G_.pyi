from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import bytechr as bytechr, byteord as byteord, safeEval as safeEval, tobytes as tobytes, tostr as tostr

SINGFormat: str

class table_S_I_N_G_(DefaultTable.DefaultTable):
    """Glyphlets SING table

    The ``SING`` table is used by Adobe's SING Glyphlets.

    See also https://web.archive.org/web/20080627183635/http://www.adobe.com/devnet/opentype/gdk/topic.html
    """
    dependencies: Incomplete
    uniqueName: Incomplete
    nameLength: Incomplete
    baseGlyphName: Incomplete
    METAMD5: Incomplete
    def decompile(self, data, ttFont) -> None: ...
    def decompileUniqueName(self, data): ...
    def compile(self, ttFont): ...
    def compilecompileUniqueName(self, name, length): ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...

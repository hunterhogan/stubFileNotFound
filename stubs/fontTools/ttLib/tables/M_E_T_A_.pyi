from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import byteord as byteord, safeEval as safeEval
from fontTools.ttLib import TTFont

METAHeaderFormat: str
METAGlyphRecordFormat: str
METAStringRecordFormat: str
METALabelDict: Incomplete

def getLabelString(labelID): ...

class table_M_E_T_A_(DefaultTable.DefaultTable):
    """Glyphlets META table

    The ``META`` table is used by Adobe's SING Glyphlets.

    See also https://web.archive.org/web/20080627183635/http://www.adobe.com/devnet/opentype/gdk/topic.html
    """

    dependencies: Incomplete
    glyphRecords: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    nMetaRecs: Incomplete
    metaFlags: Incomplete
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

class GlyphRecord:
    glyphID: int
    nMetaEntry: int
    offset: int
    stringRecs: Incomplete
    def __init__(self) -> None: ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    def compile(self, parentTable): ...

def mapXMLToUTF8(string): ...
def mapUTF8toXML(string): ...

class StringRecord:
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    string: Incomplete
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    def compile(self, parentTable): ...

from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.ttLib import TTFont

log: Incomplete
tsi0Format: str

def fixlongs(glyphID, textLength, textOffset): ...

class table_T_S_I__0(DefaultTable.DefaultTable):
    dependencies: Incomplete
    indices: Incomplete
    extra_indices: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def set(self, indices, extra_indices) -> None: ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...

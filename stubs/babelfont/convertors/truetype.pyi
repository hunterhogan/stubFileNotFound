
from babelfont.convertors import BaseConvertor
from fontTools.ttLib.ttGlyphSet import _TTGlyph
from typing import Dict

def compile_panose(data): # -> Panose:
    ...

class TrueType(BaseConvertor):
    suffix = ...
    SAVE_FILTERS = ...
    def calculate_a_gvar(self, f, model, g, ttglyphsets: dict[str, dict[str, _TTGlyph]]): # -> list[Any] | None:
        ...



class OpenType(TrueType):
    suffix = ...

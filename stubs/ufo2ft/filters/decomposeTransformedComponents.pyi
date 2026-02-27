
from fontTools.misc.transform import Identity
from typing import TypeAlias
from ufo2ft.filters.decomposeComponents import DecomposeComponentsFilter, DecomposeComponentsIFilter

IDENTITY_2x2: TypeAlias = Identity[: 4]
class DecomposeTransformedComponentsFilter(DecomposeComponentsFilter):
    def filter(self, glyph): # -> bool:
        ...



class DecomposeTransformedComponentsIFilter(DecomposeComponentsIFilter):
    def filter(self, glyphName, glyphs): # -> bool:
        ...

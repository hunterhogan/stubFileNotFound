
from defcon.pens.glyphObjectPointPen import GlyphObjectPointPen

class DecomposeComponentPointPen(GlyphObjectPointPen):
    def __init__(self, glyph, layer) -> None:
        ...

    def addComponent(self, baseGlyphName, transformation, identifier=..., **kwargs): # -> None:
        ...

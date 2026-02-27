
from fontParts.base import BaseLayer
from fontParts.fontshell.base import RBaseObject
from fontParts.fontshell.glyph import RGlyph
from fontParts.fontshell.lib import RLib

class RLayer(RBaseObject, BaseLayer):
    wrapClass = ...
    libClass = RLib
    glyphClass = RGlyph

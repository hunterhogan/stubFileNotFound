
from fontParts.base import BaseGlyph
from fontParts.fontshell.anchor import RAnchor
from fontParts.fontshell.base import RBaseObject
from fontParts.fontshell.component import RComponent
from fontParts.fontshell.contour import RContour
from fontParts.fontshell.guideline import RGuideline
from fontParts.fontshell.image import RImage
from fontParts.fontshell.lib import RLib

class RGlyph(RBaseObject, BaseGlyph):
    wrapClass = ...
    contourClass = RContour
    componentClass = RComponent
    anchorClass = RAnchor
    guidelineClass = RGuideline
    imageClass = RImage
    libClass = RLib
    def getPen(self):
        ...

    def getPointPen(self):
        ...

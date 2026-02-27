
from fontParts.base import BaseContour
from fontParts.fontshell.base import RBaseObject
from fontParts.fontshell.bPoint import RBPoint
from fontParts.fontshell.point import RPoint
from fontParts.fontshell.segment import RSegment

class RContour(RBaseObject, BaseContour):
    wrapClass = ...
    pointClass = RPoint
    segmentClass = RSegment
    bPointClass = RBPoint

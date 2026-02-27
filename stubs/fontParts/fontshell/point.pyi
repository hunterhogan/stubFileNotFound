
from fontParts.base import BasePoint
from fontParts.fontshell.base import RBaseObject

class RPoint(RBaseObject, BasePoint):
    wrapClass = ...
    def changed(self): # -> None:
        ...

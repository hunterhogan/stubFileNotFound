from _typeshed import Incomplete
from fontTools.ttLib.ttFont import TTFont
from fontTools.ttLib.ttVisitor import TTVisitor

__all__ = ['scale_upem', 'ScalerVisitor']

class ScalerVisitor(TTVisitor):
    scaleFactor: Incomplete
    def __init__(self, scaleFactor) -> None: ...
    def scale(self, v): ...

def scale_upem(font: TTFont, new_upem: int) -> None:
    """Change the units-per-EM of font to the new value."""


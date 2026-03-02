from .parser import parse_path as parse_path
from _typeshed import Incomplete

__all__ = ['SVGPath', 'parse_path']

class SVGPath:
    """Parse SVG ``path`` elements from a file or string, and draw them
    onto a glyph object that supports the FontTools Pen protocol.

    For example, reading from an SVG file and drawing to a Defcon Glyph:

    .. code-block::

        import defcon
        glyph = defcon.Glyph()
        pen = glyph.getPen()
        svg = SVGPath("path/to/a.svg")
        svg.draw(pen)

    Or reading from a string containing SVG data, using the alternative
    \'fromstring\' (a class method):

    .. code-block::

        data = \'<?xml version="1.0" ...\'
        svg = SVGPath.fromstring(data)
        svg.draw(pen)

    Both constructors can optionally take a \'transform\' matrix (6-float
    tuple, or a FontTools Transform object) to modify the draw output.
    """

    root: Incomplete
    transform: Incomplete
    def __init__(self, filename=None, transform=None) -> None: ...
    @classmethod
    def fromstring(cls, data, transform=None): ...
    def draw(self, pen) -> None: ...

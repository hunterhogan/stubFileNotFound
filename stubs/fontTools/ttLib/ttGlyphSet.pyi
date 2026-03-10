from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from fontTools.misc.transform import DecomposedTransform as DecomposedTransform, Transform as Transform
from fontTools.misc.vector import Vector as Vector
from fontTools.pens.basePen import BasePen
from fontTools.pens.pointPen import AbstractPointPen
from fontTools.pens.recordingPen import DecomposingRecordingPen as DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPen as TransformPen, TransformPointPen as TransformPointPen
from fontTools.ttLib import TTFont
import abc

class _TTGlyphSet(Mapping, metaclass=abc.ABCMeta):
    """Generic dict-like GlyphSet class that pulls metrics from hmtx and glyph shape from TrueType or CFF."""

    defaultLocationNormalized: Incomplete
    depth: int
    font: TTFont
    glyphsMapping: Mapping[str, _TTGlyph]
    hMetrics: Incomplete
    hvarInstancer: Incomplete
    hvarTable: Incomplete
    location: Incomplete
    locationStack: Incomplete
    originalLocation: Incomplete
    rawLocation: Incomplete
    rawLocationStack: Incomplete
    recalcBounds: bool
    vMetrics: Incomplete
    def __init__(self, font: TTFont, location, glyphsMapping: Mapping[str, _TTGlyph], *, recalcBounds: bool = True) -> None: ...
    @contextmanager
    def pushLocation(self, location, reset: bool): ...
    @contextmanager
    def pushDepth(self) -> Generator[Incomplete]: ...
    def __contains__(self, glyphName: str) -> bool: ...  # pyright: ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def has_key(self, glyphName: str) -> bool: ...

class _TTGlyphSetGlyf(_TTGlyphSet):
    glyfTable: Incomplete
    gvarTable: Incomplete
    def __init__(self, font: TTFont, location, recalcBounds: bool = True) -> None: ...
    def __getitem__(self, glyphName: str): ...

class _TTGlyphSetCFF(_TTGlyphSet):
    charStrings: Incomplete
    def __init__(self, font: TTFont, location) -> None: ...
    def __getitem__(self, glyphName: str): ...
    blender: Incomplete
    def setLocation(self, location) -> None: ...
    @contextmanager
    def pushLocation(self, location, reset: bool): ...

class _TTGlyphSetVARC(_TTGlyphSet):
    glyphSet: Incomplete
    varcTable: Incomplete
    def __init__(self, font: TTFont, location, glyphSet) -> None: ...
    def __getitem__(self, glyphName: str): ...

class _TTGlyph(ABC, metaclass=abc.ABCMeta):
    """Glyph object that supports the Pen protocol, meaning that it has .draw() and .drawPoints() methods that take a pen object as their only argument.

    Additionally there are 'width' and 'lsb' attributes, read from
    the 'hmtx' table.

    If the font contains a 'vmtx' table, there will also be 'height' and 'tsb'
    attributes.

    """

    glyphSet: _TTGlyphSet
    name: str
    recalcBounds: bool
    def __init__(self, glyphSet: _TTGlyphSet, glyphName: str, *, recalcBounds: bool = True) -> None: ...
    @abstractmethod
    def draw(self, pen: BasePen) -> None:
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details how that works."""
    def drawPoints(self, pen: AbstractPointPen) -> None:
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details how that works."""

class _TTGlyphGlyf(_TTGlyph):
    def draw(self, pen: BasePen) -> None:
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details how that works."""
    def drawPoints(self, pen: AbstractPointPen) -> None:
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details how that works."""

class _TTGlyphCFF(_TTGlyph):
    def draw(self, pen: BasePen) -> None:
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details how that works."""

class _TTGlyphVARC(_TTGlyph):
    def draw(self, pen: BasePen) -> None: ...
    def drawPoints(self, pen: AbstractPointPen) -> None: ...

class LerpGlyphSet(Mapping):
    """A glyphset that interpolates between two other glyphsets.

    Factor is typically between 0 and 1. 0 means the first glyphset,
    1 means the second glyphset, and 0.5 means the average of the
    two glyphsets. Other values are possible, and can be useful to
    extrapolate. Defaults to 0.5.

    """

    glyphset1: Incomplete
    glyphset2: Incomplete
    factor: Incomplete
    def __init__(self, glyphset1, glyphset2, factor: float = 0.5) -> None: ...
    def __getitem__(self, glyphname): ...
    def __contains__(self, glyphname) -> bool: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...

class LerpGlyph:
    glyphset: Incomplete
    glyphname: Incomplete
    def __init__(self, glyphname, glyphset) -> None: ...
    def draw(self, pen) -> None: ...

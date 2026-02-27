from _typeshed import Incomplete
from fontTools.pens.basePen import AbstractPen as AbstractPen, DecomposingPen as DecomposingPen
from fontTools.pens.pointPen import (
	AbstractPointPen as AbstractPointPen, DecomposingPointPen as DecomposingPointPen, ReverseFlipped as ReverseFlipped)
from fontTools.pens.recordingPen import RecordingPen as RecordingPen

class _PassThruComponentsMixin:
    def addComponent(self, glyphName, transformation, **kwargs) -> None: ...

class FilterPen(_PassThruComponentsMixin, AbstractPen):
    """Base class for pens that apply some transformation to the coordinates
    they receive and pass them to another pen.

    You can override any of its methods. The default implementation does
    nothing, but passes the commands unmodified to the other pen.

    >>> from fontTools.pens.recordingPen import RecordingPen
    >>> rec = RecordingPen()
    >>> pen = FilterPen(rec)
    >>> v = iter(rec.value)

    >>> pen.moveTo((0, 0))
    >>> next(v)
    ('moveTo', ((0, 0),))

    >>> pen.lineTo((1, 1))
    >>> next(v)
    ('lineTo', ((1, 1),))

    >>> pen.curveTo((2, 2), (3, 3), (4, 4))
    >>> next(v)
    ('curveTo', ((2, 2), (3, 3), (4, 4)))

    >>> pen.qCurveTo((5, 5), (6, 6), (7, 7), (8, 8))
    >>> next(v)
    ('qCurveTo', ((5, 5), (6, 6), (7, 7), (8, 8)))

    >>> pen.closePath()
    >>> next(v)
    ('closePath', ())

    >>> pen.moveTo((9, 9))
    >>> next(v)
    ('moveTo', ((9, 9),))

    >>> pen.endPath()
    >>> next(v)
    ('endPath', ())

    >>> pen.addComponent('foo', (1, 0, 0, 1, 0, 0))
    >>> next(v)
    ('addComponent', ('foo', (1, 0, 0, 1, 0, 0)))
    """

    _outPen: Incomplete
    current_pt: Incomplete
    def __init__(self, outPen) -> None: ...
    def moveTo(self, pt) -> None: ...
    def lineTo(self, pt) -> None: ...
    def curveTo(self, *points) -> None: ...
    def qCurveTo(self, *points) -> None: ...
    def closePath(self) -> None: ...
    def endPath(self) -> None: ...

class ContourFilterPen(_PassThruComponentsMixin, RecordingPen):
    """A "buffered" filter pen that accumulates contour data, passes
    it through a ``filterContour`` method when the contour is closed or ended,
    and finally draws the result with the output pen.

    Components are passed through unchanged.
    """

    _outPen: Incomplete
    def __init__(self, outPen) -> None: ...
    def closePath(self) -> None: ...
    def endPath(self) -> None: ...
    value: Incomplete
    def _flushContour(self) -> None: ...
    def filterContour(self, contour) -> None:
        """Subclasses must override this to perform the filtering.

        The contour is a list of pen (operator, operands) tuples.
        Operators are strings corresponding to the AbstractPen methods:
        "moveTo", "lineTo", "curveTo", "qCurveTo", "closePath" and
        "endPath". The operands are the positional arguments that are
        passed to each method.

        If the method doesn\'t return a value (i.e. returns None), it\'s
        assumed that the argument was modified in-place.
        Otherwise, the return value is drawn with the output pen.
        """

class FilterPointPen(_PassThruComponentsMixin, AbstractPointPen):
    """Baseclass for point pens that apply some transformation to the
    coordinates they receive and pass them to another point pen.

    You can override any of its methods. The default implementation does
    nothing, but passes the commands unmodified to the other pen.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> rec = RecordingPointPen()
    >>> pen = FilterPointPen(rec)
    >>> v = iter(rec.value)
    >>> pen.beginPath(identifier="abc")
    >>> next(v)
    (\'beginPath\', (), {\'identifier\': \'abc\'})
    >>> pen.addPoint((1, 2), "line", False)
    >>> next(v)
    (\'addPoint\', ((1, 2), \'line\', False, None), {})
    >>> pen.addComponent("a", (2, 0, 0, 2, 10, -10), identifier="0001")
    >>> next(v)
    (\'addComponent\', (\'a\', (2, 0, 0, 2, 10, -10)), {\'identifier\': \'0001\'})
    >>> pen.endPath()
    >>> next(v)
    (\'endPath\', (), {})
    """

    _outPen: Incomplete
    def __init__(self, outPen) -> None: ...
    def beginPath(self, identifier=None, **kwargs) -> None: ...
    def endPath(self) -> None: ...
    def addPoint(self, pt, segmentType=None, smooth: bool = False, name=None, identifier=None, **kwargs) -> None: ...

class _DecomposingFilterMixinBase:
    """Base mixin class with common `addComponent` logic for decomposing filter pens."""

    include: Incomplete
    def addComponent(self, baseGlyphName, transformation, **kwargs) -> None: ...

class _DecomposingFilterPenMixin(_DecomposingFilterMixinBase):
    """Mixin class that decomposes components as regular contours for segment pens.

    Used by DecomposingFilterPen.

    Takes two required parameters, another segment pen 'outPen' to draw
    with, and a 'glyphSet' dict of drawable glyph objects to draw components from.

    The 'skipMissingComponents' and 'reverseFlipped' optional arguments work the
    same as in the DecomposingPen. reverseFlipped is bool only (True/False).

    In addition, the decomposing filter pens also take the following two options:

    'include' is an optional set of component base glyph names to consider for
    decomposition; the default include=None means decompose all components no matter
    the base glyph name).

    'decomposeNested' (bool) controls whether to recurse decomposition into nested
    components of components (this only matters when 'include' was also provided);
    if False, only decompose top-level components included in the set, but not
    also their children.
    """

    skipMissingComponents: bool
    include: Incomplete
    decomposeNested: Incomplete
    def __init__(self, outPen, glyphSet, skipMissingComponents=None, reverseFlipped: bool = False, include: set[str] | None = None, decomposeNested: bool = True, **kwargs) -> None: ...

class _DecomposingFilterPointPenMixin(_DecomposingFilterMixinBase):
    """Mixin class that decomposes components as regular contours for point pens.

    Takes two required parameters, another point pen 'outPen' to draw
    with, and a 'glyphSet' dict of drawable glyph objects to draw components from.

    The 'skipMissingComponents' and 'reverseFlipped' optional arguments work the
    same as in the DecomposingPointPen. reverseFlipped accepts bool | ReverseFlipped
    (see DecomposingPointPen).

    In addition, the decomposing filter pens also take the following two options:

    'include' is an optional set of component base glyph names to consider for
    decomposition; the default include=None means decompose all components no matter
    the base glyph name).

    'decomposeNested' (bool) controls whether to recurse decomposition into nested
    components of components (this only matters when 'include' was also provided);
    if False, only decompose top-level components included in the set, but not
    also their children.
    """

    skipMissingComponents: bool
    include: Incomplete
    decomposeNested: Incomplete
    def __init__(self, outPen, glyphSet, skipMissingComponents=None, reverseFlipped: bool | ReverseFlipped = False, include: set[str] | None = None, decomposeNested: bool = True, **kwargs) -> None: ...

class DecomposingFilterPen(_DecomposingFilterPenMixin, DecomposingPen, FilterPen):
    """Filter pen that draws components as regular contours."""
class DecomposingFilterPointPen(_DecomposingFilterPointPenMixin, DecomposingPointPen, FilterPointPen):
    """Filter point pen that draws components as regular contours."""

class ContourFilterPointPen(_PassThruComponentsMixin, AbstractPointPen):
    """A "buffered" filter point pen that accumulates contour data, passes
    it through a ``filterContour`` method when the contour is closed or ended,
    and finally draws the result with the output point pen.

    Components are passed through unchanged.

    The ``filterContour`` method can modify the contour in-place (return None)
    or return a new contour to replace it.
    """

    _outPen: Incomplete
    currentContour: Incomplete
    currentContourKwargs: Incomplete
    def __init__(self, outPen) -> None: ...
    def beginPath(self, identifier=None, **kwargs) -> None: ...
    def endPath(self) -> None: ...
    def _flushContour(self) -> None:
        """Flush the current contour to the output pen."""
    def filterContour(self, contour) -> None:
        """Subclasses must override this to perform the filtering.

        The contour is a list of (pt, segmentType, smooth, name, kwargs) tuples.
        If the method doesn't return a value (i.e. returns None), it's
        assumed that the contour was modified in-place.
        Otherwise, the return value replaces the original contour.
        """
    def addPoint(self, pt, segmentType=None, smooth: bool = False, name=None, identifier=None, **kwargs) -> None: ...

class OnCurveFirstPointPen(ContourFilterPointPen):
    """Filter point pen that ensures closed contours start with an on-curve point.

    If a closed contour starts with an off-curve point (segmentType=None), it rotates
    the points list so that the first on-curve point (segmentType != None) becomes
    the start point. Open contours and contours already starting with on-curve points
    are passed through unchanged.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> rec = RecordingPointPen()
    >>> pen = OnCurveFirstPointPen(rec)
    >>> # Closed contour starting with off-curve - will be rotated
    >>> pen.beginPath()
    >>> pen.addPoint((0, 0), None)  # off-curve
    >>> pen.addPoint((100, 100), "line")  # on-curve - will become start
    >>> pen.addPoint((200, 0), None)  # off-curve
    >>> pen.addPoint((300, 100), "curve")  # on-curve
    >>> pen.endPath()
    >>> # The contour should now start with (100, 100) "line"
    >>> rec.value[0]
    (\'beginPath\', (), {})
    >>> rec.value[1]
    (\'addPoint\', ((100, 100), \'line\', False, None), {})
    >>> rec.value[2]
    (\'addPoint\', ((200, 0), None, False, None), {})
    >>> rec.value[3]
    (\'addPoint\', ((300, 100), \'curve\', False, None), {})
    >>> rec.value[4]
    (\'addPoint\', ((0, 0), None, False, None), {})
    """

    def filterContour(self, contour):
        """Rotate closed contour to start with first on-curve point if needed."""

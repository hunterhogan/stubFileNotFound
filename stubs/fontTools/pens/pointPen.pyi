from _typeshed import Incomplete
from fontTools.misc.enumTools import StrEnum
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform
from fontTools.pens.basePen import AbstractPen, MissingComponentError
from typing import Any

__all__ = ['AbstractPointPen', 'BasePointToSegmentPen', 'PointToSegmentPen', 'SegmentToPointPen', 'GuessSmoothPointPen', 'ReverseContourPointPen', 'ReverseFlipped']

Point = tuple[float, float]
PointName = str | None
SegmentPointList = list[tuple[Point | None, bool, PointName, Any]]
SegmentType = str | None
SegmentList = list[tuple[SegmentType, SegmentPointList]]

class ReverseFlipped(StrEnum):
    """How to handle flipped components during decomposition.

    NO: Don't reverse flipped components
    KEEP_START: Reverse flipped components, keeping original starting point
    ON_CURVE_FIRST: Reverse flipped components, ensuring first point is on-curve
    """
    NO = 'no'
    KEEP_START = 'keep_start'
    ON_CURVE_FIRST = 'on_curve_first'

class AbstractPointPen:
    """Baseclass for all PointPens."""
    def beginPath(self, identifier: str | None = None, **kwargs: Any) -> None:
        """Start a new sub path."""
    def endPath(self) -> None:
        """End the current sub path."""
    def addPoint(self, pt: tuple[float, float], segmentType: str | None = None, smooth: bool = False, name: str | None = None, identifier: str | None = None, **kwargs: Any) -> None:
        """Add a point to the current sub path."""
    def addComponent(self, baseGlyphName: str, transformation: tuple[float, float, float, float, float, float], identifier: str | None = None, **kwargs: Any) -> None:
        """Add a sub glyph."""
    def addVarComponent(self, glyphName: str, transformation: DecomposedTransform, location: dict[str, float], identifier: str | None = None, **kwargs: Any) -> None:
        """Add a VarComponent sub glyph. The 'transformation' argument
        must be a DecomposedTransform from the fontTools.misc.transform module,
        and the 'location' argument must be a dictionary mapping axis tags
        to their locations.
        """

class BasePointToSegmentPen(AbstractPointPen):
    """
    Base class for retrieving the outline in a segment-oriented
    way. The PointPen protocol is simple yet also a little tricky,
    so when you need an outline presented as segments but you have
    as points, do use this base implementation as it properly takes
    care of all the edge cases.
    """
    currentPath: Incomplete
    def __init__(self) -> None: ...
    def beginPath(self, identifier=None, **kwargs) -> None: ...
    def _flushContour(self, segments: SegmentList) -> None:
        '''Override this method.

        It will be called for each non-empty sub path with a list
        of segments: the \'segments\' argument.

        The segments list contains tuples of length 2:
                (segmentType, points)

        segmentType is one of "move", "line", "curve" or "qcurve".
        "move" may only occur as the first segment, and it signifies
        an OPEN path. A CLOSED path does NOT start with a "move", in
        fact it will not contain a "move" at ALL.

        The \'points\' field in the 2-tuple is a list of point info
        tuples. The list has 1 or more items, a point tuple has
        four items:
                (point, smooth, name, kwargs)
        \'point\' is an (x, y) coordinate pair.

        For a closed path, the initial moveTo point is defined as
        the last point of the last segment.

        The \'points\' list of "move" and "line" segments always contains
        exactly one point tuple.
        '''
    def endPath(self) -> None: ...
    def addPoint(self, pt, segmentType=None, smooth: bool = False, name=None, identifier=None, **kwargs) -> None: ...

class PointToSegmentPen(BasePointToSegmentPen):
    """
    Adapter class that converts the PointPen protocol to the
    (Segment)Pen protocol.

    NOTE: The segment pen does not support and will drop point names, identifiers
    and kwargs.
    """
    pen: Incomplete
    outputImpliedClosingLine: Incomplete
    def __init__(self, segmentPen, outputImpliedClosingLine: bool = False) -> None: ...
    def _flushContour(self, segments) -> None: ...
    def addComponent(self, glyphName, transform, identifier=None, **kwargs) -> None: ...

class SegmentToPointPen(AbstractPen):
    """
    Adapter class that converts the (Segment)Pen protocol to the
    PointPen protocol.
    """
    pen: Incomplete
    contour: list[tuple[Point, SegmentType]] | None
    def __init__(self, pointPen, guessSmooth: bool = True) -> None: ...
    def _flushContour(self) -> None: ...
    def moveTo(self, pt) -> None: ...
    def lineTo(self, pt) -> None: ...
    def curveTo(self, *pts) -> None: ...
    def qCurveTo(self, *pts) -> None: ...
    def closePath(self) -> None: ...
    def endPath(self) -> None: ...
    def addComponent(self, glyphName, transform) -> None: ...

class GuessSmoothPointPen(AbstractPointPen):
    '''
    Filtering PointPen that tries to determine whether an on-curve point
    should be "smooth", ie. that it\'s a "tangent" point or a "curve" point.
    '''
    _outPen: Incomplete
    _error: Incomplete
    _points: Incomplete
    def __init__(self, outPen, error: float = 0.05) -> None: ...
    def _flushContour(self) -> None: ...
    def beginPath(self, identifier=None, **kwargs) -> None: ...
    def endPath(self) -> None: ...
    def addPoint(self, pt, segmentType=None, smooth: bool = False, name=None, identifier=None, **kwargs) -> None: ...
    def addComponent(self, glyphName, transformation, identifier=None, **kwargs) -> None: ...
    def addVarComponent(self, glyphName, transformation, location, identifier=None, **kwargs) -> None: ...

class ReverseContourPointPen(AbstractPointPen):
    """
    This is a PointPen that passes outline data to another PointPen, but
    reversing the winding direction of all contours. Components are simply
    passed through unchanged.

    Closed contours are reversed in such a way that the first point remains
    the first point.
    """
    pen: Incomplete
    currentContour: Incomplete
    def __init__(self, outputPointPen) -> None: ...
    def _flushContour(self) -> None: ...
    currentContourIdentifier: Incomplete
    onCurve: Incomplete
    def beginPath(self, identifier=None, **kwargs) -> None: ...
    def endPath(self) -> None: ...
    def addPoint(self, pt, segmentType=None, smooth: bool = False, name=None, identifier=None, **kwargs) -> None: ...
    def addComponent(self, glyphName, transform, identifier=None, **kwargs) -> None: ...

class DecomposingPointPen(LogMixin, AbstractPointPen):
    """Implements a 'addComponent' method that decomposes components
    (i.e. draws them onto self as simple contours).
    It can also be used as a mixin class (e.g. see DecomposingRecordingPointPen).

    You must override beginPath, addPoint, endPath. You may
    additionally override addVarComponent and addComponent.

    By default a warning message is logged when a base glyph is missing;
    set the class variable ``skipMissingComponents`` to False if you want
    all instances of a sub-class to raise a :class:`MissingComponentError`
    exception by default.
    """
    skipMissingComponents: bool
    MissingComponentError = MissingComponentError
    glyphSet: Incomplete
    reverseFlipped: Incomplete
    def __init__(self, glyphSet, *args, skipMissingComponents=None, reverseFlipped: bool | ReverseFlipped = False, **kwargs) -> None:
        """Takes a 'glyphSet' argument (dict), in which the glyphs that are referenced
        as components are looked up by their name.

        If the optional 'reverseFlipped' argument is True or a ReverseFlipped enum value,
        components whose transformation matrix has a negative determinant will be decomposed
        with a reversed path direction to compensate for the flip.

        The reverseFlipped parameter can be:
        - False or ReverseFlipped.NO: Don't reverse flipped components
        - True or ReverseFlipped.KEEP_START: Reverse, keeping original starting point
        - ReverseFlipped.ON_CURVE_FIRST: Reverse, ensuring first point is on-curve

        The optional 'skipMissingComponents' argument can be set to True/False to
        override the homonymous class attribute for a given pen instance.
        """
    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs) -> None:
        """Transform the points of the base glyph and draw it onto self.

        The `identifier` parameter and any extra kwargs are ignored.
        """

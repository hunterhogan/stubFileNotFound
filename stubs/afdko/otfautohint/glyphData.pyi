from _typeshed import Incomplete
from fontTools.pens.basePen import BasePen

def norm_float(value):
    """Converts a float (whose decimal part is zero) to integer"""
def feq(a, b, factor: float = 1.52e-05):
    """Returns True if a and b are close enough to be considered equal"""
def fne(a, b, factor: float = 1.52e-05):
    """Returns True if a and b are not close enough to be considered equal"""

class pt(tuple):
    """A 2-tuple representing a point in 2D space"""
    tl: Incomplete
    @classmethod
    def setAlign(cls, vertical: bool = False) -> None:
        '''
        Class-level method to control the value of properties a and o.
        "a" is meant so suggest "aligned" and "o" is meant to suggest
        "opposite".

        When called with vertical==False (the default):
            a will be equivalent to x
            o will be equivalent to y
        When called with vertical==True
            a will be equivalent to y
            o will be equivalent to x

        Note that the internal align variable is thread-specific
        '''
    @classmethod
    def clearAlign(cls) -> None:
        """
        Class-level method to unset the internal align variable
        so that accessing properties a or o will result in an error
        """
    def __new__(cls, x: int = 0, y: int = 0, roundCoords: bool = False):
        """
        Creates a new pt object initialied with x and y.

        If roundCoords is True the values are rounded before storing
        """
    @property
    def x(self): ...
    @property
    def y(self): ...
    @property
    def a(self):
        """See note in setAlign"""
    @property
    def o(self):
        """See note in setAlign"""
    def ao(self):
        """See note in setAlign"""
    def norm_float(self): ...
    def __add__(self, other):
        """
        Returns a new pt object representing the sum of respective
        values of the two arguments
        """
    def __sub__(self, other):
        """
        Returns a new pt object representing the difference of respective
        values of the two arguments
        """
    def avg(self, other):
        """
        Returns a new pt object representing the average of this pt object
        with the argument
        """
    def dot(self, other):
        """
        Returns a numeric value representing the dot product of this
        pt object with the argument
        """
    def scross(self, other):
        """
        Returns a numeric value representing the cross product of this
        pt object with the argument
        """
    def distsq(self, other):
        """
        Returns a numerical value representing the squared distance
        between this pt object and the argument
        """
    def a_dist(self, other):
        '''
        Returns a numerical value representing the distance between this
        pt object and the argument in the "a" dimension
        '''
    def o_dist(self, other):
        '''
        Returns a numerical value representing the distance between this
        pt object and the argument in the "o" dimension
        '''
    def normsq(self):
        """Returns the squared magnitude of the pt (treated as a vector)"""
    def abs(self):
        """
        Returns a new pt object with the absolute values of the coordinates
        """
    def round(self, dec: int = 0):
        """Returns a new pt object with rounded coordinate values"""
    def __mul__(self, other):
        """
        Returns a new pt object with this object's coordinates multiplied by
        a scalar value
        """
    def __rmul__(self, other):
        """Same as __mul__ for right-multiplication"""
    def __eq__(self, other, factor: float = 1.52e-05):
        """Returns True if each coordinate is feq to that of the argument"""
    def eq_exact(self, other):
        """Returns True if each coordinate is equal to that of the argument"""

class stem(tuple):
    """
    A 2-tuple representing a stem hint.

    self[0] is the bottom/left coordinate
    self[1] is the top/right coordinate

    If self[1] is less than self[0] the stem represents a ghost hint
    and the difference should be 20 or 21 points
    """
    BandMargin: int
    def __new__(cls, lb: int = 0, rt: int = 0): ...
    @property
    def lb(self):
        """The left or bottom value, depending on stem alignment"""
    @property
    def rt(self):
        """The right or top value, depending on stem alignment"""
    def isGhost(self, doBool: bool = False):
        """Returns True if the stem is a ghost hint"""
    def ghostCompat(self, other): ...
    def isBad(self):
        """Returns True if the stem is malformed"""
    def relVals(self, last=None):
        '''
        Returns a tuple of "relative" stem values (start relative to
        the passed last stem, then width) appropriate for
        vstem/hstem/vstemhm/hstemhm output
        '''
    def UFOVals(self):
        """Returns a tuple of stem values appropriate for UFO output"""
    def overlaps(self, other):
        """
        Returns True if this stem is within BandMargin of overlapping the
        passed stem
        """
    def distance(self, loc):
        """
        Returns the distance between this stem and the passed location,
        which is zero if the location falls within the stem
        """

class boundsState:
    """
    Calculates and stores the bounds of a pathElement (spline) and the point
    locations that define the boundaries.
    """
    lb: Incomplete
    bounds: Incomplete
    def __init__(self, c) -> None:
        """
        Initialize the object with the passed pathElement and calculate the
        bounds
        """
    def mergePt(self, b, p, t, doExt: bool = True) -> None:
        """
        Add the passed point into the bounds as a potential extreme.

        If it is an extreme
            store the point at the appropriate extpts subscripts
            store the t value at the same tmap subscripts
        """
    tmap: Incomplete
    extpts: Incomplete
    def linearBounds(self, c):
        """
        Calculate the bounds of the line betwen the start and end points of
        the passed pathElement.
        """
    def calcCurveBounds(self, pe) -> None:
        """
        Calculate the bounds of the passed path element relative to the
        already-calculated linear bounds
        """
    def farthestExtreme(self, doY: bool = False):
        """
        Returns the location, defining point, and t value for the
        bound farthest from the linear bounds in the dimension selected
        with doY. If the linear bounds are the curve bounds returns
        0, None, None

        The fourth return value is False if the defining point's location
        is less than the linear bound and True if it is greater
        """
    def intersects(self, other, margin: int = 0):
        """
        Returns True if the bounds of this object are within those of
        the argument
        """

class pathBoundsState:
    """
    Calculates and stores the bounds of a glyphData object (path) and
    the pathElements (splines) that define the boundaries.
    """
    bounds: Incomplete
    extpes: Incomplete
    def __init__(self, pe) -> None:
        """Initialize the bounds with those of a single pathElement"""
    def merge(self, other) -> None:
        """Merge this pathBoundsState object with the bounds of another"""
    def within(self, other):
        """
        Returns True if the bounds of this object are within those of
        the argument
        """

class pathElement:
    """
    Stores the coordinates of a spline (line or curve) and
        hintmask values to add directly before the spline
        Whether the spline is the first or second part of a flex hint
        a boundsState object for the spline
        The position (subpath, offset) of the spline in the glyphData path

        self.s is the first point, self.e is the last.
        If the spline is a curve self.cs is the first control point and
        self.ce is the second.

        When segment_sub is not None it must either be a list or an
        integer. If it is a list then that list must contain pathElements,
        representing the same path (in order). These will be used in place
        of the pathElement during segment generation. Each of these
        substitution pathElements must have segment_sub equal to the
        offset in the parent's segment_sub list.
    """
    assocMatchFactor: int
    tSlop: float
    middleMult: int
    is_line: bool
    is_close: Incomplete
    s: Incomplete
    e: Incomplete
    cs: Incomplete
    ce: Incomplete
    masks: Incomplete
    flex: Incomplete
    bounds: Incomplete
    position: Incomplete
    segment_sub: Incomplete
    def __init__(self, *args, is_close: bool = False, masks=None, flex: bool = False, position=None) -> None: ...
    def getBounds(self):
        """Returns the bounds object for the object, generating it if needed"""
    def clearTempState(self) -> None: ...
    def isLine(self):
        """Returns True if the spline is a line"""
    def isClose(self):
        """Returns True if this pathElement implicitly closes a subpath"""
    def isStart(self):
        """Returns True if this pathElement starts a subpath"""
    def isTiny(self):
        """
        Returns True if the start and end points of the spline are within
        two em-units in both dimensions
        """
    def isShort(self):
        """
        Returns True if the start and end points of the spline are within
        about six em-units
        """
    def convertToLine(self) -> None:
        """
        If the pathElement is not already a line, make it one with the same
        start and end points
        """
    def convertToCurve(self, sRatio: float = 0.333333, eRatio=None, roundCoords: bool = False) -> None:
        """
        If the pathElement is not already a curve, make it one. The control
        points are made colinear to preseve the shape. self.cs will be
        positioned at ratio sRatio from self.s and self.ce will be positioned
        at eRatio away from self.e
        """
    def clearHints(self, doVert: bool = False) -> None:
        """Clear the vertical or horizontal masks, if any"""
    def cubicParameters(self):
        """Returns the fontTools cubic parameters for this pathElement"""
    def getAssocFactor(self, loose: bool = False): ...
    def containsPoint(self, p, factor, returnT: bool = False): ...
    def slopePoint(self, t):
        """
        Returns the point definiing the slope of the pathElement
        (relative to the on-curve point) at t==0 or t==1
        """
    def __deepcopy__(self, memo):
        """Don't deepcopy pathElement objects"""
    @staticmethod
    def stemBytes(masks):
        """Calculate bytes corresponding to a (boolean array) hintmask"""
    def relVals(self):
        """
        Return relative coordinates appropriate for an rLineTo or
        rCurveTo T2 operator
        """
    def T2(self, is_start=None):
        """Returns an array of T2 operators corresponding to the pathElement"""
    def splitAtInflectionsForSegs(self): ...
    def splitAt(self, t): ...
    def atT(self, t): ...
    def fonttoolsSegment(self): ...

class glyphData(BasePen):
    """Stores state corresponding to a T2 CharString"""
    roundCoords: Incomplete
    subpaths: Incomplete
    hstems: Incomplete
    vstems: Incomplete
    startmasks: Incomplete
    cntr: Incomplete
    name: Incomplete
    wdth: Incomplete
    is_hm: Incomplete
    flex_count: int
    lastcp: Incomplete
    nextmasks: Incomplete
    nextflex: Incomplete
    changed: bool
    pathEdited: bool
    boundsMap: Incomplete
    hhs: Incomplete
    def __init__(self, roundCoords, name: str = '') -> None: ...
    def getPosition(self):
        """Returns position (subpath idx, offset) of next spline to be drawn"""
    def nextIsFlex(self) -> None:
        """quasi-pen method noting that next spline starts a flex hint"""
    def hStem(self, data, is_hm) -> None:
        """
        quasi-pen method to pass horizontal stem data (in relative format)
        """
    def vStem(self, data, is_hm) -> None:
        """quasi-pen method passing vertical stem data (in relative format)"""
    def hintmask(self, hhints, vhints) -> None:
        """quasi-pen method passing hintmask data"""
    def cntrmask(self, hhints, vhints) -> None:
        """quasi-pen method passing cntrmask data"""
    def setWidth(self, width) -> None: ...
    def getWidth(self): ...
    def isEmpty(self):
        """Returns true if there are no subpaths"""
    def hasFlex(self):
        """Returns True if at least one curve pair is flex-hinted"""
    def hasHints(self, doVert: bool = False, both: bool = False, either: bool = False):
        """
        Returns True if there are hints of the parameter-specified type(s)
        """
    def syncPositions(self) -> None:
        """
        Reset the pathElement.position tuples if the path has been edited
        """
    def setPathEdited(self) -> None: ...
    def getBounds(self, subpath=None):
        """
        Returns the bounds of the specified subpath, or of the whole
        path if subpath is None
        """
    def T2(self, version: int = 1):
        """Returns an array of T2 operators corresponding to the object"""
    def drawPoints(self, pen, ufoH=None) -> None:
        """
        Calls pointPen commands on pen to draw the glyph, optionally naming
        some points and building a library of hint annotations
        """
    def reorder(self, neworder) -> None:
        """Change the order of subpaths according to neworder"""
    def first(self):
        """Returns the first pathElement of the path"""
    def last(self):
        """Returns the last (implicit close) pathElement of the path"""
    def next(self, c, segSub: bool = False):
        """
        If c == self, returns the first elemeht of the path

        If c is a pathElement, returns the following element of the path
        or None if there is no such element
        """
    def nextForHints(self, c):
        '''
        Like next() but returns the next element in "hint order", with
        implicit close elements coming first in the subpath instead of
        last
        '''
    def inSubpath(self, c, i, skipTiny, closeWrapOK, segSub):
        """Utility function for nextInSubpath and prevInSubpath"""
    def nextInSubpath(self, c, skipTiny: bool = False, closeWrapOK: bool = True, segSub: bool = False):
        """
        Returns the next element in the subpath after c.

        If c is the last element and closeWrapOK is True returns the
        first element of the subpath.

        If c is the last element and closeWrapOK is False returns None
        """
    def prevInSubpath(self, c, skipTiny: bool = False, closeWrapOK: bool = True, segSub: bool = False):
        """
        Returns the previous element in the subpath before c.

        If c is the first element and closeWrapOK is True returns the
        last element of the subpath.

        If c is the first element and closeWrapOK is False returns None
        """
    def nextSlopePoint(self, c):
        """Returns the slope point of the element of the subpath after c"""
    def prevSlopePoint(self, c):
        """Returns the slope point of the element of the subpath before c"""
    class glyphiter:
        """An iterator for a glyphData path"""
        gd: Incomplete
        def __init__(self, gd) -> None: ...
        pos: Incomplete
        def __next__(self): ...
        def __iter__(self): ...
    def __iter__(self): ...
    def checkAssocPoint(self, segs, spe, ope, sp, op, mapEnd, loose, factor=None): ...
    def associatePath(self, orig, loose: bool = False) -> None: ...
    def addNullClose(self, si) -> None: ...
    def getStemMasks(self):
        """Utility function for pen methods"""
    def checkFlex(self, is_curve):
        """Utility function for pen methods"""
    def toStems(self, data):
        """Converts relative T2 charstring stem data to stem object array"""
    def fromStems(self, stems):
        """Converts stem array to relative T2 charstring stem data"""
    def clearFlex(self) -> None:
        """Clears any flex hints"""
    def clearHints(self, doVert: bool = False) -> None:
        """Clears stem hints in specified dimension"""
    def clearTempState(self) -> None: ...

from _typeshed import Incomplete
from collections.abc import Generator
from fontTools.misc.roundTools import otRound as otRound
from fontTools.misc.vector import Vector as _Vector

def calcBounds(array):
    """Calculate the bounding rectangle of a 2D points array.

    Args:
        array: A sequence of 2D tuples.

    Returns
    -------
        A four-item tuple representing the bounding rectangle ``(xMin, yMin, xMax, yMax)``.
    """
def calcIntBounds(array, round=...):
    """Calculate the integer bounding rectangle of a 2D points array.

    Values are rounded to closest integer towards ``+Infinity`` using the
    :func:`fontTools.misc.fixedTools.otRound` function by default, unless
    an optional ``round`` function is passed.

    Args:
        array: A sequence of 2D tuples.
        round: A rounding function of type ``f(x: float) -> int``.

    Returns
    -------
        A four-item tuple of integers representing the bounding rectangle:
        ``(xMin, yMin, xMax, yMax)``.
    """
def updateBounds(bounds, p, min=..., max=...):
    """Add a point to a bounding rectangle.

    Args:
        bounds: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax), or None``.
        p: A 2D tuple representing a point.
        min,max: functions to compute the minimum and maximum.

    Returns
    -------
        The updated bounding rectangle ``(xMin, yMin, xMax, yMax)``.
    """
def pointInRect(p, rect):
    """Test if a point is inside a bounding rectangle.

    Args:
        p: A 2D tuple representing a point.
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns
    -------
        ``True`` if the point is inside the rectangle, ``False`` otherwise.
    """
def pointsInRect(array, rect):
    """Determine which points are inside a bounding rectangle.

    Args:
        array: A sequence of 2D tuples.
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns
    -------
        A list containing the points inside the rectangle.
    """
def vectorLength(vector):
    """Calculate the length of the given vector.

    Args:
        vector: A 2D tuple.

    Returns
    -------
        The Euclidean length of the vector.
    """
def asInt16(array):
    """Round a list of floats to 16-bit signed integers.

    Args:
        array: List of float values.

    Returns
    -------
        A list of rounded integers.
    """
def normRect(rect):
    """Normalize a bounding box rectangle.

    This function "turns the rectangle the right way up", so that the following
    holds::

        xMin <= xMax and yMin <= yMax

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns
    -------
        A normalized bounding rectangle.
    """
def scaleRect(rect, x, y):
    """Scale a bounding box rectangle.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        x: Factor to scale the rectangle along the X axis.
        Y: Factor to scale the rectangle along the Y axis.

    Returns
    -------
        A scaled bounding rectangle.
    """
def offsetRect(rect, dx, dy):
    """Offset a bounding box rectangle.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        dx: Amount to offset the rectangle along the X axis.
        dY: Amount to offset the rectangle along the Y axis.

    Returns
    -------
        An offset bounding rectangle.
    """
def insetRect(rect, dx, dy):
    """Inset a bounding box rectangle on all sides.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        dx: Amount to inset the rectangle along the X axis.
        dY: Amount to inset the rectangle along the Y axis.

    Returns
    -------
        An inset bounding rectangle.
    """
def sectRect(rect1, rect2):
    """Test for rectangle-rectangle intersection.

    Args:
        rect1: First bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.
        rect2: Second bounding rectangle.

    Returns
    -------
        A boolean and a rectangle.
        If the input rectangles intersect, returns ``True`` and the intersecting
        rectangle. Returns ``False`` and ``(0, 0, 0, 0)`` if the input
        rectangles don't intersect.
    """
def unionRect(rect1, rect2):
    """Determine union of bounding rectangles.

    Args:
        rect1: First bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.
        rect2: Second bounding rectangle.

    Returns
    -------
        The smallest rectangle in which both input rectangles are fully
        enclosed.
    """
def rectCenter(rect):
    """Determine rectangle center.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns
    -------
        A 2D tuple representing the point at the center of the rectangle.
    """
def rectArea(rect):
    """Determine rectangle area.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns
    -------
        The area of the rectangle.
    """
def intRect(rect):
    """Round a rectangle to integer values.

    Guarantees that the resulting rectangle is NOT smaller than the original.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns
    -------
        A rounded bounding rectangle.
    """
def quantizeRect(rect, factor: int = 1):
    """
    >>> bounds = (72.3, -218.4, 1201.3, 919.1)
    >>> quantizeRect(bounds)
    (72, -219, 1202, 920)
    >>> quantizeRect(bounds, factor=10)
    (70, -220, 1210, 920)
    >>> quantizeRect(bounds, factor=100)
    (0, -300, 1300, 1000)
    """

class Vector(_Vector):
    def __init__(self, *args, **kwargs) -> None: ...

def pairwise(iterable, reverse: bool = False) -> Generator[Incomplete]:
    """Iterate over current and next items in iterable.

    Args:
        iterable: An iterable
        reverse: If true, iterate in reverse order.

    Returns
    -------
        A iterable yielding two elements per iteration.

    Example:

        >>> tuple(pairwise([]))
        ()
        >>> tuple(pairwise([], reverse=True))
        ()
        >>> tuple(pairwise([0]))
        ((0, 0),)
        >>> tuple(pairwise([0], reverse=True))
        ((0, 0),)
        >>> tuple(pairwise([0, 1]))
        ((0, 1), (1, 0))
        >>> tuple(pairwise([0, 1], reverse=True))
        ((1, 0), (0, 1))
        >>> tuple(pairwise([0, 1, 2]))
        ((0, 1), (1, 2), (2, 0))
        >>> tuple(pairwise([0, 1, 2], reverse=True))
        ((2, 1), (1, 0), (0, 2))
        >>> tuple(pairwise(['a', 'b', 'c', 'd']))
        (('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a'))
        >>> tuple(pairwise(['a', 'b', 'c', 'd'], reverse=True))
        (('d', 'c'), ('c', 'b'), ('b', 'a'), ('a', 'd'))
    """
def _test() -> None:
    """
    >>> import math
    >>> calcBounds([])
    (0, 0, 0, 0)
    >>> calcBounds([(0, 40), (0, 100), (50, 50), (80, 10)])
    (0, 10, 80, 100)
    >>> updateBounds((0, 0, 0, 0), (100, 100))
    (0, 0, 100, 100)
    >>> pointInRect((50, 50), (0, 0, 100, 100))
    True
    >>> pointInRect((0, 0), (0, 0, 100, 100))
    True
    >>> pointInRect((100, 100), (0, 0, 100, 100))
    True
    >>> not pointInRect((101, 100), (0, 0, 100, 100))
    True
    >>> list(pointsInRect([(50, 50), (0, 0), (100, 100), (101, 100)], (0, 0, 100, 100)))
    [True, True, True, False]
    >>> vectorLength((3, 4))
    5.0
    >>> vectorLength((1, 1)) == math.sqrt(2)
    True
    >>> list(asInt16([0, 0.1, 0.5, 0.9]))
    [0, 0, 1, 1]
    >>> normRect((0, 10, 100, 200))
    (0, 10, 100, 200)
    >>> normRect((100, 200, 0, 10))
    (0, 10, 100, 200)
    >>> scaleRect((10, 20, 50, 150), 1.5, 2)
    (15.0, 40, 75.0, 300)
    >>> offsetRect((10, 20, 30, 40), 5, 6)
    (15, 26, 35, 46)
    >>> insetRect((10, 20, 50, 60), 5, 10)
    (15, 30, 45, 50)
    >>> insetRect((10, 20, 50, 60), -5, -10)
    (5, 10, 55, 70)
    >>> intersects, rect = sectRect((0, 10, 20, 30), (0, 40, 20, 50))
    >>> not intersects
    True
    >>> intersects, rect = sectRect((0, 10, 20, 30), (5, 20, 35, 50))
    >>> intersects
    1
    >>> rect
    (5, 20, 20, 30)
    >>> unionRect((0, 10, 20, 30), (0, 40, 20, 50))
    (0, 10, 20, 50)
    >>> rectCenter((0, 0, 100, 200))
    (50.0, 100.0)
    >>> rectCenter((0, 0, 100, 199.0))
    (50.0, 99.5)
    >>> intRect((0.9, 2.9, 3.1, 4.1))
    (0, 2, 4, 5)
    """

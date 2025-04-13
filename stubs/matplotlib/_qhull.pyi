import numpy
from typing import Any, overload

def delaunay(x: numpy.ndarray[numpy.float64], y: numpy.ndarray[numpy.float64], verbose: int) -> tuple:
    """delaunay(x: numpy.ndarray[numpy.float64], y: numpy.ndarray[numpy.float64], verbose: int) -> tuple

    --

    Compute a Delaunay triangulation.

    Parameters
    ----------
    x, y : 1d arrays
        The coordinates of the point set, which must consist of at least
        three unique points.
    verbose : int
        Python's verbosity level.

    Returns
    -------
    triangles, neighbors : int arrays, shape (ntri, 3)
        Indices of triangle vertices and indices of triangle neighbors.

    """
@overload
def version() -> str:
    """version() -> str

    version()
    --

    Return the qhull version string.
    """
@overload
def version() -> Any:
    """version() -> str

    version()
    --

    Return the qhull version string.
    """

from _typeshed import Incomplete
from matplotlib import _api as _api, _docstring as _docstring
from matplotlib.collections import PolyCollection as PolyCollection, TriMesh as TriMesh
from matplotlib.tri._triangulation import Triangulation as Triangulation

def tripcolor(ax, *args, alpha: float = 1.0, norm: Incomplete | None = None, cmap: Incomplete | None = None, vmin: Incomplete | None = None, vmax: Incomplete | None = None, shading: str = 'flat', facecolors: Incomplete | None = None, **kwargs):
    """
    Create a pseudocolor plot of an unstructured triangular grid.

    Call signatures::

      tripcolor(triangulation, c, *, ...)
      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)

    The triangular grid can be specified either by passing a `.Triangulation`
    object as the first parameter, or by passing the points *x*, *y* and
    optionally the *triangles* and a *mask*. See `.Triangulation` for an
    explanation of these parameters.

    It is possible to pass the triangles positionally, i.e.
    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
    For more clarity, pass *triangles* via keyword argument.

    If neither of *triangulation* or *triangles* are given, the triangulation
    is calculated on the fly. In this case, it does not make sense to provide
    colors at the triangle faces via *c* or *facecolors* because there are
    multiple possible triangulations for a group of points and you don't know
    which triangles will be constructed.

    Parameters
    ----------
    triangulation : `.Triangulation`
        An already created triangular grid.
    x, y, triangles, mask
        Parameters defining the triangular grid. See `.Triangulation`.
        This is mutually exclusive with specifying *triangulation*.
    c : array-like
        The color values, either for the points or for the triangles. Which one
        is automatically inferred from the length of *c*, i.e. does it match
        the number of points or the number of triangles. If there are the same
        number of points and triangles in the triangulation it is assumed that
        color values are defined at points; to force the use of color values at
        triangles use the keyword argument ``facecolors=c`` instead of just
        ``c``.
        This parameter is position-only.
    facecolors : array-like, optional
        Can be used alternatively to *c* to specify colors at the triangle
        faces. This parameter takes precedence over *c*.
    shading : {'flat', 'gouraud'}, default: 'flat'
        If  'flat' and the color values *c* are defined at points, the color
        values used for each triangle are from the mean c of the triangle's
        three points. If *shading* is 'gouraud' then color values must be
        defined at points.
    %(cmap_doc)s

    %(norm_doc)s

    %(vmin_vmax_doc)s

    %(colorizer_doc)s

    Returns
    -------
    `~matplotlib.collections.PolyCollection` or `~matplotlib.collections.TriMesh`
        The result depends on *shading*: For ``shading='flat'`` the result is a
        `.PolyCollection`, for ``shading='gouraud'`` the result is a `.TriMesh`.

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.collections.Collection` properties

        %(Collection:kwdoc)s
    """

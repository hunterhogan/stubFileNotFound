from _typeshed import Incomplete
from matplotlib import _api as _api
from matplotlib.tri._triangulation import Triangulation as Triangulation

class TriRefiner:
    """
    Abstract base class for classes implementing mesh refinement.

    A TriRefiner encapsulates a Triangulation object and provides tools for
    mesh refinement and interpolation.

    Derived classes must implement:

    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
      the optional keyword arguments *kwargs* are defined in each
      TriRefiner concrete implementation, and which returns:

      - a refined triangulation,
      - optionally (depending on *return_tri_index*), for each
        point of the refined triangulation: the index of
        the initial triangulation triangle to which it belongs.

    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:

      - *z* array of field values (to refine) defined at the base
        triangulation nodes,
      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
      - the other optional keyword arguments *kwargs* are defined in
        each TriRefiner concrete implementation;

      and which returns (as a tuple) a refined triangular mesh and the
      interpolated values of the field at the refined triangulation nodes.
    """
    _triangulation: Incomplete
    def __init__(self, triangulation) -> None: ...

class UniformTriRefiner(TriRefiner):
    """
    Uniform mesh refinement by recursive subdivisions.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The encapsulated triangulation (to be refined)
    """
    def __init__(self, triangulation) -> None: ...
    def refine_triangulation(self, return_tri_index: bool = False, subdiv: int = 3):
        """
        Compute a uniformly refined triangulation *refi_triangulation* of
        the encapsulated :attr:`triangulation`.

        This function refines the encapsulated triangulation by splitting each
        father triangle into 4 child sub-triangles built on the edges midside
        nodes, recursing *subdiv* times.  In the end, each triangle is hence
        divided into ``4**subdiv`` child triangles.

        Parameters
        ----------
        return_tri_index : bool, default: False
            Whether an index table indicating the father triangle index of each
            point is returned.
        subdiv : int, default: 3
            Recursion level for the subdivision.
            Each triangle is divided into ``4**subdiv`` child triangles;
            hence, the default results in 64 refined subtriangles for each
            triangle of the initial triangulation.

        Returns
        -------
        refi_triangulation : `~matplotlib.tri.Triangulation`
            The refined triangulation.
        found_index : int array
            Index of the initial triangulation containing triangle, for each
            point of *refi_triangulation*.
            Returned only if *return_tri_index* is set to True.
        """
    def refine_field(self, z, triinterpolator: Incomplete | None = None, subdiv: int = 3):
        """
        Refine a field defined on the encapsulated triangulation.

        Parameters
        ----------
        z : (npoints,) array-like
            Values of the field to refine, defined at the nodes of the
            encapsulated triangulation. (``n_points`` is the number of points
            in the initial triangulation)
        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
            Interpolator used for field interpolation. If not specified,
            a `~matplotlib.tri.CubicTriInterpolator` will be used.
        subdiv : int, default: 3
            Recursion level for the subdivision.
            Each triangle is divided into ``4**subdiv`` child triangles.

        Returns
        -------
        refi_tri : `~matplotlib.tri.Triangulation`
             The returned refined triangulation.
        refi_z : 1D array of length: *refi_tri* node count.
             The returned interpolated field (at *refi_tri* nodes).
        """
    @staticmethod
    def _refine_triangulation_once(triangulation, ancestors: Incomplete | None = None):
        """
        Refine a `.Triangulation` by splitting each triangle into 4
        child-masked_triangles built on the edges midside nodes.

        Masked triangles, if present, are also split, but their children
        returned masked.

        If *ancestors* is not provided, returns only a new triangulation:
        child_triangulation.

        If the array-like key table *ancestor* is given, it shall be of shape
        (ntri,) where ntri is the number of *triangulation* masked_triangles.
        In this case, the function returns
        (child_triangulation, child_ancestors)
        child_ancestors is defined so that the 4 child masked_triangles share
        the same index as their father: child_ancestors.shape = (4 * ntri,).
        """

import numpy

class TrapezoidMapTriFinder:
    def __init__(self, triangulation: Triangulation) -> None:
        """__init__(self: matplotlib._tri.TrapezoidMapTriFinder, triangulation: matplotlib._tri.Triangulation) -> None

        Create a new C++ TrapezoidMapTriFinder object.
        This should not be called directly, use the python class
        matplotlib.tri.TrapezoidMapTriFinder instead.

        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def find_many(self, arg0: numpy.ndarray[numpy.float64], arg1: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.int32]:
        """find_many(self: matplotlib._tri.TrapezoidMapTriFinder, arg0: numpy.ndarray[numpy.float64], arg1: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.int32]

        Find indices of triangles containing the point coordinates (x, y).
        """
    def get_tree_stats(self) -> list:
        """get_tree_stats(self: matplotlib._tri.TrapezoidMapTriFinder) -> list

        Return statistics about the tree used by the trapezoid map.
        """
    def initialize(self) -> None:
        """initialize(self: matplotlib._tri.TrapezoidMapTriFinder) -> None

        Initialize this object, creating the trapezoid map from the triangulation.
        """
    def print_tree(self) -> None:
        """print_tree(self: matplotlib._tri.TrapezoidMapTriFinder) -> None

        Print the search tree as text to stdout; useful for debug purposes.
        """

class TriContourGenerator:
    def __init__(self, triangulation: Triangulation, z: numpy.ndarray[numpy.float64]) -> None:
        """__init__(self: matplotlib._tri.TriContourGenerator, triangulation: matplotlib._tri.Triangulation, z: numpy.ndarray[numpy.float64]) -> None

        Create a new C++ TriContourGenerator object.
        This should not be called directly, use the functions
        matplotlib.axes.tricontour and tricontourf instead.

        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def create_contour(self, arg0: float) -> tuple:
        """create_contour(self: matplotlib._tri.TriContourGenerator, arg0: float) -> tuple

        Create and return a non-filled contour.
        """
    def create_filled_contour(self, arg0: float, arg1: float) -> tuple:
        """create_filled_contour(self: matplotlib._tri.TriContourGenerator, arg0: float, arg1: float) -> tuple

        Create and return a filled contour.
        """

class Triangulation:
    def __init__(self, x: numpy.ndarray[numpy.float64], y: numpy.ndarray[numpy.float64], triangles: numpy.ndarray[numpy.int32], mask: numpy.ndarray[bool], edges: numpy.ndarray[numpy.int32], neighbors: numpy.ndarray[numpy.int32], correct_triangle_orientations: bool) -> None:
        """__init__(self: matplotlib._tri.Triangulation, x: numpy.ndarray[numpy.float64], y: numpy.ndarray[numpy.float64], triangles: numpy.ndarray[numpy.int32], mask: numpy.ndarray[bool], edges: numpy.ndarray[numpy.int32], neighbors: numpy.ndarray[numpy.int32], correct_triangle_orientations: bool) -> None

        Create a new C++ Triangulation object.
        This should not be called directly, use the python class
        matplotlib.tri.Triangulation instead.

        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def calculate_plane_coefficients(self, arg0: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]:
        """calculate_plane_coefficients(self: matplotlib._tri.Triangulation, arg0: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]

        Calculate plane equation coefficients for all unmasked triangles.
        """
    def get_edges(self) -> numpy.ndarray[numpy.int32]:
        """get_edges(self: matplotlib._tri.Triangulation) -> numpy.ndarray[numpy.int32]

        Return edges array.
        """
    def get_neighbors(self) -> numpy.ndarray[numpy.int32]:
        """get_neighbors(self: matplotlib._tri.Triangulation) -> numpy.ndarray[numpy.int32]

        Return neighbors array.
        """
    def set_mask(self, arg0: numpy.ndarray[bool]) -> None:
        """set_mask(self: matplotlib._tri.Triangulation, arg0: numpy.ndarray[bool]) -> None

        Set or clear the mask array.
        """

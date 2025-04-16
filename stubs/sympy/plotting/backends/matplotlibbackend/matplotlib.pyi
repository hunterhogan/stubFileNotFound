import sympy.plotting.backends.base_backend as base_backend
from _typeshed import Incomplete
from sympy.core.basic import Basic as Basic
from sympy.external import import_module as import_module
from sympy.printing.latex import latex as latex

def _str_or_latex(label): ...
def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib ``fill`` command from a list of bounding
    rectangular intervals
    """

class MatplotlibBackend(base_backend.Plot):
    """ This class implements the functionalities to use Matplotlib with SymPy
    plotting functions.
    """
    matplotlib: Incomplete
    plt: Incomplete
    cm: Incomplete
    LineCollection: Incomplete
    aspect: Incomplete
    _plotgrid_fig: Incomplete
    _plotgrid_ax: Incomplete
    def __init__(self, *series, **kwargs) -> None: ...
    fig: Incomplete
    ax: Incomplete
    def _create_figure(self) -> None: ...
    @staticmethod
    def get_segments(x, y, z: Incomplete | None = None):
        """ Convert two list of coordinates to a list of segments to be used
        with Matplotlib's :external:class:`~matplotlib.collections.LineCollection`.

        Parameters
        ==========
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates

            z : list
                List of z-coordinates for a 3D line.
        """
    def _process_series(self, series, ax) -> None: ...
    def process_series(self) -> None:
        """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
    def show(self) -> None: ...
    def save(self, path) -> None: ...
    def close(self) -> None: ...

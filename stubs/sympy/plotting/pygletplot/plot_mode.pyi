from .plot_interval import PlotInterval as PlotInterval
from .plot_object import PlotObject as PlotObject
from .util import parse_option_string as parse_option_string
from _typeshed import Incomplete
from sympy.core.symbol import Symbol as Symbol
from sympy.core.sympify import sympify as sympify
from sympy.geometry.entity import GeometryEntity as GeometryEntity
from sympy.utilities.iterables import is_sequence as is_sequence

class PlotMode(PlotObject):
    """
    Grandparent class for plotting
    modes. Serves as interface for
    registration, lookup, and init
    of modes.

    To create a new plot mode,
    inherit from PlotModeBase
    or one of its children, such
    as PlotSurface or PlotCurve.
    """
    i_vars: Incomplete
    d_vars: Incomplete
    intervals: Incomplete
    aliases: Incomplete
    is_default: bool
    def draw(self) -> None: ...
    _mode_alias_list: Incomplete
    _mode_map: Incomplete
    _mode_default_map: Incomplete
    _i_var_max: Incomplete
    _d_var_max: Incomplete
    def __new__(cls, *args, **kwargs):
        """
        This is the function which interprets
        arguments given to Plot.__init__ and
        Plot.__setattr__. Returns an initialized
        instance of the appropriate child class.
        """
    @staticmethod
    def _get_mode(mode_arg, i_var_count, d_var_count):
        """
        Tries to return an appropriate mode class.
        Intended to be called only by __new__.

        mode_arg
            Can be a string or a class. If it is a
            PlotMode subclass, it is simply returned.
            If it is a string, it can an alias for
            a mode or an empty string. In the latter
            case, we try to find a default mode for
            the i_var_count and d_var_count.

        i_var_count
            The number of independent variables
            needed to evaluate the d_vars.

        d_var_count
            The number of dependent variables;
            usually the number of functions to
            be evaluated in plotting.

        For example, a Cartesian function y = f(x) has
        one i_var (x) and one d_var (y). A parametric
        form x,y,z = f(u,v), f(u,v), f(u,v) has two
        two i_vars (u,v) and three d_vars (x,y,z).
        """
    @staticmethod
    def _get_default_mode(i, d, i_vars: int = -1): ...
    @staticmethod
    def _get_aliased_mode(alias, i, d, i_vars: int = -1): ...
    @classmethod
    def _register(cls) -> None:
        """
        Called once for each user-usable plot mode.
        For Cartesian2D, it is invoked after the
        class definition: Cartesian2D._register()
        """
    @classmethod
    def _init_mode(cls):
        """
        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        """
    _was_initialized: bool
    @staticmethod
    def _find_i_vars(functions, intervals): ...
    def _fill_i_vars(self, i_vars) -> None: ...
    def _fill_intervals(self, intervals) -> None: ...
    @staticmethod
    def _interpret_args(args): ...
    @staticmethod
    def _extract_options(args, kwargs): ...

def var_count_error(is_independent, is_plotting):
    """
    Used to format an error message which differs
    slightly in 4 places.
    """

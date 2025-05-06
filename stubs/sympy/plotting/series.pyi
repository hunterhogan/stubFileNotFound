from .intervalmath import interval as interval
from _typeshed import Incomplete
from sympy.calculus.util import continuous_domain as continuous_domain
from sympy.concrete import Product as Product, Sum as Sum
from sympy.core.containers import Tuple as Tuple
from sympy.core.expr import Expr as Expr
from sympy.core.function import arity as arity
from sympy.core.relational import Equality as Equality, GreaterThan as GreaterThan, LessThan as LessThan, Ne as Ne, Relational as Relational
from sympy.core.sorting import default_sort_key as default_sort_key
from sympy.core.symbol import Symbol as Symbol
from sympy.core.sympify import sympify as sympify
from sympy.external import import_module as import_module
from sympy.functions import atan2 as atan2, ceiling as ceiling, floor as floor, frac as frac, im as im, zeta as zeta
from sympy.logic.boolalg import BooleanFunction as BooleanFunction
from sympy.plotting.utils import _get_free_symbols as _get_free_symbols, extract_solution as extract_solution
from sympy.printing.latex import latex as latex
from sympy.printing.precedence import precedence as precedence
from sympy.printing.pycode import PythonCodePrinter as PythonCodePrinter
from sympy.sets.sets import Interval as Interval, Set as Set, Union as Union
from sympy.simplify.simplify import nsimplify as nsimplify
from sympy.utilities.exceptions import sympy_deprecation_warning as sympy_deprecation_warning
from sympy.utilities.lambdify import lambdify as lambdify

class IntervalMathPrinter(PythonCodePrinter):
    """A printer to be used inside `plot_implicit` when `adaptive=True`,
    in which case the interval arithmetic module is going to be used, which
    requires the following edits.
    """
    def _print_And(self, expr): ...
    def _print_Or(self, expr): ...

def _uniform_eval(f1, f2, *args, modules: Incomplete | None = None, force_real_eval: bool = False, has_sum: bool = False):
    """
    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.
    """
def _adaptive_eval(f, x):
    """Evaluate f(x) with an adaptive algorithm. Post-process the result.
    If a symbolic expression is evaluated with SymPy, it might returns
    another symbolic expression, containing additions, ...
    Force evaluation to a float.

    Parameters
    ==========
    f : callable
    x : float
    """
def _get_wrapper_for_expr(ret): ...

class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Notes
    =====

    The backend should check if it supports the data series that is given.
    (e.g. TextBackend supports only LineOver1DRangeSeries).
    It is the backend responsibility to know how to use the class of
    data series that is given.

    Some data series classes are grouped (using a class attribute like is_2Dline)
    according to the api they present (based only on convention). The backend is
    not obliged to use that api (e.g. LineOver1DRangeSeries belongs to the
    is_2Dline group and presents the get_points method, but the
    TextBackend does not use the get_points method).

    BaseSeries
    """
    is_2Dline: bool
    is_3Dline: bool
    is_3Dsurface: bool
    is_contour: bool
    is_implicit: bool
    is_interactive: bool
    is_parametric: bool
    is_generic: bool
    is_vector: bool
    is_2Dvector: bool
    is_3Dvector: bool
    _N: int
    only_integers: Incomplete
    modules: Incomplete
    show_in_legend: Incomplete
    colorbar: Incomplete
    use_cm: Incomplete
    is_polar: Incomplete
    is_point: Incomplete
    _label: str
    _ranges: Incomplete
    _n: Incomplete
    _scales: Incomplete
    _params: Incomplete
    _tx: Incomplete
    _ty: Incomplete
    _tz: Incomplete
    _tp: Incomplete
    _functions: Incomplete
    _signature: Incomplete
    _force_real_eval: Incomplete
    _discretized_domain: Incomplete
    _interactive_ranges: bool
    _needs_to_be_int: Incomplete
    color_func: Incomplete
    _eval_color_func_with_signature: bool
    def __init__(self, *args, **kwargs) -> None: ...
    def _block_lambda_functions(self, *exprs) -> None:
        """Some data series can be used to plot numerical functions, others
        cannot. Execute this method inside the `__init__` to prevent the
        processing of numerical functions.
        """
    def _check_fs(self) -> None:
        """ Checks if there are enogh parameters and free symbols.
        """
    def _create_lambda_func(self):
        """Create the lambda functions to be used by the uniform meshing
        strategy.

        Notes
        =====
        The old sympy.plotting used experimental_lambdify. It created one
        lambda function each time an evaluation was requested. If that failed,
        it went on to create a different lambda function and evaluated it,
        and so on.

        This new module changes strategy: it creates right away the default
        lambda function as well as the backup one. The reason is that the
        series could be interactive, hence the numerical function will be
        evaluated multiple times. So, let's create the functions just once.

        This approach works fine for the majority of cases, in which the
        symbolic expression is relatively short, hence the lambdification
        is fast. If the expression is very long, this approach takes twice
        the time to create the lambda functions. Be aware of that!
        """
    def _update_range_value(self, t):
        """If the value of a plotting range is a symbolic expression,
        substitute the parameters in order to get a numerical value.
        """
    def _create_discretized_domain(self) -> None:
        """Discretize the ranges for uniform meshing strategy.
        """
    def _create_discretized_domain_helper(self, discr_symbols, discretizations) -> None:
        """Create 2D or 3D discretized grids.

        Subclasses should override this method in order to implement a
        different behaviour.
        """
    def _evaluate(self, cast_to_real: bool = True):
        """Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """
    def _aggregate_args(self):
        """Create a list of arguments to be passed to the lambda function,
        sorted accoring to self._signature.
        """
    @property
    def expr(self):
        """Return the expression (or expressions) of the series."""
    _expr: Incomplete
    @expr.setter
    def expr(self, e) -> None:
        """Set the expression (or expressions) of the series."""
    @property
    def is_3D(self): ...
    @property
    def is_line(self): ...
    def _line_surface_color(self, prop, val) -> None:
        """This method enables back-compatibility with old sympy.plotting"""
    @property
    def line_color(self): ...
    @line_color.setter
    def line_color(self, val) -> None: ...
    @property
    def n(self):
        """Returns a list [n1, n2, n3] of numbers of discratization points.
        """
    @n.setter
    def n(self, v) -> None:
        """Set the numbers of discretization points. ``v`` must be an int or
        a list.

        Let ``s`` be a series. Then:

        * to set the number of discretization points along the x direction (or
          first parameter): ``s.n = 10``
        * to set the number of discretization points along the x and y
          directions (or first and second parameters): ``s.n = [10, 15]``
        * to set the number of discretization points along the x, y and z
          directions: ``s.n = [10, 15, 20]``

        The following is highly unreccomended, because it prevents
        the execution of necessary code in order to keep updated data:
        ``s.n[1] = 15``
        """
    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
    @params.setter
    def params(self, p) -> None: ...
    adaptive: bool
    def _post_init(self) -> None: ...
    @property
    def scales(self): ...
    @scales.setter
    def scales(self, v) -> None: ...
    @property
    def surface_color(self): ...
    @surface_color.setter
    def surface_color(self, val) -> None: ...
    @property
    def rendering_kw(self): ...
    _rendering_kw: Incomplete
    @rendering_kw.setter
    def rendering_kw(self, kwargs) -> None: ...
    @staticmethod
    def _discretize(start, end, N, scale: str = 'linear', only_integers: bool = False):
        """Discretize a 1D domain.

        Returns
        =======

        domain : np.ndarray with dtype=float or complex
            The domain's dtype will be float or complex (depending on the
            type of start/end) even if only_integers=True. It is left for
            the downstream code to perform further casting, if necessary.
        """
    @staticmethod
    def _correct_shape(a, b):
        """Convert ``a`` to a np.ndarray of the same shape of ``b``.

        Parameters
        ==========

        a : int, float, complex, np.ndarray
            Usually, this is the result of a numerical evaluation of a
            symbolic expression. Even if a discretized domain was used to
            evaluate the function, the result can be a scalar (int, float,
            complex). Think for example to ``expr = Float(2)`` and
            ``f = lambdify(x, expr)``. No matter the shape of the numerical
            array representing x, the result of the evaluation will be
            a single value.

        b : np.ndarray
            It represents the correct shape that ``a`` should have.

        Returns
        =======
        new_a : np.ndarray
            An array with the correct shape.
        """
    def eval_color_func(self, *args):
        """Evaluate the color function.

        Parameters
        ==========

        args : tuple
            Arguments to be passed to the coloring function. Can be coordinates
            or parameters or both.

        Notes
        =====

        The backend will request the data series to generate the numerical
        data. Depending on the data series, either the data series itself or
        the backend will eventually execute this function to generate the
        appropriate coloring value.
        """
    def get_data(self) -> None:
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
    def _get_wrapped_label(self, label, wrapper):
        '''Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        '''
    def get_label(self, use_latex: bool = False, wrapper: str = '$%s$'):
        '''Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        '''
    @property
    def label(self): ...
    @label.setter
    def label(self, val) -> None:
        """Set the labels associated to this series."""
    @property
    def ranges(self): ...
    @ranges.setter
    def ranges(self, val) -> None: ...
    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
    def _str_helper(self, s): ...

def _detect_poles_numerical_helper(x, y, eps: float = 0.01, expr: Incomplete | None = None, symb: Incomplete | None = None, symbolic: bool = False):
    """Compute the steepness of each segment. If it's greater than a
    threshold, set the right-point y-value non NaN and record the
    corresponding x-location for further processing.

    Returns
    =======
    x : np.ndarray
        Unchanged x-data.
    yy : np.ndarray
        Modified y-data with NaN values.
    """
def _detect_poles_symbolic_helper(expr, symb, start, end):
    """Attempts to compute symbolic discontinuities.

    Returns
    =======
    pole : list
        List of symbolic poles, possibily empty.
    """

class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """
    is_2Dline: bool
    _dim: int
    _N: int
    steps: Incomplete
    is_point: Incomplete
    is_filled: Incomplete
    adaptive: Incomplete
    depth: Incomplete
    use_cm: Incomplete
    color_func: Incomplete
    line_color: Incomplete
    detect_poles: Incomplete
    eps: Incomplete
    is_polar: Incomplete
    unwrap: Incomplete
    poles_locations: Incomplete
    exclude: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def get_data(self):
        """Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates

        y: np.ndarray
            y-coordinates

        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries,
            Parametric3DLineInteractiveSeries

        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries (and their
            corresponding interactive series).
        """
    def get_segments(self): ...
    def _insert_exclusions(self, points):
        """Add NaN to each of the exclusion point. Practically, this adds a
        NaN to the exlusion point, plus two other nearby points evaluated with
        the numerical functions associated to this data series.
        These nearby points are important when the number of discretization
        points is low, or the scale is logarithm.

        NOTE: it would be easier to just add exclusion points to the
        discretized domain before evaluation, then after evaluation add NaN
        to the exclusion points. But that's only work with adaptive=False.
        The following approach work even with adaptive=True.
        """
    @property
    def var(self): ...
    @property
    def start(self): ...
    @property
    def end(self): ...
    @property
    def xscale(self): ...
    scales: Incomplete
    @xscale.setter
    def xscale(self, v) -> None: ...
    def get_color_array(self): ...

class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""
    list_x: Incomplete
    list_y: Incomplete
    _expr: Incomplete
    is_polar: Incomplete
    label: Incomplete
    rendering_kw: Incomplete
    is_parametric: bool
    def __init__(self, list_x, list_y, label: str = '', **kwargs) -> None: ...
    def __str__(self) -> str: ...
    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
    def _eval_color_func_and_return(self, *data): ...

class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""
    expr: Incomplete
    _label: Incomplete
    _latex_label: Incomplete
    ranges: Incomplete
    _cast: Incomplete
    _return: Incomplete
    adaptive: bool
    def __init__(self, expr, var_start_end, label: str = '', **kwargs) -> None: ...
    @property
    def nb_of_points(self): ...
    n: Incomplete
    @nb_of_points.setter
    def nb_of_points(self, v) -> None: ...
    def __str__(self) -> str: ...
    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates
        """
    def _adaptive_sampling(self): ...
    def _adaptive_sampling_helper(self, f):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.
        """
    def _uniform_sampling(self): ...
    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        """

class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric: bool
    _label: Incomplete
    _latex_label: Incomplete
    def _set_parametric_line_label(self, label) -> None:
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
    def get_label(self, use_latex: bool = False, wrapper: str = '$%s$'): ...
    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
    def get_parameter_points(self): ...
    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates
            y : list
                List of y-coordinates
            z : list
                List of z-coordinates, only for 3D parametric line plot.
        """
    @property
    def nb_of_points(self): ...
    n: Incomplete
    @nb_of_points.setter
    def nb_of_points(self, v) -> None: ...

class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric SymPy expressions
    over a range."""
    is_2Dline: bool
    expr_x: Incomplete
    expr_y: Incomplete
    expr: Incomplete
    ranges: Incomplete
    _cast: Incomplete
    use_cm: Incomplete
    def __init__(self, expr_x, expr_y, var_start_end, label: str = '', **kwargs) -> None: ...
    def __str__(self) -> str: ...
    def _adaptive_sampling(self): ...
    def _adaptive_sampling_helper(self, f_x, f_y):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
            Luiz Henrique de Figueiredo.
        """

class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""
    is_2Dline: bool
    is_3Dline: bool
    _dim: int
    def __init__(self) -> None: ...

class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""
    is_2Dline: bool
    is_3Dline: bool
    expr_x: Incomplete
    expr_y: Incomplete
    expr_z: Incomplete
    expr: Incomplete
    ranges: Incomplete
    _cast: Incomplete
    adaptive: bool
    use_cm: Incomplete
    _xlim: Incomplete
    _ylim: Incomplete
    _zlim: Incomplete
    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label: str = '', **kwargs) -> None: ...
    def __str__(self) -> str: ...
    def get_data(self): ...

class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""
    is_3Dsurface: bool
    use_cm: Incomplete
    is_polar: Incomplete
    surface_color: Incomplete
    color_func: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    _label: Incomplete
    _latex_label: Incomplete
    def _set_surface_label(self, label) -> None: ...
    def get_color_array(self): ...

class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""
    expr: Incomplete
    ranges: Incomplete
    _xlim: Incomplete
    _ylim: Incomplete
    def __init__(self, expr, var_start_end_x, var_start_end_y, label: str = '', **kwargs) -> None: ...
    @property
    def var_x(self): ...
    @property
    def var_y(self): ...
    @property
    def start_x(self): ...
    @property
    def end_x(self): ...
    @property
    def start_y(self): ...
    @property
    def end_y(self): ...
    @property
    def nb_of_points_x(self): ...
    n: Incomplete
    @nb_of_points_x.setter
    def nb_of_points_x(self, v) -> None: ...
    @property
    def nb_of_points_y(self): ...
    @nb_of_points_y.setter
    def nb_of_points_y(self, v) -> None: ...
    def __str__(self) -> str: ...
    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
    _zlim: Incomplete
    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray
            Discretized x-domain.
        mesh_y : np.ndarray
            Discretized y-domain.
        mesh_z : np.ndarray
            Results of the evaluation.
        """

class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric SymPy
    expressions and a range."""
    is_parametric: bool
    expr_x: Incomplete
    expr_y: Incomplete
    expr_z: Incomplete
    expr: Incomplete
    ranges: Incomplete
    color_func: Incomplete
    def __init__(self, expr_x, expr_y, expr_z, var_start_end_u, var_start_end_v, label: str = '', **kwargs) -> None: ...
    @property
    def var_u(self): ...
    @property
    def var_v(self): ...
    @property
    def start_u(self): ...
    @property
    def end_u(self): ...
    @property
    def start_v(self): ...
    @property
    def end_v(self): ...
    @property
    def nb_of_points_u(self): ...
    n: Incomplete
    @nb_of_points_u.setter
    def nb_of_points_u(self, v) -> None: ...
    @property
    def nb_of_points_v(self): ...
    @nb_of_points_v.setter
    def nb_of_points_v(self, v) -> None: ...
    def __str__(self) -> str: ...
    def get_parameter_meshes(self): ...
    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
    _xlim: Incomplete
    _ylim: Incomplete
    _zlim: Incomplete
    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x : np.ndarray [n2 x n1]
            x-coordinates.
        y : np.ndarray [n2 x n1]
            y-coordinates.
        z : np.ndarray [n2 x n1]
            z-coordinates.
        mesh_u : np.ndarray [n2 x n1]
            Discretized u range.
        mesh_v : np.ndarray [n2 x n1]
            Discretized v range.
        """

class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""
    is_3Dsurface: bool
    is_contour: bool
    is_filled: Incomplete
    show_clabels: Incomplete
    rendering_kw: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class GenericDataSeries(BaseSeries):
    '''Represents generic numerical data.

    Notes
    =====
    This class serves the purpose of back-compatibility with the "markers,
    annotations, fill, rectangles" keyword arguments that represent
    user-provided numerical data. In particular, it solves the problem of
    combining together two or more plot-objects with the ``extend`` or
    ``append`` methods: user-provided numerical data is also taken into
    consideration because it is stored in this series class.

    Also note that the current implementation is far from optimal, as each
    keyword argument is stored into an attribute in the ``Plot`` class, which
    requires a hard-coded if-statement in the ``MatplotlibBackend`` class.
    The implementation suggests that it is ok to add attributes and
    if-statements to provide more and more functionalities for user-provided
    numerical data (e.g. adding horizontal lines, or vertical lines, or bar
    plots, etc). However, in doing so one would reinvent the wheel: plotting
    libraries (like Matplotlib) already implements the necessary API.

    Instead of adding more keyword arguments and attributes, users interested
    in adding custom numerical data to a plot should retrieve the figure
    created by this plotting module. For example, this code:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import Symbol, plot, cos
       x = Symbol("x")
       p = plot(cos(x), markers=[{"args": [[0, 1, 2], [0, 1, -1], "*"]}])

    Becomes:

    .. plot::
       :context: close-figs
       :include-source: True

       p = plot(cos(x), backend="matplotlib")
       fig, ax = p._backend.fig, p._backend.ax[0]
       ax.plot([0, 1, 2], [0, 1, -1], "*")
       fig

    Which is far better in terms of readibility. Also, it gives access to the
    full plotting library capabilities, without the need to reinvent the wheel.
    '''
    is_generic: bool
    type: Incomplete
    args: Incomplete
    rendering_kw: Incomplete
    def __init__(self, tp, *args, **kwargs) -> None: ...
    def get_data(self): ...

class ImplicitSeries(BaseSeries):
    """Representation for 2D Implicit plot."""
    is_implicit: bool
    use_cm: bool
    _N: int
    adaptive: Incomplete
    _label: Incomplete
    _latex_label: Incomplete
    ranges: Incomplete
    _color: Incomplete
    depth: Incomplete
    def __init__(self, expr, var_start_end_x, var_start_end_y, label: str = '', **kwargs) -> None: ...
    @property
    def expr(self): ...
    _adaptive_expr: Incomplete
    has_equality: Incomplete
    _non_adaptive_expr: Incomplete
    _is_equality: bool
    @expr.setter
    def expr(self, expr) -> None: ...
    @property
    def line_color(self): ...
    @line_color.setter
    def line_color(self, v) -> None: ...
    color = line_color
    def _has_equality(self, expr): ...
    def __str__(self) -> str: ...
    def get_data(self):
        '''Returns numerical data.

        Returns
        =======

        If the series is evaluated with the `adaptive=True` it returns:

        interval_list : list
            List of bounding rectangular intervals to be postprocessed and
            eventually used with Matplotlib\'s ``fill`` command.
        dummy : str
            A string containing ``"fill"``.

        Otherwise, it returns 2D numpy arrays to be used with Matplotlib\'s
        ``contour`` or ``contourf`` commands:

        x_array : np.ndarray
        y_array : np.ndarray
        z_array : np.ndarray
        plot_type : str
            A string specifying which plot command to use, ``"contour"``
            or ``"contourf"``.
        '''
    def _adaptive_eval(self):
        """
        References
        ==========

        .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
        Mathematical Formulae with Two Free Variables.

        .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
        Arithmetic. Master's thesis. University of Toronto, 1996
        """
    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
    @staticmethod
    def _preprocess_meshgrid_expression(expr, adaptive):
        """If the expression is a Relational, rewrite it as a single
        expression.

        Returns
        =======

        expr : Expr
            The rewritten expression

        equality : Boolean
            Wheter the original expression was an Equality or not.
        """
    def get_label(self, use_latex: bool = False, wrapper: str = '$%s$'):
        '''Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        '''

def centers_of_segments(array): ...
def centers_of_faces(array): ...
def flat(x, y, z, eps: float = 0.001):
    """Checks whether three points are almost collinear"""
def _set_discretization_points(kwargs, pt):
    """Allow the use of the keyword arguments ``n, n1, n2`` to
    specify the number of discretization points in one and two
    directions, while keeping back-compatibility with older keyword arguments
    like, ``nb_of_points, nb_of_points_*, points``.

    Parameters
    ==========

    kwargs : dict
        Dictionary of keyword arguments passed into a plotting function.
    pt : type
        The type of the series, which indicates the kind of plot we are
        trying to create.
    """

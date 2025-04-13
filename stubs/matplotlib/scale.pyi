from _typeshed import Incomplete
from matplotlib import _api as _api, _docstring as _docstring
from matplotlib.ticker import AsinhLocator as AsinhLocator, AutoLocator as AutoLocator, AutoMinorLocator as AutoMinorLocator, LogFormatterSciNotation as LogFormatterSciNotation, LogLocator as LogLocator, LogitFormatter as LogitFormatter, LogitLocator as LogitLocator, NullFormatter as NullFormatter, NullLocator as NullLocator, ScalarFormatter as ScalarFormatter, SymmetricalLogLocator as SymmetricalLogLocator
from matplotlib.transforms import IdentityTransform as IdentityTransform, Transform as Transform

class ScaleBase:
    '''
    The base class for all scales.

    Scales are separable transformations, working on a single dimension.

    Subclasses should override

    :attr:`name`
        The scale\'s name.
    :meth:`get_transform`
        A method returning a `.Transform`, which converts data coordinates to
        scaled coordinates.  This transform should be invertible, so that e.g.
        mouse positions can be converted back to data coordinates.
    :meth:`set_default_locators_and_formatters`
        A method that sets default locators and formatters for an `~.axis.Axis`
        that uses this scale.
    :meth:`limit_range_for_scale`
        An optional method that "fixes" the axis range to acceptable values,
        e.g. restricting log-scaled axes to positive values.
    '''
    def __init__(self, axis) -> None:
        """
        Construct a new scale.

        Notes
        -----
        The following note is for scale implementers.

        For back-compatibility reasons, scales take an `~matplotlib.axis.Axis`
        object as first argument.  However, this argument should not
        be used: a single scale object should be usable by multiple
        `~matplotlib.axis.Axis`\\es at the same time.
        """
    def get_transform(self) -> None:
        """
        Return the `.Transform` object associated with this scale.
        """
    def set_default_locators_and_formatters(self, axis) -> None:
        """
        Set the locators and formatters of *axis* to instances suitable for
        this scale.
        """
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Return the range *vmin*, *vmax*, restricted to the
        domain supported by this scale (if any).

        *minpos* should be the minimum positive value in the data.
        This is used by log scales to determine a minimum value.
        """

class LinearScale(ScaleBase):
    """
    The default linear scale.
    """
    name: str
    def __init__(self, axis) -> None:
        """
        """
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def get_transform(self):
        """
        Return the transform for linear scaling, which is just the
        `~matplotlib.transforms.IdentityTransform`.
        """

class FuncTransform(Transform):
    """
    A simple transform that takes and arbitrary function for the
    forward and inverse transform.
    """
    input_dims: int
    output_dims: int
    _forward: Incomplete
    _inverse: Incomplete
    def __init__(self, forward, inverse) -> None:
        """
        Parameters
        ----------
        forward : callable
            The forward function for the transform.  This function must have
            an inverse and, for best behavior, be monotonic.
            It must have the signature::

               def forward(values: array-like) -> array-like

        inverse : callable
            The inverse of the forward function.  Signature as ``forward``.
        """
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class FuncScale(ScaleBase):
    """
    Provide an arbitrary scale with user-supplied function for the axis.
    """
    name: str
    _transform: Incomplete
    def __init__(self, axis, functions) -> None:
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

               def forward(values: array-like) -> array-like
        """
    def get_transform(self):
        """Return the `.FuncTransform` associated with this scale."""
    def set_default_locators_and_formatters(self, axis) -> None: ...

class LogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    _clip: Incomplete
    def __init__(self, base, nonpositive: str = 'clip') -> None: ...
    def __str__(self) -> str: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class InvertedLogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    def __init__(self, base) -> None: ...
    def __str__(self) -> str: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class LogScale(ScaleBase):
    """
    A standard logarithmic scale.  Care is taken to only plot positive values.
    """
    name: str
    _transform: Incomplete
    subs: Incomplete
    def __init__(self, axis, *, base: int = 10, subs: Incomplete | None = None, nonpositive: str = 'clip') -> None:
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        base : float, default: 10
            The base of the logarithm.
        nonpositive : {'clip', 'mask'}, default: 'clip'
            Determines the behavior for non-positive values. They can either
            be masked as invalid, or clipped to a very small positive number.
        subs : sequence of int, default: None
            Where to place the subticks between each major tick.  For example,
            in a log10 scale, ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place 8
            logarithmically spaced minor ticks between each major tick.
        """
    base: Incomplete
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def get_transform(self):
        """Return the `.LogTransform` associated with this scale."""
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Limit the domain to positive values."""

class FuncScaleLog(LogScale):
    """
    Provide an arbitrary scale with user-supplied function for the axis and
    then put on a logarithmic axes.
    """
    name: str
    subs: Incomplete
    _transform: Incomplete
    def __init__(self, axis, functions, base: int = 10) -> None:
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

                def forward(values: array-like) -> array-like

        base : float, default: 10
            Logarithmic base of the scale.
        """
    @property
    def base(self): ...
    def get_transform(self):
        """Return the `.Transform` associated with this scale."""

class SymmetricalLogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    linthresh: Incomplete
    linscale: Incomplete
    _linscale_adj: Incomplete
    _log_base: Incomplete
    def __init__(self, base, linthresh, linscale) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class InvertedSymmetricalLogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    linthresh: Incomplete
    invlinthresh: Incomplete
    linscale: Incomplete
    _linscale_adj: Incomplete
    def __init__(self, base, linthresh, linscale) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class SymmetricalLogScale(ScaleBase):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    See :doc:`/gallery/scales/symlog_demo` for a detailed description.

    Parameters
    ----------
    base : float, default: 10
        The base of the logarithm.

    linthresh : float, default: 2
        Defines the range ``(-x, x)``, within which the plot is linear.
        This avoids having the plot go to infinity around zero.

    subs : sequence of int
        Where to place the subticks between each major tick.
        For example, in a log10 scale: ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place
        8 logarithmically spaced minor ticks between each major tick.

    linscale : float, optional
        This allows the linear range ``(-linthresh, linthresh)`` to be
        stretched relative to the logarithmic range. Its value is the number of
        decades to use for each half of the linear range. For example, when
        *linscale* == 1.0 (the default), the space used for the positive and
        negative halves of the linear range will be equal to one decade in
        the logarithmic range.
    """
    name: str
    _transform: Incomplete
    subs: Incomplete
    def __init__(self, axis, *, base: int = 10, linthresh: int = 2, subs: Incomplete | None = None, linscale: int = 1) -> None: ...
    base: Incomplete
    linthresh: Incomplete
    linscale: Incomplete
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def get_transform(self):
        """Return the `.SymmetricalLogTransform` associated with this scale."""

class AsinhTransform(Transform):
    """Inverse hyperbolic-sine transformation used by `.AsinhScale`"""
    input_dims: int
    output_dims: int
    linear_width: Incomplete
    def __init__(self, linear_width) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class InvertedAsinhTransform(Transform):
    """Hyperbolic sine transformation used by `.AsinhScale`"""
    input_dims: int
    output_dims: int
    linear_width: Incomplete
    def __init__(self, linear_width) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class AsinhScale(ScaleBase):
    '''
    A quasi-logarithmic scale based on the inverse hyperbolic sine (asinh)

    For values close to zero, this is essentially a linear scale,
    but for large magnitude values (either positive or negative)
    it is asymptotically logarithmic. The transition between these
    linear and logarithmic regimes is smooth, and has no discontinuities
    in the function gradient in contrast to
    the `.SymmetricalLogScale` ("symlog") scale.

    Specifically, the transformation of an axis coordinate :math:`a` is
    :math:`a \\rightarrow a_0 \\sinh^{-1} (a / a_0)` where :math:`a_0`
    is the effective width of the linear region of the transformation.
    In that region, the transformation is
    :math:`a \\rightarrow a + \\mathcal{O}(a^3)`.
    For large values of :math:`a` the transformation behaves as
    :math:`a \\rightarrow a_0 \\, \\mathrm{sgn}(a) \\ln |a| + \\mathcal{O}(1)`.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.
    '''
    name: str
    auto_tick_multipliers: Incomplete
    _transform: Incomplete
    _base: Incomplete
    _subs: Incomplete
    def __init__(self, axis, *, linear_width: float = 1.0, base: int = 10, subs: str = 'auto', **kwargs) -> None:
        """
        Parameters
        ----------
        linear_width : float, default: 1
            The scale parameter (elsewhere referred to as :math:`a_0`)
            defining the extent of the quasi-linear region,
            and the coordinate values beyond which the transformation
            becomes asymptotically logarithmic.
        base : int, default: 10
            The number base used for rounding tick locations
            on a logarithmic scale. If this is less than one,
            then rounding is to the nearest integer multiple
            of powers of ten.
        subs : sequence of int
            Multiples of the number base used for minor ticks.
            If set to 'auto', this will use built-in defaults,
            e.g. (2, 5) for base=10.
        """
    linear_width: Incomplete
    def get_transform(self): ...
    def set_default_locators_and_formatters(self, axis) -> None: ...

class LogitTransform(Transform):
    input_dims: int
    output_dims: int
    _nonpositive: Incomplete
    _clip: Incomplete
    def __init__(self, nonpositive: str = 'mask') -> None: ...
    def transform_non_affine(self, values):
        """logit transform (base 10), masked or clipped"""
    def inverted(self): ...
    def __str__(self) -> str: ...

class LogisticTransform(Transform):
    input_dims: int
    output_dims: int
    _nonpositive: Incomplete
    def __init__(self, nonpositive: str = 'mask') -> None: ...
    def transform_non_affine(self, values):
        """logistic transform (base 10)"""
    def inverted(self): ...
    def __str__(self) -> str: ...

class LogitScale(ScaleBase):
    """
    Logit scale for data between zero and one, both excluded.

    This scale is similar to a log scale close to zero and to one, and almost
    linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.
    """
    name: str
    _transform: Incomplete
    _use_overline: Incomplete
    _one_half: Incomplete
    def __init__(self, axis, nonpositive: str = 'mask', *, one_half: str = '\\frac{1}{2}', use_overline: bool = False) -> None:
        '''
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            Currently unused.
        nonpositive : {\'mask\', \'clip\'}
            Determines the behavior for values beyond the open interval ]0, 1[.
            They can either be masked as invalid, or clipped to a number very
            close to 0 or 1.
        use_overline : bool, default: False
            Indicate the usage of survival notation (\\overline{x}) in place of
            standard notation (1-x) for probability close to one.
        one_half : str, default: r"\\frac{1}{2}"
            The string used for ticks formatter to represent 1/2.
        '''
    def get_transform(self):
        """Return the `.LogitTransform` associated with this scale."""
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to values between 0 and 1 (excluded).
        """

_scale_mapping: Incomplete

def get_scale_names():
    """Return the names of the available scales."""
def scale_factory(scale, axis, **kwargs):
    """
    Return a scale class by name.

    Parameters
    ----------
    scale : {%(names)s}
    axis : `~matplotlib.axis.Axis`
    """
def register_scale(scale_class) -> None:
    """
    Register a new kind of scale.

    Parameters
    ----------
    scale_class : subclass of `ScaleBase`
        The scale to register.
    """
def _get_scale_docs():
    """
    Helper function for generating docstrings related to scales.
    """

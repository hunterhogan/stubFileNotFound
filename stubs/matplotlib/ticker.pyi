import string
from _typeshed import Incomplete

__all__ = ['TickHelper', 'Formatter', 'FixedFormatter', 'NullFormatter', 'FuncFormatter', 'FormatStrFormatter', 'StrMethodFormatter', 'ScalarFormatter', 'LogFormatter', 'LogFormatterExponent', 'LogFormatterMathtext', 'LogFormatterSciNotation', 'LogitFormatter', 'EngFormatter', 'PercentFormatter', 'Locator', 'IndexLocator', 'FixedLocator', 'NullLocator', 'LinearLocator', 'LogLocator', 'AutoLocator', 'MultipleLocator', 'MaxNLocator', 'AutoMinorLocator', 'SymmetricalLogLocator', 'AsinhLocator', 'LogitLocator']

class _DummyAxis:
    __name__: str
    _data_interval: Incomplete
    _view_interval: Incomplete
    _minpos: Incomplete
    def __init__(self, minpos: int = 0) -> None: ...
    def get_view_interval(self): ...
    def set_view_interval(self, vmin, vmax) -> None: ...
    def get_minpos(self): ...
    def get_data_interval(self): ...
    def set_data_interval(self, vmin, vmax) -> None: ...
    def get_tick_space(self): ...

class TickHelper:
    axis: Incomplete
    def set_axis(self, axis) -> None: ...
    def create_dummy_axis(self, **kwargs) -> None: ...

class Formatter(TickHelper):
    """
    Create a string based on a tick value and location.
    """
    locs: Incomplete
    def __call__(self, x, pos: Incomplete | None = None) -> None:
        """
        Return the format for tick value *x* at position pos.
        ``pos=None`` indicates an unspecified location.
        """
    def format_ticks(self, values):
        """Return the tick labels for all the ticks at once."""
    def format_data(self, value):
        """
        Return the full string representation of the value with the
        position unspecified.
        """
    def format_data_short(self, value):
        """
        Return a short string version of the tick value.

        Defaults to the position-independent long value.
        """
    def get_offset(self): ...
    def set_locs(self, locs) -> None:
        """
        Set the locations of the ticks.

        This method is called before computing the tick labels because some
        formatters need to know all tick locations to do so.
        """
    @staticmethod
    def fix_minus(s):
        """
        Some classes may want to replace a hyphen for minus with the proper
        Unicode symbol (U+2212) for typographical correctness.  This is a
        helper method to perform such a replacement when it is enabled via
        :rc:`axes.unicode_minus`.
        """
    def _set_locator(self, locator) -> None:
        """Subclasses may want to override this to set a locator."""

class NullFormatter(Formatter):
    """Always return the empty string."""
    def __call__(self, x, pos: Incomplete | None = None): ...

class FixedFormatter(Formatter):
    """
    Return fixed strings for tick labels based only on position, not value.

    .. note::
        `.FixedFormatter` should only be used together with `.FixedLocator`.
        Otherwise, the labels may end up in unexpected positions.
    """
    seq: Incomplete
    offset_string: str
    def __init__(self, seq) -> None:
        """Set the sequence *seq* of strings that will be used for labels."""
    def __call__(self, x, pos: Incomplete | None = None):
        """
        Return the label that matches the position, regardless of the value.

        For positions ``pos < len(seq)``, return ``seq[i]`` regardless of
        *x*. Otherwise return empty string. ``seq`` is the sequence of
        strings that this object was initialized with.
        """
    def get_offset(self): ...
    def set_offset_string(self, ofs) -> None: ...

class FuncFormatter(Formatter):
    """
    Use a user-defined function for formatting.

    The function should take in two inputs (a tick value ``x`` and a
    position ``pos``), and return a string containing the corresponding
    tick label.
    """
    func: Incomplete
    offset_string: str
    def __init__(self, func) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None):
        """
        Return the value of the user defined function.

        *x* and *pos* are passed through as-is.
        """
    def get_offset(self): ...
    def set_offset_string(self, ofs) -> None: ...

class FormatStrFormatter(Formatter):
    '''
    Use an old-style (\'%\' operator) format string to format the tick.

    The format string should have a single variable format (%) in it.
    It will be applied to the value (not the position) of the tick.

    Negative numeric values (e.g., -1) will use a dash, not a Unicode minus;
    use mathtext to get a Unicode minus by wrapping the format specifier with $
    (e.g. "$%g$").
    '''
    fmt: Incomplete
    def __init__(self, fmt) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None):
        """
        Return the formatted label string.

        Only the value *x* is formatted. The position is ignored.
        """

class _UnicodeMinusFormat(string.Formatter):
    """
    A specialized string formatter so that `.StrMethodFormatter` respects
    :rc:`axes.unicode_minus`.  This implementation relies on the fact that the
    format string is only ever called with kwargs *x* and *pos*, so it blindly
    replaces dashes by unicode minuses without further checking.
    """
    def format_field(self, value, format_spec): ...

class StrMethodFormatter(Formatter):
    """
    Use a new-style format string (as used by `str.format`) to format the tick.

    The field used for the tick value must be labeled *x* and the field used
    for the tick position must be labeled *pos*.

    The formatter will respect :rc:`axes.unicode_minus` when formatting
    negative numeric values.

    It is typically unnecessary to explicitly construct `.StrMethodFormatter`
    objects, as `~.Axis.set_major_formatter` directly accepts the format string
    itself.
    """
    fmt: Incomplete
    def __init__(self, fmt) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None):
        """
        Return the formatted label string.

        *x* and *pos* are passed to `str.format` as keyword arguments
        with those exact names.
        """

class ScalarFormatter(Formatter):
    """
    Format tick values as a number.

    Parameters
    ----------
    useOffset : bool or float, default: :rc:`axes.formatter.useoffset`
        Whether to use offset notation. See `.set_useOffset`.
    useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
        Whether to use fancy math formatting. See `.set_useMathText`.
    useLocale : bool, default: :rc:`axes.formatter.use_locale`.
        Whether to use locale settings for decimal sign and positive sign.
        See `.set_useLocale`.
    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the
        numbers in the formatter.

        .. versionadded:: 3.10

    Notes
    -----
    In addition to the parameters above, the formatting of scientific vs.
    floating point representation can be configured via `.set_scientific`
    and `.set_powerlimits`).

    **Offset notation and scientific notation**

    Offset notation and scientific notation look quite similar at first sight.
    Both split some information from the formatted tick values and display it
    at the end of the axis.

    - The scientific notation splits up the order of magnitude, i.e. a
      multiplicative scaling factor, e.g. ``1e6``.

    - The offset notation separates an additive constant, e.g. ``+1e6``. The
      offset notation label is always prefixed with a ``+`` or ``-`` sign
      and is thus distinguishable from the order of magnitude label.

    The following plot with x limits ``1_000_000`` to ``1_000_010`` illustrates
    the different formatting. Note the labels at the right edge of the x axis.

    .. plot::

        lim = (1_000_000, 1_000_010)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'hspace': 2})
        ax1.set(title='offset notation', xlim=lim)
        ax2.set(title='scientific notation', xlim=lim)
        ax2.xaxis.get_major_formatter().set_useOffset(False)
        ax3.set(title='floating-point notation', xlim=lim)
        ax3.xaxis.get_major_formatter().set_useOffset(False)
        ax3.xaxis.get_major_formatter().set_scientific(False)

    """
    _offset_threshold: Incomplete
    orderOfMagnitude: int
    format: str
    _scientific: bool
    _powerlimits: Incomplete
    def __init__(self, useOffset: Incomplete | None = None, useMathText: Incomplete | None = None, useLocale: Incomplete | None = None, *, usetex: Incomplete | None = None) -> None: ...
    def get_usetex(self):
        """Return whether TeX's math mode is enabled for rendering."""
    _usetex: Incomplete
    def set_usetex(self, val) -> None:
        """Set whether to use TeX's math mode for rendering numbers in the formatter."""
    usetex: Incomplete
    def get_useOffset(self):
        """
        Return whether automatic mode for offset notation is active.

        This returns True if ``set_useOffset(True)``; it returns False if an
        explicit offset was set, e.g. ``set_useOffset(1000)``.

        See Also
        --------
        ScalarFormatter.set_useOffset
        """
    offset: int
    _useOffset: Incomplete
    def set_useOffset(self, val) -> None:
        """
        Set whether to use offset notation.

        When formatting a set numbers whose value is large compared to their
        range, the formatter can separate an additive constant. This can
        shorten the formatted numbers so that they are less likely to overlap
        when drawn on an axis.

        Parameters
        ----------
        val : bool or float
            - If False, do not use offset notation.
            - If True (=automatic mode), use offset notation if it can make
              the residual numbers significantly shorter. The exact behavior
              is controlled by :rc:`axes.formatter.offset_threshold`.
            - If a number, force an offset of the given value.

        Examples
        --------
        With active offset notation, the values

        ``100_000, 100_002, 100_004, 100_006, 100_008``

        will be formatted as ``0, 2, 4, 6, 8`` plus an offset ``+1e5``, which
        is written to the edge of the axis.
        """
    useOffset: Incomplete
    def get_useLocale(self):
        """
        Return whether locale settings are used for formatting.

        See Also
        --------
        ScalarFormatter.set_useLocale
        """
    _useLocale: Incomplete
    def set_useLocale(self, val) -> None:
        """
        Set whether to use locale settings for decimal sign and positive sign.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_locale`.
        """
    useLocale: Incomplete
    def _format_maybe_minus_and_locale(self, fmt, arg):
        """
        Format *arg* with *fmt*, applying Unicode minus and locale if desired.
        """
    def get_useMathText(self):
        """
        Return whether to use fancy math formatting.

        See Also
        --------
        ScalarFormatter.set_useMathText
        """
    _useMathText: Incomplete
    def set_useMathText(self, val) -> None:
        """
        Set whether to use fancy math formatting.

        If active, scientific notation is formatted as :math:`1.2 \\times 10^3`.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_mathtext`.
        """
    useMathText: Incomplete
    def __call__(self, x, pos: Incomplete | None = None):
        """
        Return the format for tick value *x* at position *pos*.
        """
    def set_scientific(self, b) -> None:
        """
        Turn scientific notation on or off.

        See Also
        --------
        ScalarFormatter.set_powerlimits
        """
    def set_powerlimits(self, lims) -> None:
        """
        Set size thresholds for scientific notation.

        Parameters
        ----------
        lims : (int, int)
            A tuple *(min_exp, max_exp)* containing the powers of 10 that
            determine the switchover threshold. For a number representable as
            :math:`a \\times 10^\\mathrm{exp}` with :math:`1 <= |a| < 10`,
            scientific notation will be used if ``exp <= min_exp`` or
            ``exp >= max_exp``.

            The default limits are controlled by :rc:`axes.formatter.limits`.

            In particular numbers with *exp* equal to the thresholds are
            written in scientific notation.

            Typically, *min_exp* will be negative and *max_exp* will be
            positive.

            For example, ``formatter.set_powerlimits((-3, 4))`` will provide
            the following formatting:
            :math:`1 \\times 10^{-3}, 9.9 \\times 10^{-3}, 0.01,`
            :math:`9999, 1 \\times 10^4`.

        See Also
        --------
        ScalarFormatter.set_scientific
        """
    def format_data_short(self, value): ...
    def format_data(self, value): ...
    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
    locs: Incomplete
    def set_locs(self, locs) -> None: ...
    def _compute_offset(self) -> None: ...
    def _set_order_of_magnitude(self) -> None: ...
    def _set_format(self) -> None: ...

class LogFormatter(Formatter):
    '''
    Base class for formatting ticks on a log or symlog scale.

    It may be instantiated directly, or subclassed.

    Parameters
    ----------
    base : float, default: 10.
        Base of the logarithm used in all calculations.

    labelOnlyBase : bool, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.

    minor_thresholds : (subset, all), default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it \'numdec\'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.

    linthresh : None or float, default: None
        If a symmetric log scale is in use, its ``linthresh``
        parameter must be supplied here.

    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.

    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.

    To disable labeling of minor ticks when \'labelOnlyBase\' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.

    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.

    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    '''
    minor_thresholds: Incomplete
    _sublabels: Incomplete
    _linthresh: Incomplete
    def __init__(self, base: float = 10.0, labelOnlyBase: bool = False, minor_thresholds: Incomplete | None = None, linthresh: Incomplete | None = None) -> None: ...
    _base: Incomplete
    def set_base(self, base) -> None:
        """
        Change the *base* for labeling.

        .. warning::
           Should always match the base used for :class:`LogLocator`
        """
    labelOnlyBase: Incomplete
    def set_label_minor(self, labelOnlyBase) -> None:
        """
        Switch minor tick labeling on or off.

        Parameters
        ----------
        labelOnlyBase : bool
            If True, label ticks only at integer powers of base.
        """
    def set_locs(self, locs: Incomplete | None = None) -> None:
        """
        Use axis view limits to control which ticks are labeled.

        The *locs* parameter is ignored in the present algorithm.
        """
    def _num_to_string(self, x, vmin, vmax): ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_data(self, value): ...
    def format_data_short(self, value): ...
    def _pprint_val(self, x, d): ...

class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """
    def _num_to_string(self, x, vmin, vmax): ...

class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """
    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""
    def __call__(self, x, pos: Incomplete | None = None): ...

class LogFormatterSciNotation(LogFormatterMathtext):
    """
    Format values following scientific notation in a logarithmic axis.
    """
    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""

class LogitFormatter(Formatter):
    """
    Probability formatter (using Math text).
    """
    _use_overline: Incomplete
    _one_half: Incomplete
    _minor: Incomplete
    _labelled: Incomplete
    _minor_threshold: Incomplete
    _minor_number: Incomplete
    def __init__(self, *, use_overline: bool = False, one_half: str = '\\frac{1}{2}', minor: bool = False, minor_threshold: int = 25, minor_number: int = 6) -> None:
        '''
        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1 - v, indicate if x should be displayed as
            $\\overline{v}$. The default is to display $1 - v$.

        one_half : str, default: r"\\\\frac{1}{2}"
            The string used to represent 1/2.

        minor : bool, default: False
            Indicate if the formatter is formatting minor ticks or not.
            Basically minor ticks are not labelled, except when only few ticks
            are provided, ticks with most space with neighbor ticks are
            labelled. See other parameters to change the default behavior.

        minor_threshold : int, default: 25
            Maximum number of locs for labelling some minor ticks. This
            parameter have no effect if minor is False.

        minor_number : int, default: 6
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        '''
    def use_overline(self, use_overline) -> None:
        """
        Switch display mode with overline for labelling p>1/2.

        Parameters
        ----------
        use_overline : bool
            If x > 1/2, with x = 1 - v, indicate if x should be displayed as
            $\\overline{v}$. The default is to display $1 - v$.
        """
    def set_one_half(self, one_half) -> None:
        """
        Set the way one half is displayed.

        one_half : str
            The string used to represent 1/2.
        """
    def set_minor_threshold(self, minor_threshold) -> None:
        """
        Set the threshold for labelling minors ticks.

        Parameters
        ----------
        minor_threshold : int
            Maximum number of locations for labelling some minor ticks. This
            parameter have no effect if minor is False.
        """
    def set_minor_number(self, minor_number) -> None:
        """
        Set the number of minor ticks to label when some minor ticks are
        labelled.

        Parameters
        ----------
        minor_number : int
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
    locs: Incomplete
    def set_locs(self, locs): ...
    def _format_value(self, x, locs, sci_notation: bool = True): ...
    def _one_minus(self, s): ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_data_short(self, value): ...

class EngFormatter(ScalarFormatter):
    """
    Format axis values using engineering prefixes to represent powers
    of 1000, plus a specified unit, e.g., 10 MHz instead of 1e7.
    """
    ENG_PREFIXES: Incomplete
    unit: Incomplete
    places: Incomplete
    sep: Incomplete
    def __init__(self, unit: str = '', places: Incomplete | None = None, sep: str = ' ', *, usetex: Incomplete | None = None, useMathText: Incomplete | None = None, useOffset: bool = False) -> None:
        '''
        Parameters
        ----------
        unit : str, default: ""
            Unit symbol to use, suitable for use with single-letter
            representations of powers of 1000. For example, \'Hz\' or \'m\'.

        places : int, default: None
            Precision with which to display the number, specified in
            digits after the decimal point (there will be between one
            and three digits before the decimal point). If it is None,
            the formatting falls back to the floating point format \'%g\',
            which displays up to 6 *significant* digits, i.e. the equivalent
            value for *places* varies between 0 and 5 (inclusive).

        sep : str, default: " "
            Separator used between the value and the prefix/unit. For
            example, one get \'3.14 mV\' if ``sep`` is " " (default) and
            \'3.14mV\' if ``sep`` is "". Besides the default behavior, some
            other useful options may be:

            * ``sep=""`` to append directly the prefix/unit to the value;
            * ``sep="\\N{THIN SPACE}"`` (``U+2009``);
            * ``sep="\\N{NARROW NO-BREAK SPACE}"`` (``U+202F``);
            * ``sep="\\N{NO-BREAK SPACE}"`` (``U+00A0``).

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX\'s math mode for rendering the
            numbers in the formatter.

        useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
            To enable/disable the use mathtext for rendering the numbers in
            the formatter.
        useOffset : bool or float, default: False
            Whether to use offset notation with :math:`10^{3*N}` based prefixes.
            This features allows showing an offset with standard SI order of
            magnitude prefix near the axis. Offset is computed similarly to
            how `ScalarFormatter` computes it internally, but here you are
            guaranteed to get an offset which will make the tick labels exceed
            3 digits. See also `.set_useOffset`.

            .. versionadded:: 3.10
        '''
    def __call__(self, x, pos: Incomplete | None = None):
        """
        Return the format for tick value *x* at position *pos*.

        If there is no currently offset in the data, it returns the best
        engineering formatting that fits the given argument, independently.
        """
    locs: Incomplete
    offset: Incomplete
    orderOfMagnitude: Incomplete
    def set_locs(self, locs) -> None: ...
    def get_offset(self): ...
    def format_eng(self, num):
        """Alias to EngFormatter.format_data"""
    def format_data(self, value):
        """
        Format a number in engineering notation, appending a letter
        representing the power of 1000 of the original number.
        Some examples:

        >>> format_data(0)        # for self.places = 0
        '0'

        >>> format_data(1000000)  # for self.places = 1
        '1.0 M'

        >>> format_data(-1e-6)  # for self.places = 2
        '-1.00 µ'
        """

class PercentFormatter(Formatter):
    """
    Format numbers as a percentage.

    Parameters
    ----------
    xmax : float
        Determines how the number is converted into a percentage.
        *xmax* is the data value that corresponds to 100%.
        Percentages are computed as ``x / xmax * 100``. So if the data is
        already scaled to be percentages, *xmax* will be 100. Another common
        situation is where *xmax* is 1.0.

    decimals : None or int
        The number of decimal places to place after the point.
        If *None* (the default), the number will be computed automatically.

    symbol : str or None
        A string that will be appended to the label. It may be
        *None* or empty to indicate that no symbol should be used. LaTeX
        special characters are escaped in *symbol* whenever latex mode is
        enabled, unless *is_latex* is *True*.

    is_latex : bool
        If *False*, reserved LaTeX characters in *symbol* will be escaped.
    """
    xmax: Incomplete
    decimals: Incomplete
    _symbol: Incomplete
    _is_latex: Incomplete
    def __init__(self, xmax: int = 100, decimals: Incomplete | None = None, symbol: str = '%', is_latex: bool = False) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None):
        """Format the tick as a percentage with the appropriate scaling."""
    def format_pct(self, x, display_range):
        """
        Format the number as a percentage number with the correct
        number of decimals and adds the percent symbol, if any.

        If ``self.decimals`` is `None`, the number of digits after the
        decimal point is set based on the *display_range* of the axis
        as follows:

        ============= ======== =======================
        display_range decimals sample
        ============= ======== =======================
        >50           0        ``x = 34.5`` => 35%
        >5            1        ``x = 34.5`` => 34.5%
        >0.5          2        ``x = 34.5`` => 34.50%
        ...           ...      ...
        ============= ======== =======================

        This method will not be very good for tiny axis ranges or
        extremely large ones. It assumes that the values on the chart
        are percentages displayed on a reasonable scale.
        """
    def convert_to_pct(self, x): ...
    @property
    def symbol(self):
        """
        The configured percent symbol as a string.

        If LaTeX is enabled via :rc:`text.usetex`, the special characters
        ``{'#', '$', '%', '&', '~', '_', '^', '\\', '{', '}'}`` are
        automatically escaped in the string.
        """
    @symbol.setter
    def symbol(self, symbol) -> None: ...

class Locator(TickHelper):
    """
    Determine tick locations.

    Note that the same locator should not be used across multiple
    `~matplotlib.axis.Axis` because the locator stores references to the Axis
    data and view limits.
    """
    MAXTICKS: int
    def tick_values(self, vmin, vmax) -> None:
        """
        Return the values of the located ticks given **vmin** and **vmax**.

        .. note::
            To get tick locations with the vmin and vmax values defined
            automatically for the associated ``axis`` simply call
            the Locator instance::

                >>> print(type(loc))
                <type 'Locator'>
                >>> print(loc())
                [1, 2, 3, 4]

        """
    def set_params(self, **kwargs) -> None:
        """
        Do nothing, and raise a warning. Any locator class not supporting the
        set_params() function will call this.
        """
    def __call__(self) -> None:
        """Return the locations of the ticks."""
    def raise_if_exceeds(self, locs):
        '''
        Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.

        This is intended to be called immediately before returning *locs* from
        ``__call__`` to inform users in case their Locator returns a huge
        number of ticks, causing Matplotlib to run out of memory.

        The "strange" name of this method dates back to when it would raise an
        exception instead of emitting a log.
        '''
    def nonsingular(self, v0, v1):
        """
        Adjust a range as needed to avoid singularities.

        This method gets called during autoscaling, with ``(v0, v1)`` set to
        the data limits on the Axes if the Axes contains any data, or
        ``(-inf, +inf)`` if not.

        - If ``v0 == v1`` (possibly up to some floating point slop), this
          method returns an expanded interval around this value.
        - If ``(v0, v1) == (-inf, +inf)``, this method returns appropriate
          default view limits.
        - Otherwise, ``(v0, v1)`` is returned without modification.
        """
    def view_limits(self, vmin, vmax):
        """
        Select a scale for the range from vmin to vmax.

        Subclasses should override this method to change locator behaviour.
        """

class IndexLocator(Locator):
    """
    Place ticks at every nth point plotted.

    IndexLocator assumes index plotting; i.e., that the ticks are placed at integer
    values in the range between 0 and len(data) inclusive.
    """
    _base: Incomplete
    offset: Incomplete
    def __init__(self, base, offset) -> None:
        """Place ticks every *base* data point, starting at *offset*."""
    def set_params(self, base: Incomplete | None = None, offset: Incomplete | None = None) -> None:
        """Set parameters within this locator"""
    def __call__(self):
        """Return the locations of the ticks"""
    def tick_values(self, vmin, vmax): ...

class FixedLocator(Locator):
    """
    Place ticks at a set of fixed values.

    If *nbins* is None ticks are placed at all values. Otherwise, the *locs* array of
    possible positions will be subsampled to keep the number of ticks
    :math:`\\leq nbins + 1`. The subsampling will be done to include the smallest
    absolute value; for example, if zero is included in the array of possibilities, then
    it will be included in the chosen ticks.
    """
    locs: Incomplete
    nbins: Incomplete
    def __init__(self, locs, nbins: Incomplete | None = None) -> None: ...
    def set_params(self, nbins: Incomplete | None = None) -> None:
        """Set parameters within this locator."""
    def __call__(self): ...
    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """

class NullLocator(Locator):
    """
    No ticks
    """
    def __call__(self): ...
    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are Null, vmin and vmax are not used in this
            method.
        """

class LinearLocator(Locator):
    """
    Place ticks at evenly spaced values.

    The first time this function is called it will try to set the
    number of ticks to make a nice tick partitioning.  Thereafter, the
    number of ticks will be fixed so that interactive navigation will
    be nice

    """
    presets: Incomplete
    def __init__(self, numticks: Incomplete | None = None, presets: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        numticks : int or None, default None
            Number of ticks. If None, *numticks* = 11.
        presets : dict or None, default: None
            Dictionary mapping ``(vmin, vmax)`` to an array of locations.
            Overrides *numticks* if there is an entry for the current
            ``(vmin, vmax)``.
        """
    @property
    def numticks(self): ...
    _numticks: Incomplete
    @numticks.setter
    def numticks(self, numticks) -> None: ...
    def set_params(self, numticks: Incomplete | None = None, presets: Incomplete | None = None) -> None:
        """Set parameters within this locator."""
    def __call__(self):
        """Return the locations of the ticks."""
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""

class MultipleLocator(Locator):
    """
    Place ticks at every integer multiple of a base plus an offset.
    """
    _edge: Incomplete
    _offset: Incomplete
    def __init__(self, base: float = 1.0, offset: float = 0.0) -> None:
        """
        Parameters
        ----------
        base : float > 0, default: 1.0
            Interval between ticks.
        offset : float, default: 0.0
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
    def set_params(self, base: Incomplete | None = None, offset: Incomplete | None = None) -> None:
        """
        Set parameters within this locator.

        Parameters
        ----------
        base : float > 0, optional
            Interval between ticks.
        offset : float, optional
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
    def __call__(self):
        """Return the locations of the ticks."""
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest tick values that contain the data.
        """

class _Edge_integer:
    """
    Helper for `.MaxNLocator`, `.MultipleLocator`, etc.

    Take floating-point precision limitations into account when calculating
    tick locations as integer multiples of a step.
    """
    step: Incomplete
    _offset: Incomplete
    def __init__(self, step, offset) -> None:
        """
        Parameters
        ----------
        step : float > 0
            Interval between ticks.
        offset : float
            Offset subtracted from the data limits prior to calculating tick
            locations.
        """
    def closeto(self, ms, edge): ...
    def le(self, x):
        """Return the largest n: n*step <= x."""
    def ge(self, x):
        """Return the smallest n: n*step >= x."""

class MaxNLocator(Locator):
    """
    Place evenly spaced ticks, with a cap on the total number of ticks.

    Finds nice tick locations with no more than :math:`nbins + 1` ticks being within the
    view limits. Locations beyond the limits are added to support autoscaling.
    """
    default_params: Incomplete
    def __init__(self, nbins: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        nbins : int or 'auto', default: 10
            Maximum number of intervals; one less than max number of
            ticks.  If the string 'auto', the number of bins will be
            automatically determined based on the length of the axis.

        steps : array-like, optional
            Sequence of acceptable tick multiples, starting with 1 and
            ending with 10. For example, if ``steps=[1, 2, 4, 5, 10]``,
            ``20, 40, 60`` or ``0.4, 0.6, 0.8`` would be possible
            sets of ticks because they are multiples of 2.
            ``30, 60, 90`` would not be generated because 3 does not
            appear in this example list of steps.

        integer : bool, default: False
            If True, ticks will take only integer values, provided at least
            *min_n_ticks* integers are found within the view limits.

        symmetric : bool, default: False
            If True, autoscaling will result in a range symmetric about zero.

        prune : {'lower', 'upper', 'both', None}, default: None
            Remove the 'lower' tick, the 'upper' tick, or ticks on 'both' sides
            *if they fall exactly on an axis' edge* (this typically occurs when
            :rc:`axes.autolimit_mode` is 'round_numbers').  Removing such ticks
            is mostly useful for stacked or ganged plots, where the upper tick
            of an Axes overlaps with the lower tick of the axes above it.

        min_n_ticks : int, default: 2
            Relax *nbins* and *integer* constraints if necessary to obtain
            this minimum number of ticks.
        """
    @staticmethod
    def _validate_steps(steps): ...
    @staticmethod
    def _staircase(steps): ...
    _nbins: Incomplete
    _symmetric: Incomplete
    _prune: Incomplete
    _min_n_ticks: Incomplete
    _steps: Incomplete
    _extended_steps: Incomplete
    _integer: Incomplete
    def set_params(self, **kwargs) -> None:
        """
        Set parameters for this locator.

        Parameters
        ----------
        nbins : int or 'auto', optional
            see `.MaxNLocator`
        steps : array-like, optional
            see `.MaxNLocator`
        integer : bool, optional
            see `.MaxNLocator`
        symmetric : bool, optional
            see `.MaxNLocator`
        prune : {'lower', 'upper', 'both', None}, optional
            see `.MaxNLocator`
        min_n_ticks : int, optional
            see `.MaxNLocator`
        """
    def _raw_ticks(self, vmin, vmax):
        """
        Generate a list of tick locations including the range *vmin* to
        *vmax*.  In some applications, one or both of the end locations
        will not be needed, in which case they are trimmed off
        elsewhere.
        """
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, dmin, dmax): ...

class LogLocator(Locator):
    """
    Place logarithmically spaced ticks.

    Places ticks at the values ``subs[j] * base**i``.
    """
    _base: Incomplete
    numticks: Incomplete
    def __init__(self, base: float = 10.0, subs=(1.0,), *, numticks: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        base : float, default: 10.0
            The base of the log used, so major ticks are placed at ``base**n``, where
            ``n`` is an integer.
        subs : None or {'auto', 'all'} or sequence of float, default: (1.0,)
            Gives the multiples of integer powers of the base at which to place ticks.
            The default of ``(1.0, )`` places ticks only at integer powers of the base.
            Permitted string values are ``'auto'`` and ``'all'``. Both of these use an
            algorithm based on the axis view limits to determine whether and how to put
            ticks between integer powers of the base:
            - ``'auto'``: Ticks are placed only between integer powers.
            - ``'all'``: Ticks are placed between *and* at integer powers.
            - ``None``: Equivalent to ``'auto'``.
        numticks : None or int, default: None
            The maximum number of ticks to allow on a given axis. The default of
            ``None`` will try to choose intelligently as long as this Locator has
            already been assigned to an axis using `~.axis.Axis.get_tick_space`, but
            otherwise falls back to 9.
        """
    def set_params(self, base: Incomplete | None = None, subs: Incomplete | None = None, *, numticks: Incomplete | None = None) -> None:
        """Set parameters within this locator."""
    _subs: str
    def _set_subs(self, subs) -> None:
        """
        Set the minor ticks for the log scaling every ``base**i*subs[j]``.
        """
    def __call__(self):
        """Return the locations of the ticks."""
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
    def nonsingular(self, vmin, vmax): ...

class SymmetricalLogLocator(Locator):
    """
    Place ticks spaced linearly near zero and spaced logarithmically beyond a threshold.
    """
    _base: Incomplete
    _linthresh: Incomplete
    _subs: Incomplete
    numticks: int
    def __init__(self, transform: Incomplete | None = None, subs: Incomplete | None = None, linthresh: Incomplete | None = None, base: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        transform : `~.scale.SymmetricalLogTransform`, optional
            If set, defines the *base* and *linthresh* of the symlog transform.
        base, linthresh : float, optional
            The *base* and *linthresh* of the symlog transform, as documented
            for `.SymmetricalLogScale`.  These parameters are only used if
            *transform* is not set.
        subs : sequence of float, default: [1]
            The multiples of integer powers of the base where ticks are placed,
            i.e., ticks are placed at
            ``[sub * base**i for i in ... for sub in subs]``.

        Notes
        -----
        Either *transform*, or both *base* and *linthresh*, must be given.
        """
    def set_params(self, subs: Incomplete | None = None, numticks: Incomplete | None = None) -> None:
        """Set parameters within this locator."""
    def __call__(self):
        """Return the locations of the ticks."""
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""

class AsinhLocator(Locator):
    """
    Place ticks spaced evenly on an inverse-sinh scale.

    Generally used with the `~.scale.AsinhScale` class.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.
    """
    linear_width: Incomplete
    numticks: Incomplete
    symthresh: Incomplete
    base: Incomplete
    subs: Incomplete
    def __init__(self, linear_width, numticks: int = 11, symthresh: float = 0.2, base: int = 10, subs: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        linear_width : float
            The scale parameter defining the extent
            of the quasi-linear region.
        numticks : int, default: 11
            The approximate number of major ticks that will fit
            along the entire axis
        symthresh : float, default: 0.2
            The fractional threshold beneath which data which covers
            a range that is approximately symmetric about zero
            will have ticks that are exactly symmetric.
        base : int, default: 10
            The number base used for rounding tick locations
            on a logarithmic scale. If this is less than one,
            then rounding is to the nearest integer multiple
            of powers of ten.
        subs : tuple, default: None
            Multiples of the number base, typically used
            for the minor ticks, e.g. (2, 5) when base=10.
        """
    def set_params(self, numticks: Incomplete | None = None, symthresh: Incomplete | None = None, base: Incomplete | None = None, subs: Incomplete | None = None) -> None:
        """Set parameters within this locator."""
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...

class LogitLocator(MaxNLocator):
    """
    Place ticks spaced evenly on a logit scale.
    """
    _minor: Incomplete
    def __init__(self, minor: bool = False, *, nbins: str = 'auto') -> None:
        """
        Parameters
        ----------
        nbins : int or 'auto', optional
            Number of ticks. Only used if minor is False.
        minor : bool, default: False
            Indicate if this locator is for minor ticks or not.
        """
    def set_params(self, minor: Incomplete | None = None, **kwargs) -> None:
        """Set parameters within this locator."""
    @property
    def minor(self): ...
    @minor.setter
    def minor(self, value) -> None: ...
    def tick_values(self, vmin, vmax): ...
    def nonsingular(self, vmin, vmax): ...

class AutoLocator(MaxNLocator):
    """
    Place evenly spaced ticks, with the step size and maximum number of ticks chosen
    automatically.

    This is a subclass of `~matplotlib.ticker.MaxNLocator`, with parameters
    *nbins = 'auto'* and *steps = [1, 2, 2.5, 5, 10]*.
    """
    def __init__(self) -> None:
        """
        To know the values of the non-public parameters, please have a
        look to the defaults of `~matplotlib.ticker.MaxNLocator`.
        """

class AutoMinorLocator(Locator):
    """
    Place evenly spaced minor ticks, with the step size and maximum number of ticks
    chosen automatically.

    The Axis must use a linear scale and have evenly spaced major ticks.
    """
    ndivs: Incomplete
    def __init__(self, n: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        n : int or 'auto', default: :rc:`xtick.minor.ndivs` or :rc:`ytick.minor.ndivs`
            The number of subdivisions of the interval between major ticks;
            e.g., n=2 will place a single minor tick midway between major ticks.

            If *n* is 'auto', it will be set to 4 or 5: if the distance
            between the major ticks equals 1, 2.5, 5 or 10 it can be perfectly
            divided in 5 equidistant sub-intervals with a length multiple of
            0.05; otherwise, it is divided in 4 sub-intervals.
        """
    def __call__(self): ...
    def tick_values(self, vmin, vmax) -> None: ...

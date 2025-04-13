from ._color_data import BASE_COLORS as BASE_COLORS, CSS4_COLORS as CSS4_COLORS, TABLEAU_COLORS as TABLEAU_COLORS, XKCD_COLORS as XKCD_COLORS
from _typeshed import Incomplete
from collections.abc import Mapping
from matplotlib import _api as _api, _cm as _cm, _image as _image, cbook as cbook, scale as scale

class _ColorMapping(dict):
    cache: Incomplete
    def __init__(self, mapping) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...

_colors_full_map: Incomplete
_REPR_PNG_SIZE: Incomplete
_BIVAR_REPR_PNG_SIZE: int

def get_named_colors_mapping():
    """Return the global mapping of names to named colors."""

class ColorSequenceRegistry(Mapping):
    """
    Container for sequences of colors that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.color_sequences`. There
    should be no need for users to instantiate `.ColorSequenceRegistry`
    themselves.

    Read access uses a dict-like interface mapping names to lists of colors::

        import matplotlib as mpl
        colors = mpl.color_sequences['tab10']

    For a list of built in color sequences, see :doc:`/gallery/color/color_sequences`.
    The returned lists are copies, so that their modification does not change
    the global definition of the color sequence.

    Additional color sequences can be added via
    `.ColorSequenceRegistry.register`::

        mpl.color_sequences.register('rgb', ['r', 'g', 'b'])
    """
    _BUILTIN_COLOR_SEQUENCES: Incomplete
    _color_sequences: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, item): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    def register(self, name, color_list) -> None:
        """
        Register a new color sequence.

        The color sequence registry stores a copy of the given *color_list*, so
        that future changes to the original list do not affect the registered
        color sequence. Think of this as the registry taking a snapshot
        of *color_list* at registration.

        Parameters
        ----------
        name : str
            The name for the color sequence.

        color_list : list of :mpltype:`color`
            An iterable returning valid Matplotlib colors when iterating over.
            Note however that the returned color sequence will always be a
            list regardless of the input type.

        """
    def unregister(self, name) -> None:
        """
        Remove a sequence from the registry.

        You cannot remove built-in color sequences.

        If the name is not registered, returns with no error.
        """

_color_sequences: Incomplete

def _sanitize_extrema(ex): ...

_nth_color_re: Incomplete

def _is_nth_color(c):
    """Return whether *c* can be interpreted as an item in the color cycle."""
def is_color_like(c):
    """Return whether *c* can be interpreted as an RGB(A) color."""
def _has_alpha_channel(c):
    """Return whether *c* is a color with an alpha channel."""
def _check_color_like(**kwargs) -> None:
    """
    For each *key, value* pair in *kwargs*, check that *value* is color-like.
    """
def same_color(c1, c2):
    """
    Return whether the colors *c1* and *c2* are the same.

    *c1*, *c2* can be single colors or lists/arrays of colors.
    """
def to_rgba(c, alpha: Incomplete | None = None):
    '''
    Convert *c* to an RGBA color.

    Parameters
    ----------
    c : Matplotlib color or ``np.ma.masked``

    alpha : float, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

    Returns
    -------
    tuple
        Tuple of floats ``(r, g, b, a)``, where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    '''
def _to_rgba_no_colorcycle(c, alpha: Incomplete | None = None):
    '''
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alpha
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    '''
def to_rgba_array(c, alpha: Incomplete | None = None):
    '''
    Convert *c* to a (n, 4) array of RGBA colors.

    Parameters
    ----------
    c : Matplotlib color or array of colors
        If *c* is a masked array, an `~numpy.ndarray` is returned with a
        (0, 0, 0, 0) row for each masked value or row in *c*.

    alpha : float or sequence of floats, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

        If *alpha* is a sequence and *c* is a single color, *c* will be
        repeated to match the length of *alpha*.

    Returns
    -------
    array
        (n, 4) array of RGBA colors,  where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    '''
def to_rgb(c):
    """Convert *c* to an RGB color, silently dropping the alpha channel."""
def to_hex(c, keep_alpha: bool = False):
    """
    Convert *c* to a hex color.

    Parameters
    ----------
    c : :ref:`color <colors_def>` or `numpy.ma.masked`

    keep_alpha : bool, default: False
      If False, use the ``#rrggbb`` format, otherwise use ``#rrggbbaa``.

    Returns
    -------
    str
      ``#rrggbb`` or ``#rrggbbaa`` hex color string
    """
cnames = CSS4_COLORS
hexColorPattern: Incomplete
rgb2hex = to_hex
hex2color = to_rgb

class ColorConverter:
    """
    A class only kept for backwards compatibility.

    Its functionality is entirely provided by module-level functions.
    """
    colors = _colors_full_map
    cache: Incomplete
    to_rgb: Incomplete
    to_rgba: Incomplete
    to_rgba_array: Incomplete

colorConverter: Incomplete

def _create_lookup_table(N, data, gamma: float = 1.0):
    """
    Create an *N* -element 1D lookup table.

    This assumes a mapping :math:`f : [0, 1] \\rightarrow [0, 1]`. The returned
    data is an array of N values :math:`y = f(x)` where x is sampled from
    [0, 1].

    By default (*gamma* = 1) x is equidistantly sampled from [0, 1]. The
    *gamma* correction factor :math:`\\gamma` distorts this equidistant
    sampling by :math:`x \\rightarrow x^\\gamma`.

    Parameters
    ----------
    N : int
        The number of elements of the created lookup table; at least 1.

    data : (M, 3) array-like or callable
        Defines the mapping :math:`f`.

        If a (M, 3) array-like, the rows define values (x, y0, y1).  The x
        values must start with x=0, end with x=1, and all x values be in
        increasing order.

        A value between :math:`x_i` and :math:`x_{i+1}` is mapped to the range
        :math:`y^1_{i-1} \\ldots y^0_i` by linear interpolation.

        For the simple case of a y-continuous mapping, y0 and y1 are identical.

        The two values of y are to allow for discontinuous mapping functions.
        E.g. a sawtooth with a period of 0.2 and an amplitude of 1 would be::

            [(0, 1, 0), (0.2, 1, 0), (0.4, 1, 0), ..., [(1, 1, 0)]

        In the special case of ``N == 1``, by convention the returned value
        is y0 for x == 1.

        If *data* is a callable, it must accept and return numpy arrays::

           data(x : ndarray) -> ndarray

        and map values between 0 - 1 to 0 - 1.

    gamma : float
        Gamma correction factor for input distribution x of the mapping.

        See also https://en.wikipedia.org/wiki/Gamma_correction.

    Returns
    -------
    array
        The lookup table where ``lut[x * (N-1)]`` gives the closest value
        for values of x between 0 and 1.

    Notes
    -----
    This function is internally used for `.LinearSegmentedColormap`.
    """

class Colormap:
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """
    name: Incomplete
    N: Incomplete
    _rgba_bad: Incomplete
    _rgba_under: Incomplete
    _rgba_over: Incomplete
    _i_under: Incomplete
    _i_over: Incomplete
    _i_bad: Incomplete
    _isinit: bool
    n_variates: int
    colorbar_extend: bool
    def __init__(self, name, N: int = 256) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        """
    def __call__(self, X, alpha: Incomplete | None = None, bytes: bool = False):
        """
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
    def _get_rgba_and_mask(self, X, alpha: Incomplete | None = None, bytes: bool = False):
        """
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\\s in the
            interval ``[0, 255]``.

        Returns
        -------
        colors : np.ndarray
            Array of RGBA values with a shape of ``X.shape + (4, )``.
        mask : np.ndarray
            Boolean array with True where the input is ``np.nan`` or masked.
        """
    def __copy__(self): ...
    def __eq__(self, other): ...
    def get_bad(self):
        """Get the color for masked values."""
    def set_bad(self, color: str = 'k', alpha: Incomplete | None = None) -> None:
        """Set the color for masked values."""
    def get_under(self):
        """Get the color for low out-of-range values."""
    def set_under(self, color: str = 'k', alpha: Incomplete | None = None) -> None:
        """Set the color for low out-of-range values."""
    def get_over(self):
        """Get the color for high out-of-range values."""
    def set_over(self, color: str = 'k', alpha: Incomplete | None = None) -> None:
        """Set the color for high out-of-range values."""
    def set_extremes(self, *, bad: Incomplete | None = None, under: Incomplete | None = None, over: Incomplete | None = None) -> None:
        """
        Set the colors for masked (*bad*) values and, when ``norm.clip =
        False``, low (*under*) and high (*over*) out-of-range values.
        """
    def with_extremes(self, *, bad: Incomplete | None = None, under: Incomplete | None = None, over: Incomplete | None = None):
        """
        Return a copy of the colormap, for which the colors for masked (*bad*)
        values and, when ``norm.clip = False``, low (*under*) and high (*over*)
        out-of-range values, have been set accordingly.
        """
    def _set_extremes(self) -> None: ...
    def _init(self) -> None:
        """Generate the lookup table, ``self._lut``."""
    def is_gray(self):
        """Return whether the colormap is grayscale."""
    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
    def reversed(self, name: Incomplete | None = None) -> None:
        '''
        Return a reversed instance of the Colormap.

        .. note:: This function is not implemented for the base class.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        See Also
        --------
        LinearSegmentedColormap.reversed
        ListedColormap.reversed
        '''
    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
    def copy(self):
        """Return a copy of the colormap."""

class LinearSegmentedColormap(Colormap):
    """
    Colormap objects based on lookup tables using linear segments.

    The lookup table is generated using linear interpolation for each
    primary color, with the 0-1 domain divided into any number of
    segments.
    """
    monochrome: bool
    _segmentdata: Incomplete
    _gamma: Incomplete
    def __init__(self, name, segmentdata, N: int = 256, gamma: float = 1.0) -> None:
        """
        Create colormap from linear mapping segments

        segmentdata argument is a dictionary with a red, green and blue
        entries. Each entry should be a list of *x*, *y0*, *y1* tuples,
        forming rows in a table. Entries for alpha are optional.

        Example: suppose you want red to increase from 0 to 1 over
        the bottom half, green to do the same over the middle half,
        and blue over the top half.  Then you would use::

            cdict = {'red':   [(0.0,  0.0, 0.0),
                               (0.5,  1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'green': [(0.0,  0.0, 0.0),
                               (0.25, 0.0, 0.0),
                               (0.75, 1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'blue':  [(0.0,  0.0, 0.0),
                               (0.5,  0.0, 0.0),
                               (1.0,  1.0, 1.0)]}

        Each row in the table for a given color is a sequence of
        *x*, *y0*, *y1* tuples.  In each sequence, *x* must increase
        monotonically from 0 to 1.  For any input value *z* falling
        between *x[i]* and *x[i+1]*, the output value of a given color
        will be linearly interpolated between *y1[i]* and *y0[i+1]*::

            row i:   x  y0  y1
                           /
                          /
            row i+1: x  y0  y1

        Hence y0 in the first row and y1 in the last row are never used.

        See Also
        --------
        LinearSegmentedColormap.from_list
            Static method; factory function for generating a smoothly-varying
            LinearSegmentedColormap.
        """
    _lut: Incomplete
    _isinit: bool
    def _init(self) -> None: ...
    def set_gamma(self, gamma) -> None:
        """Set a new gamma value and regenerate colormap."""
    @staticmethod
    def from_list(name, colors, N: int = 256, gamma: float = 1.0):
        """
        Create a `LinearSegmentedColormap` from a list of colors.

        Parameters
        ----------
        name : str
            The name of the colormap.
        colors : list of :mpltype:`color` or list of (value, color)
            If only colors are given, they are equidistantly mapped from the
            range :math:`[0, 1]`; i.e. 0 maps to ``colors[0]`` and 1 maps to
            ``colors[-1]``.
            If (value, color) pairs are given, the mapping is from *value*
            to *color*. This can be used to divide the range unevenly.
        N : int
            The number of RGB quantization levels.
        gamma : float
        """
    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
    @staticmethod
    def _reverser(func, x): ...
    def reversed(self, name: Incomplete | None = None):
        '''
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        LinearSegmentedColormap
            The reversed colormap.
        '''

class ListedColormap(Colormap):
    """
    Colormap object generated from a list of colors.

    This may be most useful when indexing directly into a colormap,
    but it can also be used to generate special colormaps for ordinary
    mapping.

    Parameters
    ----------
    colors : list, array
        Sequence of Matplotlib color specifications (color names or RGB(A)
        values).
    name : str, optional
        String to identify the colormap.
    N : int, optional
        Number of entries in the map. The default is *None*, in which case
        there is one colormap entry for each element in the list of colors.
        If ::

            N < len(colors)

        the list will be truncated at *N*. If ::

            N > len(colors)

        the list will be extended by repetition.
    """
    monochrome: bool
    colors: Incomplete
    def __init__(self, colors, name: str = 'from_list', N: Incomplete | None = None) -> None: ...
    _lut: Incomplete
    _isinit: bool
    def _init(self) -> None: ...
    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
    def reversed(self, name: Incomplete | None = None):
        '''
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        ListedColormap
            A reversed instance of the colormap.
        '''

class MultivarColormap:
    """
    Class for holding multiple `~matplotlib.colors.Colormap` for use in a
    `~matplotlib.cm.ScalarMappable` object
    """
    name: Incomplete
    _colormaps: Incomplete
    _combination_mode: Incomplete
    n_variates: Incomplete
    _rgba_bad: Incomplete
    def __init__(self, colormaps, combination_mode, name: str = 'multivariate colormap') -> None:
        """
        Parameters
        ----------
        colormaps: list or tuple of `~matplotlib.colors.Colormap` objects
            The individual colormaps that are combined
        combination_mode: str, 'sRGB_add' or 'sRGB_sub'
            Describe how colormaps are combined in sRGB space

            - If 'sRGB_add' -> Mixing produces brighter colors
              `sRGB = sum(colors)`
            - If 'sRGB_sub' -> Mixing produces darker colors
              `sRGB = 1 - sum(1 - colors)`
        name : str, optional
            The name of the colormap family.
        """
    def __call__(self, X, alpha: Incomplete | None = None, bytes: bool = False, clip: bool = True):
        """
        Parameters
        ----------
        X : tuple (X0, X1, ...) of length equal to the number of colormaps
            X0, X1 ...:
            float or int, `~numpy.ndarray` or scalar
            The data value(s) to convert to RGBA.
            For floats, *Xi...* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *Xi...*  should be in the interval ``[0, self[i].N)`` to
            return RGBA values *indexed* from colormap [i] with index ``Xi``, where
            self[i] is colormap i.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching *Xi*, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\\s in the
            interval ``[0, 255]``.
        clip : bool, default: True
            If True, clip output to 0 to 1

        Returns
        -------
        Tuple of RGBA values if X[0] is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
    def copy(self):
        """Return a copy of the multivarcolormap."""
    def __copy__(self): ...
    def __eq__(self, other): ...
    def __getitem__(self, item): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    def get_bad(self):
        """Get the color for masked values."""
    def resampled(self, lutshape):
        """
        Return a new colormap with *lutshape* entries.

        Parameters
        ----------
        lutshape : tuple of (`int`, `None`)
            The tuple must have a length matching the number of variates.
            For each element in the tuple, if `int`, the corresponding colorbar
            is resampled, if `None`, the corresponding colorbar is not resampled.

        Returns
        -------
        MultivarColormap
        """
    def with_extremes(self, *, bad: Incomplete | None = None, under: Incomplete | None = None, over: Incomplete | None = None):
        """
        Return a copy of the `MultivarColormap` with modified out-of-range attributes.

        The *bad* keyword modifies the copied `MultivarColormap` while *under* and
        *over* modifies the attributes of the copied component colormaps.
        Note that *under* and *over* colors are subject to the mixing rules determined
        by the *combination_mode*.

        Parameters
        ----------
        bad: :mpltype:`color`, default: None
            If Matplotlib color, the bad value is set accordingly in the copy

        under tuple of :mpltype:`color`, default: None
            If tuple, the `under` value of each component is set with the values
            from the tuple.

        over tuple of :mpltype:`color`, default: None
            If tuple, the `over` value of each component is set with the values
            from the tuple.

        Returns
        -------
        MultivarColormap
            copy of self with attributes set

        """
    @property
    def combination_mode(self): ...
    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
    def _repr_html_(self):
        """Generate an HTML representation of the MultivarColormap."""

class BivarColormap:
    """
    Base class for all bivariate to RGBA mappings.

    Designed as a drop-in replacement for Colormap when using a 2D
    lookup table. To be used with `~matplotlib.cm.ScalarMappable`.
    """
    name: Incomplete
    N: Incomplete
    M: Incomplete
    _shape: Incomplete
    _rgba_bad: Incomplete
    _rgba_outside: Incomplete
    _isinit: bool
    n_variates: int
    _origin: Incomplete
    def __init__(self, N: int = 256, M: int = 256, shape: str = 'square', origin=(0, 0), name: str = 'bivariate colormap') -> None:
        """
        Parameters
        ----------
        N : int, default: 256
            The number of RGB quantization levels along the first axis.
        M : int, default: 256
            The number of RGB quantization levels along the second axis.
        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - 'square' each variate is clipped to [0,1] independently
            - 'circle' the variates are clipped radially to the center
              of the colormap, and a circular mask is applied when the colormap
              is displayed
            - 'ignore' the variates are not clipped, but instead assigned the
              'outside' color
            - 'circleignore' a circular mask is applied, but the data is not
              clipped and instead assigned the 'outside' color

        origin : (float, float), default: (0,0)
            The relative origin of the colormap. Typically (0, 0), for colormaps
            that are linear on both axis, and (.5, .5) for circular colormaps.
            Used when getting 1D colormaps from 2D colormaps.
        name : str, optional
            The name of the colormap.
        """
    def __call__(self, X, alpha: Incomplete | None = None, bytes: bool = False):
        """
        Parameters
        ----------
        X : tuple (X0, X1), X0 and X1: float or int or array-like
            The data value(s) to convert to RGBA.

            - For floats, *X* should be in the interval ``[0.0, 1.0]`` to
              return the RGBA values ``X*100`` percent along the Colormap.
            - For integers, *X* should be in the interval ``[0, Colormap.N)`` to
              return RGBA values *indexed* from the Colormap with index ``X``.

        alpha : float or array-like or None, default: None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X0, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
    @property
    def lut(self):
        """
        For external access to the lut, i.e. for displaying the cmap.
        For circular colormaps this returns a lut with a circular mask.

        Internal functions (such as to_rgb()) should use _lut
        which stores the lut without a circular mask
        A lut without the circular mask is needed in to_rgb() because the
        conversion from floats to ints results in some some pixel-requests
        just outside of the circular mask

        """
    def __copy__(self): ...
    def __eq__(self, other): ...
    def get_bad(self):
        """Get the color for masked values."""
    def get_outside(self):
        """Get the color for out-of-range values."""
    def resampled(self, lutshape, transposed: bool = False):
        """
        Return a new colormap with *lutshape* entries.

        Note that this function does not move the origin.

        Parameters
        ----------
        lutshape : tuple of ints or None
            The tuple must be of length 2, and each entry is either an int or None.

            - If an int, the corresponding axis is resampled.
            - If negative the corresponding axis is resampled in reverse
            - If -1, the axis is inverted
            - If 1 or None, the corresponding axis is not resampled.

        transposed : bool, default: False
            if True, the axes are swapped after resampling

        Returns
        -------
        BivarColormap
        """
    def reversed(self, axis_0: bool = True, axis_1: bool = True):
        """
        Reverses both or one of the axis.
        """
    def transposed(self):
        """
        Transposes the colormap by swapping the order of the axis
        """
    def with_extremes(self, *, bad: Incomplete | None = None, outside: Incomplete | None = None, shape: Incomplete | None = None, origin: Incomplete | None = None):
        """
        Return a copy of the `BivarColormap` with modified attributes.

        Note that the *outside* color is only relevant if `shape` = 'ignore'
        or 'circleignore'.

        Parameters
        ----------
        bad : None or :mpltype:`color`
            If Matplotlib color, the *bad* value is set accordingly in the copy

        outside : None or :mpltype:`color`
            If Matplotlib color and shape is 'ignore' or 'circleignore', values
            *outside* the colormap are colored accordingly in the copy

        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - If 'square' each variate is clipped to [0,1] independently
            - If 'circle' the variates are clipped radially to the center
              of the colormap, and a circular mask is applied when the colormap
              is displayed
            - If 'ignore' the variates are not clipped, but instead assigned the
              *outside* color
            - If 'circleignore' a circular mask is applied, but the data is not
              clipped and instead assigned the *outside* color

        origin : (float, float)
            The relative origin of the colormap. Typically (0, 0), for colormaps
            that are linear on both axis, and (.5, .5) for circular colormaps.
            Used when getting 1D colormaps from 2D colormaps.

        Returns
        -------
        BivarColormap
            copy of self with attributes set
        """
    def _init(self) -> None:
        """Generate the lookup table, ``self._lut``."""
    @property
    def shape(self): ...
    @property
    def origin(self): ...
    def _clip(self, X) -> None:
        """
        For internal use when applying a BivarColormap to data.
        i.e. cm.ScalarMappable().to_rgba()
        Clips X[0] and X[1] according to 'self.shape'.
        X is modified in-place.

        Parameters
        ----------
        X: np.array
            array of floats or ints to be clipped
        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - If 'square' each variate is clipped to [0,1] independently
            - If 'circle' the variates are clipped radially to the center
              of the colormap.
              It is assumed that a circular mask is applied when the colormap
              is displayed
            - If 'ignore' the variates are not clipped, but instead assigned the
              'outside' color
            - If 'circleignore' a circular mask is applied, but the data is not clipped
              and instead assigned the 'outside' color

        """
    def __getitem__(self, item):
        """Creates and returns a colorbar along the selected axis"""
    def _repr_png_(self):
        """Generate a PNG representation of the BivarColormap."""
    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
    def copy(self):
        """Return a copy of the colormap."""

class SegmentedBivarColormap(BivarColormap):
    """
    BivarColormap object generated by supersampling a regular grid.

    Parameters
    ----------
    patch : np.array
        Patch is required to have a shape (k, l, 3), and will get supersampled
        to a lut of shape (N, N, 4).
    N : int
        The number of RGB quantization levels along each axis.
    shape : {'square', 'circle', 'ignore', 'circleignore'}

        - If 'square' each variate is clipped to [0,1] independently
        - If 'circle' the variates are clipped radially to the center
          of the colormap, and a circular mask is applied when the colormap
          is displayed
        - If 'ignore' the variates are not clipped, but instead assigned the
          'outside' color
        - If 'circleignore' a circular mask is applied, but the data is not clipped

    origin : (float, float)
        The relative origin of the colormap. Typically (0, 0), for colormaps
        that are linear on both axis, and (.5, .5) for circular colormaps.
        Used when getting 1D colormaps from 2D colormaps.

    name : str, optional
        The name of the colormap.
    """
    patch: Incomplete
    def __init__(self, patch, N: int = 256, shape: str = 'square', origin=(0, 0), name: str = 'segmented bivariate colormap') -> None: ...
    _lut: Incomplete
    _isinit: bool
    def _init(self) -> None: ...

class BivarColormapFromImage(BivarColormap):
    """
    BivarColormap object generated by supersampling a regular grid.

    Parameters
    ----------
    lut : nparray of shape (N, M, 3) or (N, M, 4)
        The look-up-table
    shape: {'square', 'circle', 'ignore', 'circleignore'}

        - If 'square' each variate is clipped to [0,1] independently
        - If 'circle' the variates are clipped radially to the center
          of the colormap, and a circular mask is applied when the colormap
          is displayed
        - If 'ignore' the variates are not clipped, but instead assigned the
          'outside' color
        - If 'circleignore' a circular mask is applied, but the data is not clipped

    origin: (float, float)
        The relative origin of the colormap. Typically (0, 0), for colormaps
        that are linear on both axis, and (.5, .5) for circular colormaps.
        Used when getting 1D colormaps from 2D colormaps.
    name : str, optional
        The name of the colormap.

    """
    _lut: Incomplete
    def __init__(self, lut, shape: str = 'square', origin=(0, 0), name: str = 'from image') -> None: ...
    _isinit: bool
    def _init(self) -> None: ...

class Normalize:
    """
    A class which, when called, maps values within the interval
    ``[vmin, vmax]`` linearly to the interval ``[0.0, 1.0]``. The mapping of
    values outside ``[vmin, vmax]`` depends on *clip*.

    Examples
    --------
    ::

        x = [-2, -1, 0, 1, 2]

        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
        norm(x)  # [-0.5, 0., 0.5, 1., 1.5]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=True)
        norm(x)  # [0., 0., 0.5, 1., 1.]

    See Also
    --------
    :ref:`colormapnorms`
    """
    _vmin: Incomplete
    _vmax: Incomplete
    _clip: Incomplete
    _scale: Incomplete
    callbacks: Incomplete
    def __init__(self, vmin: Incomplete | None = None, vmax: Incomplete | None = None, clip: bool = False) -> None:
        """
        Parameters
        ----------
        vmin, vmax : float or None
            Values within the range ``[vmin, vmax]`` from the input data will be
            linearly mapped to ``[0, 1]``. If either *vmin* or *vmax* is not
            provided, they default to the minimum and maximum values of the input,
            respectively.

        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Notes
        -----
        If ``vmin == vmax``, input data will be mapped to 0.
        """
    @property
    def vmin(self): ...
    @vmin.setter
    def vmin(self, value) -> None: ...
    @property
    def vmax(self): ...
    @vmax.setter
    def vmax(self, value) -> None: ...
    @property
    def clip(self): ...
    @clip.setter
    def clip(self, value) -> None: ...
    def _changed(self) -> None:
        """
        Call this whenever the norm is changed to notify all the
        callback listeners to the 'changed' signal.
        """
    @staticmethod
    def process_value(value):
        """
        Homogenize the input *value* for easy and efficient normalization.

        *value* can be a scalar or sequence.

        Parameters
        ----------
        value
            Data to normalize.

        Returns
        -------
        result : masked array
            Masked array with the same shape as *value*.
        is_scalar : bool
            Whether *value* is a scalar.

        Notes
        -----
        Float dtypes are preserved; integer types with two bytes or smaller are
        converted to np.float32, and larger types are converted to np.float64.
        Preserving float32 when possible, and using in-place operations,
        greatly improves speed for large arrays.
        """
    def __call__(self, value, clip: Incomplete | None = None):
        """
        Normalize the data and return the normalized data.

        Parameters
        ----------
        value
            Data to normalize.
        clip : bool, optional
            See the description of the parameter *clip* in `.Normalize`.

            If ``None``, defaults to ``self.clip`` (which defaults to
            ``False``).

        Notes
        -----
        If not already initialized, ``self.vmin`` and ``self.vmax`` are
        initialized using ``self.autoscale_None(value)``.
        """
    def inverse(self, value):
        """
        Maps the normalized value (i.e., index in the colormap) back to image
        data value.

        Parameters
        ----------
        value
            Normalized value.
        """
    def autoscale(self, A) -> None:
        """Set *vmin*, *vmax* to min, max of *A*."""
    def autoscale_None(self, A) -> None:
        """If *vmin* or *vmax* are not set, use the min/max of *A* to set them."""
    def scaled(self):
        """Return whether *vmin* and *vmax* are both set."""

class TwoSlopeNorm(Normalize):
    _vcenter: Incomplete
    def __init__(self, vcenter, vmin: Incomplete | None = None, vmax: Incomplete | None = None) -> None:
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
            ...                               vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """
    @property
    def vcenter(self): ...
    @vcenter.setter
    def vcenter(self, value) -> None: ...
    vmin: Incomplete
    vmax: Incomplete
    def autoscale_None(self, A) -> None:
        """
        Get vmin and vmax.

        If vcenter isn't in the range [vmin, vmax], either vmin or vmax
        is expanded so that vcenter lies in the middle of the modified range
        [vmin, vmax].
        """
    def __call__(self, value, clip: Incomplete | None = None):
        """
        Map value to the interval [0, 1]. The *clip* argument is unused.
        """
    def inverse(self, value): ...

class CenteredNorm(Normalize):
    _vcenter: Incomplete
    def __init__(self, vcenter: int = 0, halfrange: Incomplete | None = None, clip: bool = False) -> None:
        """
        Normalize symmetrical data around a center (0 by default).

        Unlike `TwoSlopeNorm`, `CenteredNorm` applies an equal rate of change
        around the center.

        Useful when mapping symmetrical data around a conceptual center
        e.g., data that range from -2 to 4, with 0 as the midpoint, and
        with equal rates of change around that midpoint.

        Parameters
        ----------
        vcenter : float, default: 0
            The data value that defines ``0.5`` in the normalization.
        halfrange : float, optional
            The range of data values that defines a range of ``0.5`` in the
            normalization, so that *vcenter* - *halfrange* is ``0.0`` and
            *vcenter* + *halfrange* is ``1.0`` in the normalization.
            Defaults to the largest absolute difference to *vcenter* for
            the values in the dataset.
        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Examples
        --------
        This maps data values -2 to 0.25, 0 to 0.5, and 4 to 1.0
        (assuming equal rates of change above and below 0.0):

            >>> import matplotlib.colors as mcolors
            >>> norm = mcolors.CenteredNorm(halfrange=4.0)
            >>> data = [-2., 0., 4.]
            >>> norm(data)
            array([0.25, 0.5 , 1.  ])
        """
    def autoscale(self, A) -> None:
        """
        Set *halfrange* to ``max(abs(A-vcenter))``, then set *vmin* and *vmax*.
        """
    def autoscale_None(self, A) -> None:
        """Set *vmin* and *vmax*."""
    @property
    def vmin(self): ...
    _vmin: Incomplete
    _vmax: Incomplete
    @vmin.setter
    def vmin(self, value) -> None: ...
    @property
    def vmax(self): ...
    @vmax.setter
    def vmax(self, value) -> None: ...
    @property
    def vcenter(self): ...
    @vcenter.setter
    def vcenter(self, vcenter) -> None: ...
    @property
    def halfrange(self): ...
    @halfrange.setter
    def halfrange(self, halfrange) -> None: ...

def make_norm_from_scale(scale_cls, base_norm_cls: Incomplete | None = None, *, init: Incomplete | None = None):
    """
    Decorator for building a `.Normalize` subclass from a `~.scale.ScaleBase`
    subclass.

    After ::

        @make_norm_from_scale(scale_cls)
        class norm_cls(Normalize):
            ...

    *norm_cls* is filled with methods so that normalization computations are
    forwarded to *scale_cls* (i.e., *scale_cls* is the scale that would be used
    for the colorbar of a mappable normalized with *norm_cls*).

    If *init* is not passed, then the constructor signature of *norm_cls*
    will be ``norm_cls(vmin=None, vmax=None, clip=False)``; these three
    parameters will be forwarded to the base class (``Normalize.__init__``),
    and a *scale_cls* object will be initialized with no arguments (other than
    a dummy axis).

    If the *scale_cls* constructor takes additional parameters, then *init*
    should be passed to `make_norm_from_scale`.  It is a callable which is
    *only* used for its signature.  First, this signature will become the
    signature of *norm_cls*.  Second, the *norm_cls* constructor will bind the
    parameters passed to it using this signature, extract the bound *vmin*,
    *vmax*, and *clip* values, pass those to ``Normalize.__init__``, and
    forward the remaining bound values (including any defaults defined by the
    signature) to the *scale_cls* constructor.
    """
def _make_norm_from_scale(scale_cls, scale_args, scale_kwargs_items, base_norm_cls, bound_init_signature):
    """
    Helper for `make_norm_from_scale`.

    This function is split out to enable caching (in particular so that
    different unpickles reuse the same class).  In order to do so,

    - ``functools.partial`` *scale_cls* is expanded into ``func, args, kwargs``
      to allow memoizing returned norms (partial instances always compare
      unequal, but we can check identity based on ``func, args, kwargs``;
    - *init* is replaced by *init_signature*, as signatures are picklable,
      unlike to arbitrary lambdas.
    """
def _create_empty_object_of_class(cls): ...
def _picklable_norm_constructor(*args): ...

class FuncNorm(Normalize):
    """
    Arbitrary normalization using functions for the forward and inverse.

    Parameters
    ----------
    functions : (callable, callable)
        two-tuple of the forward and inverse functions for the normalization.
        The forward function must be monotonic.

        Both functions must have the signature ::

           def forward(values: array-like) -> array-like

    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.

    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values outside the range ``[vmin, vmax]`` are also
        transformed by the function, resulting in values outside ``[0, 1]``.
        This behavior is usually desirable, as colormaps can mark these *under*
        and *over* values with specific colors.

        If clipping is on, values below *vmin* are mapped to 0 and values above
        *vmax* are mapped to 1. Such values become indistinguishable from
        regular boundary values, which may cause misinterpretation of the data.
    """

LogNorm: Incomplete

class SymLogNorm(Normalize):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    Parameters
    ----------
    linthresh : float
        The range within which the plot is linear (to avoid having the plot
        go to infinity around zero).
    linscale : float, default: 1
        This allows the linear range (-*linthresh* to *linthresh*) to be
        stretched relative to the logarithmic range. Its value is the
        number of decades to use for each half of the linear range. For
        example, when *linscale* == 1.0 (the default), the space used for
        the positive and negative halves of the linear range will be equal
        to one decade in the logarithmic range.
    base : float, default: 10
    """
    @property
    def linthresh(self): ...
    @linthresh.setter
    def linthresh(self, value) -> None: ...

class AsinhNorm(Normalize):
    """
    The inverse hyperbolic sine scale is approximately linear near
    the origin, but becomes logarithmic for larger positive
    or negative values. Unlike the `SymLogNorm`, the transition between
    these linear and logarithmic regions is smooth, which may reduce
    the risk of visual artifacts.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.

    Parameters
    ----------
    linear_width : float, default: 1
        The effective width of the linear region, beyond which
        the transformation becomes asymptotically logarithmic
    """
    @property
    def linear_width(self): ...
    @linear_width.setter
    def linear_width(self, value) -> None: ...

class PowerNorm(Normalize):
    """
    Linearly map a given value to the 0-1 range and then apply
    a power-law normalization over that range.

    Parameters
    ----------
    gamma : float
        Power law exponent.
    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.
    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values above *vmax* are transformed by the power
        function, resulting in values above 1, and values below *vmin* are linearly
        transformed resulting in values below 0. This behavior is usually desirable, as
        colormaps can mark these *under* and *over* values with specific colors.

        If clipping is on, values below *vmin* are mapped to 0 and values above
        *vmax* are mapped to 1. Such values become indistinguishable from
        regular boundary values, which may cause misinterpretation of the data.

    Notes
    -----
    The normalization formula is

    .. math::

        \\left ( \\frac{x - v_{min}}{v_{max}  - v_{min}} \\right )^{\\gamma}

    For input values below *vmin*, gamma is set to one.
    """
    gamma: Incomplete
    def __init__(self, gamma, vmin: Incomplete | None = None, vmax: Incomplete | None = None, clip: bool = False) -> None: ...
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value): ...

class BoundaryNorm(Normalize):
    """
    Generate a colormap index based on discrete intervals.

    Unlike `Normalize` or `LogNorm`, `BoundaryNorm` maps values to integers
    instead of to the interval 0-1.
    """
    boundaries: Incomplete
    N: Incomplete
    Ncmap: Incomplete
    extend: Incomplete
    _scale: Incomplete
    _n_regions: Incomplete
    _offset: int
    def __init__(self, boundaries, ncolors, clip: bool = False, *, extend: str = 'neither') -> None:
        """
        Parameters
        ----------
        boundaries : array-like
            Monotonically increasing sequence of at least 2 bin edges:  data
            falling in the n-th bin will be mapped to the n-th color.

        ncolors : int
            Number of colors in the colormap to be used.

        clip : bool, optional
            If clip is ``True``, out of range values are mapped to 0 if they
            are below ``boundaries[0]`` or mapped to ``ncolors - 1`` if they
            are above ``boundaries[-1]``.

            If clip is ``False``, out of range values are mapped to -1 if
            they are below ``boundaries[0]`` or mapped to *ncolors* if they are
            above ``boundaries[-1]``. These are then converted to valid indices
            by `Colormap.__call__`.

        extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
            Extend the number of bins to include one or both of the
            regions beyond the boundaries.  For example, if ``extend``
            is 'min', then the color to which the region between the first
            pair of boundaries is mapped will be distinct from the first
            color in the colormap, and by default a
            `~matplotlib.colorbar.Colorbar` will be drawn with
            the triangle extension on the left or lower end.

        Notes
        -----
        If there are fewer bins (including extensions) than colors, then the
        color index is chosen by linearly interpolating the ``[0, nbins - 1]``
        range onto the ``[0, ncolors - 1]`` range, effectively skipping some
        colors in the middle of the colormap.
        """
    def __call__(self, value, clip: Incomplete | None = None):
        """
        This method behaves similarly to `.Normalize.__call__`, except that it
        returns integers or arrays of int16.
        """
    def inverse(self, value) -> None:
        """
        Raises
        ------
        ValueError
            BoundaryNorm is not invertible, so calling this method will always
            raise an error
        """

class NoNorm(Normalize):
    """
    Dummy replacement for `Normalize`, for the case where we want to use
    indices directly in a `~matplotlib.cm.ScalarMappable`.
    """
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value): ...

def rgb_to_hsv(arr):
    """
    Convert an array of float RGB values (in the range [0, 1]) to HSV values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to HSV values in range [0, 1]
    """
def hsv_to_rgb(hsv):
    """
    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to RGB values in range [0, 1]
    """
def _vector_magnitude(arr): ...

class LightSource:
    '''
    Create a light source coming from the specified azimuth and elevation.
    Angles are in degrees, with the azimuth measured
    clockwise from north and elevation up from the zero plane of the surface.

    `shade` is used to produce "shaded" RGB values for a data array.
    `shade_rgb` can be used to combine an RGB image with an elevation map.
    `hillshade` produces an illumination map of a surface.
    '''
    azdeg: Incomplete
    altdeg: Incomplete
    hsv_min_val: Incomplete
    hsv_max_val: Incomplete
    hsv_min_sat: Incomplete
    hsv_max_sat: Incomplete
    def __init__(self, azdeg: int = 315, altdeg: int = 45, hsv_min_val: int = 0, hsv_max_val: int = 1, hsv_min_sat: int = 1, hsv_max_sat: int = 0) -> None:
        '''
        Specify the azimuth (measured clockwise from south) and altitude
        (measured up from the plane of the surface) of the light source
        in degrees.

        Parameters
        ----------
        azdeg : float, default: 315 degrees (from the northwest)
            The azimuth (0-360, degrees clockwise from North) of the light
            source.
        altdeg : float, default: 45 degrees
            The altitude (0-90, degrees up from horizontal) of the light
            source.
        hsv_min_val : number, default: 0
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to.
        hsv_max_val : number, default: 1
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to.
        hsv_min_sat : number, default: 1
            The minimum saturation value that the *intensity* map can shift the output
            image to.
        hsv_max_sat : number, default: 0
            The maximum saturation value that the *intensity* map can shift the output
            image to.

        Notes
        -----
        For backwards compatibility, the parameters *hsv_min_val*,
        *hsv_max_val*, *hsv_min_sat*, and *hsv_max_sat* may be supplied at
        initialization as well.  However, these parameters will only be used if
        "blend_mode=\'hsv\'" is passed into `shade` or `shade_rgb`.
        See the documentation for `blend_hsv` for more details.
        '''
    @property
    def direction(self):
        """The unit vector direction towards the light source."""
    def hillshade(self, elevation, vert_exag: int = 1, dx: int = 1, dy: int = 1, fraction: float = 1.0):
        """
        Calculate the illumination intensity for a surface using the defined
        azimuth and elevation for the light source.

        This computes the normal vectors for the surface, and then passes them
        on to `shade_normals`

        Parameters
        ----------
        elevation : 2D array-like
            The height values used to generate an illumination map
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topographic effects.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """
    def shade_normals(self, normals, fraction: float = 1.0):
        """
        Calculate the illumination intensity for the normal vectors of a
        surface using the defined azimuth and elevation for the light source.

        Imagine an artificial sun placed at infinity in some azimuth and
        elevation position illuminating our surface. The parts of the surface
        that slope toward the sun should brighten while those sides facing away
        should become darker.

        Parameters
        ----------
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """
    def shade(self, data, cmap, norm: Incomplete | None = None, blend_mode: str = 'overlay', vmin: Incomplete | None = None, vmax: Incomplete | None = None, vert_exag: int = 1, dx: int = 1, dy: int = 1, fraction: int = 1, **kwargs):
        '''
        Combine colormapped data values with an illumination intensity map
        (a.k.a.  "hillshade") of the values.

        Parameters
        ----------
        data : 2D array-like
            The height values used to generate a shaded map.
        cmap : `~matplotlib.colors.Colormap`
            The colormap used to color the *data* array. Note that this must be
            a `~matplotlib.colors.Colormap` instance.  For example, rather than
            passing in ``cmap=\'gist_earth\'``, use
            ``cmap=plt.get_cmap(\'gist_earth\')`` instead.
        norm : `~matplotlib.colors.Normalize` instance, optional
            The normalization used to scale values before colormapping. If
            None, the input will be linearly scaled between its min and max.
        blend_mode : {\'hsv\', \'overlay\', \'soft\'} or callable, optional
            The type of blending used to combine the colormapped data
            values with the illumination intensity.  Default is
            "overlay".  Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to
            combine an (M, N, 3) RGB array of floats (ranging 0 to 1) with
            an (M, N, 1) hillshade array (also 0 to 1).  (Call signature
            ``func(rgb, illum, **kwargs)``) Additional kwargs supplied
            to this function will be passed on to the *blend_mode*
            function.
        vmin : float or None, optional
            The minimum value used in colormapping *data*. If *None* the
            minimum value in *data* is used. If *norm* is specified, then this
            argument will be ignored.
        vmax : float or None, optional
            The maximum value used in colormapping *data*. If *None* the
            maximum value in *data* is used. If *norm* is specified, then this
            argument will be ignored.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        **kwargs
            Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 4) array of floats ranging between 0-1.
        '''
    def shade_rgb(self, rgb, elevation, fraction: float = 1.0, blend_mode: str = 'hsv', vert_exag: int = 1, dx: int = 1, dy: int = 1, **kwargs):
        '''
        Use this light source to adjust the colors of the *rgb* input array to
        give the impression of a shaded relief map with the given *elevation*.

        Parameters
        ----------
        rgb : array-like
            An (M, N, 3) RGB array, assumed to be in the range of 0 to 1.
        elevation : array-like
            An (M, N) array of the height values used to generate a shaded map.
        fraction : number
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        blend_mode : {\'hsv\', \'overlay\', \'soft\'} or callable, optional
            The type of blending used to combine the colormapped data values
            with the illumination intensity.  For backwards compatibility, this
            defaults to "hsv". Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to combine an
            (M, N, 3) RGB array of floats (ranging 0 to 1) with an (M, N, 1)
            hillshade array (also 0 to 1).  (Call signature
            ``func(rgb, illum, **kwargs)``)
            Additional kwargs supplied to this function will be passed on to
            the *blend_mode* function.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        **kwargs
            Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        `~numpy.ndarray`
            An (m, n, 3) array of floats ranging between 0-1.
        '''
    def blend_hsv(self, rgb, intensity, hsv_max_sat: Incomplete | None = None, hsv_max_val: Incomplete | None = None, hsv_min_val: Incomplete | None = None, hsv_min_sat: Incomplete | None = None):
        '''
        Take the input data array, convert to HSV values in the given colormap,
        then adjust those color values to give the impression of a shaded
        relief map with a specified light source.  RGBA values are returned,
        which can then be used to plot the shaded image with imshow.

        The color of the resulting image will be darkened by moving the (s, v)
        values (in HSV colorspace) toward (hsv_min_sat, hsv_min_val) in the
        shaded regions, or lightened by sliding (s, v) toward (hsv_max_sat,
        hsv_max_val) in regions that are illuminated.  The default extremes are
        chose so that completely shaded points are nearly black (s = 1, v = 0)
        and completely illuminated points are nearly white (s = 0, v = 1).

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).
        hsv_max_sat : number, optional
            The maximum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_min_sat : number, optional
            The minimum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_max_val : number, optional
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.
        hsv_min_val : number, optional
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        '''
    def blend_soft_light(self, rgb, intensity):
        '''
        Combine an RGB image with an intensity map using "soft light" blending,
        using the "pegtop" formula.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        '''
    def blend_overlay(self, rgb, intensity):
        '''
        Combine an RGB image with an intensity map using "overlay" blending.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        ndarray
            An (M, N, 3) RGB array representing the combined images.
        '''

def from_levels_and_colors(levels, colors, extend: str = 'neither'):
    '''
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf\'s levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the `BoundaryNorm`.
        Value ``v`` is quantized to level ``i`` if ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If *extend* is "neither" there
        must be ``n_level - 1`` colors. For an *extend* of "min" or "max" add
        one extra color, and for an *extend* of "both" add two colors.
    extend : {\'neither\', \'min\', \'max\', \'both\'}, optional
        The behaviour when a value falls out of range of the given levels.
        See `~.Axes.contourf` for details.

    Returns
    -------
    cmap : `~matplotlib.colors.Colormap`
    norm : `~matplotlib.colors.Normalize`
    '''

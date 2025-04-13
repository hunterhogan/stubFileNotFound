from _typeshed import Incomplete
from matplotlib import _api as _api, artist as artist, cbook as cbook, colors as colors, scale as scale

class Colorizer:
    """
    Data to color pipeline.

    This pipeline is accessible via `.Colorizer.to_rgba` and executed via
    the `.Colorizer.norm` and `.Colorizer.cmap` attributes.

    Parameters
    ----------
    cmap: colorbar.Colorbar or str or None, default: None
        The colormap used to color data.

    norm: colors.Normalize or str or None, default: None
        The normalization used to normalize the data
    """
    _cmap: Incomplete
    _id_norm: Incomplete
    _norm: Incomplete
    callbacks: Incomplete
    colorbar: Incomplete
    def __init__(self, cmap: Incomplete | None = None, norm: Incomplete | None = None) -> None: ...
    def _scale_norm(self, norm, vmin, vmax, A) -> None:
        """
        Helper for initial scaling.

        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
    @property
    def norm(self): ...
    @norm.setter
    def norm(self, norm) -> None: ...
    def to_rgba(self, x, alpha: Incomplete | None = None, bytes: bool = False, norm: bool = True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this Colorizer.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
    @staticmethod
    def _pass_image_data(x, alpha: Incomplete | None = None, bytes: bool = False, norm: bool = True):
        """
        Helper function to pass ndarray of shape (...,3) or (..., 4)
        through `to_rgba()`, see `to_rgba()` for docstring.
        """
    def autoscale(self, A) -> None:
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
    def autoscale_None(self, A) -> None:
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
    def _set_cmap(self, cmap) -> None:
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
    @property
    def cmap(self): ...
    @cmap.setter
    def cmap(self, cmap) -> None: ...
    def set_clim(self, vmin: Incomplete | None = None, vmax: Incomplete | None = None) -> None:
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
    stale: bool
    def changed(self) -> None:
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
    @property
    def vmin(self): ...
    @vmin.setter
    def vmin(self, vmin) -> None: ...
    @property
    def vmax(self): ...
    @vmax.setter
    def vmax(self, vmax) -> None: ...
    @property
    def clip(self): ...
    @clip.setter
    def clip(self, clip) -> None: ...

class _ColorizerInterface:
    """
    Base class that contains the interface to `Colorizer` objects from
    a `ColorizingArtist` or `.cm.ScalarMappable`.

    Note: This class only contain functions that interface the .colorizer
    attribute. Other functions that as shared between `.ColorizingArtist`
    and `.cm.ScalarMappable` are not included.
    """
    def _scale_norm(self, norm, vmin, vmax) -> None: ...
    def to_rgba(self, x, alpha: Incomplete | None = None, bytes: bool = False, norm: bool = True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this Colorizer.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
    def set_clim(self, vmin: Incomplete | None = None, vmax: Incomplete | None = None) -> None:
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             For scalar data, the limits may also be passed as a
             tuple (*vmin*, *vmax*) as a single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
    def get_alpha(self): ...
    @property
    def cmap(self): ...
    @cmap.setter
    def cmap(self, cmap) -> None: ...
    def get_cmap(self):
        """Return the `.Colormap` instance."""
    def set_cmap(self, cmap) -> None:
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
    @property
    def norm(self): ...
    @norm.setter
    def norm(self, norm) -> None: ...
    def set_norm(self, norm) -> None:
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or str or None

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
    def autoscale(self) -> None:
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
    def autoscale_None(self) -> None:
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
    @property
    def colorbar(self):
        """
        The last colorbar associated with this object. May be None
        """
    @colorbar.setter
    def colorbar(self, colorbar) -> None: ...
    def _format_cursor_data_override(self, data): ...

class _ScalarMappable(_ColorizerInterface):
    """
    A mixin class to map one or multiple sets of scalar data to RGBA.

    The ScalarMappable applies data normalization before returning RGBA colors from
    the given `~matplotlib.colors.Colormap`.
    """
    _A: Incomplete
    _colorizer: Incomplete
    colorbar: Incomplete
    _id_colorizer: Incomplete
    callbacks: Incomplete
    def __init__(self, norm: Incomplete | None = None, cmap: Incomplete | None = None, *, colorizer: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
    def set_array(self, A) -> None:
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
    def get_array(self):
        """
        Return the array of values, that are mapped to colors.

        The base class `.ScalarMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
    stale: bool
    def changed(self) -> None:
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
    @staticmethod
    def _check_exclusionary_keywords(colorizer, **kwargs) -> None:
        """
        Raises a ValueError if any kwarg is not None while colorizer is not None
        """
    @staticmethod
    def _get_colorizer(cmap, norm, colorizer): ...

class ColorizingArtist(_ScalarMappable, artist.Artist):
    """
    Base class for artists that make map data to color using a `.colorizer.Colorizer`.

    The `.colorizer.Colorizer` applies data normalization before
    returning RGBA colors from a `~matplotlib.colors.Colormap`.

    """
    def __init__(self, colorizer, **kwargs) -> None:
        """
        Parameters
        ----------
        colorizer : `.colorizer.Colorizer`
        """
    @property
    def colorizer(self): ...
    _colorizer: Incomplete
    _id_colorizer: Incomplete
    @colorizer.setter
    def colorizer(self, cl) -> None: ...
    def _set_colorizer_check_keywords(self, colorizer, **kwargs) -> None:
        """
        Raises a ValueError if any kwarg is not None while colorizer is not None.
        """

def _auto_norm_from_scale(scale_cls):
    '''
    Automatically generate a norm class from *scale_cls*.

    This differs from `.colors.make_norm_from_scale` in the following points:

    - This function is not a class decorator, but directly returns a norm class
      (as if decorating `.Normalize`).
    - The scale is automatically constructed with ``nonpositive="mask"``, if it
      supports such a parameter, to work around the difference in defaults
      between standard scales (which use "clip") and norms (which use "mask").

    Note that ``make_norm_from_scale`` caches the generated norm classes
    (not the instances) and reuses them for later calls.  For example,
    ``type(_auto_norm_from_scale("log")) == LogNorm``.
    '''

from matplotlib._image import *
import matplotlib.colorizer as mcolorizer
from _typeshed import Incomplete
from matplotlib import _api as _api, _image as _image, cbook as cbook
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase
from matplotlib.transforms import Affine2D as Affine2D, Bbox as Bbox, BboxBase as BboxBase, BboxTransform as BboxTransform, BboxTransformTo as BboxTransformTo, IdentityTransform as IdentityTransform, TransformedBbox as TransformedBbox

_log: Incomplete
_interpd_: Incomplete
interpolations_names: Incomplete

def composite_images(images, renderer, magnification: float = 1.0):
    """
    Composite a number of RGBA images into one.  The images are
    composited in the order in which they appear in the *images* list.

    Parameters
    ----------
    images : list of Images
        Each must have a `make_image` method.  For each image,
        `can_composite` should return `True`, though this is not
        enforced by this function.  Each image must have a purely
        affine transformation with no shear.

    renderer : `.RendererBase`

    magnification : float, default: 1
        The additional magnification to apply for the renderer in use.

    Returns
    -------
    image : (M, N, 4) `numpy.uint8` array
        The composited RGBA image.
    offset_x, offset_y : float
        The (left, bottom) offset where the composited image should be placed
        in the output figure.
    """
def _draw_list_compositing_images(renderer, parent, artists, suppress_composite: Incomplete | None = None) -> None:
    """
    Draw a sorted list of artists, compositing images into a single
    image where possible.

    For internal Matplotlib use only: It is here to reduce duplication
    between `Figure.draw` and `Axes.draw`, but otherwise should not be
    generally useful.
    """
def _resample(image_obj, data, out_shape, transform, *, resample: Incomplete | None = None, alpha: int = 1):
    """
    Convenience wrapper around `._image.resample` to resample *data* to
    *out_shape* (with a third dimension if *data* is RGBA) that takes care of
    allocating the output array and fetching the relevant properties from the
    Image object *image_obj*.
    """
def _rgb_to_rgba(A):
    """
    Convert an RGB image to RGBA, as required by the image resample C++
    extension.
    """

class _ImageBase(mcolorizer.ColorizingArtist):
    """
    Base class for images.

    interpolation and cmap default to their rc settings

    cmap is a colors.Colormap instance
    norm is a colors.Normalize instance to map luminance to 0-1

    extent is data axes (left, right, bottom, top) for making image plots
    registered with data plots.  Default is to label the pixel
    centers with the zero-based row and column indices.

    Additional kwargs are matplotlib.artist properties
    """
    zorder: int
    origin: Incomplete
    axes: Incomplete
    _imcache: Incomplete
    def __init__(self, ax, cmap: Incomplete | None = None, norm: Incomplete | None = None, colorizer: Incomplete | None = None, interpolation: Incomplete | None = None, origin: Incomplete | None = None, filternorm: bool = True, filterrad: float = 4.0, resample: bool = False, *, interpolation_stage: Incomplete | None = None, **kwargs) -> None: ...
    def __str__(self) -> str: ...
    def __getstate__(self): ...
    def get_size(self):
        """Return the size of the image as tuple (numrows, numcols)."""
    def get_shape(self):
        """
        Return the shape of the image as tuple (numrows, numcols, channels).
        """
    def set_alpha(self, alpha) -> None:
        """
        Set the alpha value used for blending - not supported on all backends.

        Parameters
        ----------
        alpha : float or 2D array-like or None
        """
    def _get_scalar_alpha(self):
        """
        Get a scalar alpha value to be applied to the artist as a whole.

        If the alpha value is a matrix, the method returns 1.0 because pixels
        have individual alpha values (see `~._ImageBase._make_image` for
        details). If the alpha value is a scalar, the method returns said value
        to be applied to the artist as a whole because pixels do not have
        individual alpha values.
        """
    def changed(self) -> None:
        """
        Call this whenever the mappable is changed so observers can update.
        """
    def _make_image(self, A, in_bbox, out_bbox, clip_bbox, magnification: float = 1.0, unsampled: bool = False, round_to_pixel_border: bool = True):
        """
        Normalize, rescale, and colormap the image *A* from the given *in_bbox*
        (in data space), to the given *out_bbox* (in pixel space) clipped to
        the given *clip_bbox* (also in pixel space), and magnified by the
        *magnification* factor.

        Parameters
        ----------
        A : ndarray

            - a (M, N) array interpreted as scalar (greyscale) image,
              with one of the dtypes `~numpy.float32`, `~numpy.float64`,
              `~numpy.float128`, `~numpy.uint16` or `~numpy.uint8`.
            - (M, N, 4) RGBA image with a dtype of `~numpy.float32`,
              `~numpy.float64`, `~numpy.float128`, or `~numpy.uint8`.

        in_bbox : `~matplotlib.transforms.Bbox`

        out_bbox : `~matplotlib.transforms.Bbox`

        clip_bbox : `~matplotlib.transforms.Bbox`

        magnification : float, default: 1

        unsampled : bool, default: False
            If True, the image will not be scaled, but an appropriate
            affine transformation will be returned instead.

        round_to_pixel_border : bool, default: True
            If True, the output image size will be rounded to the nearest pixel
            boundary.  This makes the images align correctly with the Axes.
            It should not be used if exact scaling is needed, such as for
            `.FigureImage`.

        Returns
        -------
        image : (M, N, 4) `numpy.uint8` array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : `~matplotlib.transforms.Affine2D`
            The affine transformation from image to pixel space.
        """
    def make_image(self, renderer, magnification: float = 1.0, unsampled: bool = False) -> None:
        """
        Normalize, rescale, and colormap this image's data for rendering using
        *renderer*, with the given *magnification*.

        If *unsampled* is True, the image will not be scaled, but an
        appropriate affine transformation will be returned instead.

        Returns
        -------
        image : (M, N, 4) `numpy.uint8` array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : `~matplotlib.transforms.Affine2D`
            The affine transformation from image to pixel space.
        """
    def _check_unsampled_image(self):
        """
        Return whether the image is better to be drawn unsampled.

        The derived class needs to override it.
        """
    stale: bool
    def draw(self, renderer) -> None: ...
    def contains(self, mouseevent):
        """Test whether the mouse event occurred within the image."""
    def write_png(self, fname) -> None:
        """Write the image to png file *fname*."""
    @staticmethod
    def _normalize_image_array(A):
        """
        Check validity of image-like input *A* and normalize it to a format suitable for
        Image subclasses.
        """
    _A: Incomplete
    def set_data(self, A) -> None:
        """
        Set the image array.

        Note that this function does *not* update the normalization used.

        Parameters
        ----------
        A : array-like or `PIL.Image.Image`
        """
    def set_array(self, A) -> None:
        """
        Retained for backwards compatibility - use set_data instead.

        Parameters
        ----------
        A : array-like
        """
    def get_interpolation(self):
        """
        Return the interpolation method the image uses when resizing.

        One of 'auto', 'antialiased', 'nearest', 'bilinear', 'bicubic',
        'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
        'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos',
        or 'none'.
        """
    _interpolation: Incomplete
    def set_interpolation(self, s) -> None:
        """
        Set the interpolation method the image uses when resizing.

        If None, use :rc:`image.interpolation`. If 'none', the image is
        shown as is without interpolating. 'none' is only supported in
        agg, ps and pdf backends and will fall back to 'nearest' mode
        for other backends.

        Parameters
        ----------
        s : {'auto', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'none'} or None
        """
    def get_interpolation_stage(self):
        """
        Return when interpolation happens during the transform to RGBA.

        One of 'data', 'rgba', 'auto'.
        """
    _interpolation_stage: Incomplete
    def set_interpolation_stage(self, s) -> None:
        """
        Set when interpolation happens during the transform to RGBA.

        Parameters
        ----------
        s : {'data', 'rgba', 'auto'} or None
            Whether to apply up/downsampling interpolation in data or RGBA
            space.  If None, use :rc:`image.interpolation_stage`.
            If 'auto' we will check upsampling rate and if less
            than 3 then use 'rgba', otherwise use 'data'.
        """
    def can_composite(self):
        """Return whether the image can be composited with its neighbors."""
    _resample: Incomplete
    def set_resample(self, v) -> None:
        """
        Set whether image resampling is used.

        Parameters
        ----------
        v : bool or None
            If None, use :rc:`image.resample`.
        """
    def get_resample(self):
        """Return whether image resampling is used."""
    _filternorm: Incomplete
    def set_filternorm(self, filternorm) -> None:
        """
        Set whether the resize filter normalizes the weights.

        See help for `~.Axes.imshow`.

        Parameters
        ----------
        filternorm : bool
        """
    def get_filternorm(self):
        """Return whether the resize filter normalizes the weights."""
    _filterrad: Incomplete
    def set_filterrad(self, filterrad) -> None:
        """
        Set the resize filter radius only applicable to some
        interpolation schemes -- see help for imshow

        Parameters
        ----------
        filterrad : positive float
        """
    def get_filterrad(self):
        """Return the filterrad setting."""

class AxesImage(_ImageBase):
    """
    An image attached to an Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The Axes the image will belong to.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar
        data to colors.
    norm : str or `~matplotlib.colors.Normalize`
        Maps luminance to 0-1.
    interpolation : str, default: :rc:`image.interpolation`
        Supported values are 'none', 'auto', 'nearest', 'bilinear',
        'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
        'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
        'sinc', 'lanczos', 'blackman'.
    interpolation_stage : {'data', 'rgba'}, default: 'data'
        If 'data', interpolation
        is carried out on the data provided by the user.  If 'rgba', the
        interpolation is carried out after the colormapping has been
        applied (visual interpolation).
    origin : {'upper', 'lower'}, default: :rc:`image.origin`
        Place the [0, 0] index of the array in the upper left or lower left
        corner of the Axes. The convention 'upper' is typically used for
        matrices and images.
    extent : tuple, optional
        The data axes (left, right, bottom, top) for making image plots
        registered with data plots.  Default is to label the pixel
        centers with the zero-based row and column indices.
    filternorm : bool, default: True
        A parameter for the antigrain image resize filter
        (see the antigrain documentation).
        If filternorm is set, the filter normalizes integer values and corrects
        the rounding errors. It doesn't do anything with the source floating
        point values, it corrects only integers according to the rule of 1.0
        which means that any sum of pixel weights must be equal to 1.0. So,
        the filter function must produce a graph of the proper shape.
    filterrad : float > 0, default: 4
        The filter radius for filters that have a radius parameter, i.e. when
        interpolation is one of: 'sinc', 'lanczos' or 'blackman'.
    resample : bool, default: False
        When True, use a full resampling method. When False, only resample when
        the output image is larger than the input image.
    **kwargs : `~matplotlib.artist.Artist` properties
    """
    _extent: Incomplete
    def __init__(self, ax, *, cmap: Incomplete | None = None, norm: Incomplete | None = None, colorizer: Incomplete | None = None, interpolation: Incomplete | None = None, origin: Incomplete | None = None, extent: Incomplete | None = None, filternorm: bool = True, filterrad: float = 4.0, resample: bool = False, interpolation_stage: Incomplete | None = None, **kwargs) -> None: ...
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def make_image(self, renderer, magnification: float = 1.0, unsampled: bool = False): ...
    def _check_unsampled_image(self):
        """Return whether the image would be better drawn unsampled."""
    stale: bool
    def set_extent(self, extent, **kwargs) -> None:
        """
        Set the image extent.

        Parameters
        ----------
        extent : 4-tuple of float
            The position and size of the image as tuple
            ``(left, right, bottom, top)`` in data coordinates.
        **kwargs
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar Axes) entries are applied, if present.

        Notes
        -----
        This updates `.Axes.dataLim`, and, if autoscaling, sets `.Axes.viewLim`
        to tightly fit the image, regardless of `~.Axes.dataLim`.  Autoscaling
        state is not changed, so a subsequent call to `.Axes.autoscale_view`
        will redo the autoscaling in accord with `~.Axes.dataLim`.
        """
    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
    def get_cursor_data(self, event):
        """
        Return the image value at the event position or *None* if the event is
        outside the image.

        See Also
        --------
        matplotlib.artist.Artist.get_cursor_data
        """

class NonUniformImage(AxesImage):
    def __init__(self, ax, *, interpolation: str = 'nearest', **kwargs) -> None:
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes the image will belong to.
        interpolation : {'nearest', 'bilinear'}, default: 'nearest'
            The interpolation scheme used in the resampling.
        **kwargs
            All other keyword arguments are identical to those of `.AxesImage`.
        """
    def _check_unsampled_image(self):
        """Return False. Do not use unsampled image."""
    def make_image(self, renderer, magnification: float = 1.0, unsampled: bool = False): ...
    _A: Incomplete
    _Ax: Incomplete
    _Ay: Incomplete
    _imcache: Incomplete
    stale: bool
    def set_data(self, x, y, A) -> None:
        """
        Set the grid for the pixel centers, and the pixel values.

        Parameters
        ----------
        x, y : 1D array-like
            Monotonic arrays of shapes (N,) and (M,), respectively, specifying
            pixel centers.
        A : array-like
            (M, N) `~numpy.ndarray` or masked array of values to be
            colormapped, or (M, N, 3) RGB array, or (M, N, 4) RGBA array.
        """
    def set_array(self, *args) -> None: ...
    def set_interpolation(self, s) -> None:
        """
        Parameters
        ----------
        s : {'nearest', 'bilinear'} or None
            If None, use :rc:`image.interpolation`.
        """
    def get_extent(self): ...
    def set_filternorm(self, filternorm) -> None: ...
    def set_filterrad(self, filterrad) -> None: ...
    def set_norm(self, norm) -> None: ...
    def set_cmap(self, cmap) -> None: ...
    def get_cursor_data(self, event): ...

class PcolorImage(AxesImage):
    """
    Make a pcolor-style plot with an irregular rectangular grid.

    This uses a variation of the original irregular image code,
    and it is used by pcolorfast for the corresponding grid type.
    """
    def __init__(self, ax, x: Incomplete | None = None, y: Incomplete | None = None, A: Incomplete | None = None, *, cmap: Incomplete | None = None, norm: Incomplete | None = None, colorizer: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes the image will belong to.
        x, y : 1D array-like, optional
            Monotonic arrays of length N+1 and M+1, respectively, specifying
            rectangle boundaries.  If not given, will default to
            ``range(N + 1)`` and ``range(M + 1)``, respectively.
        A : array-like
            The data to be color-coded. The interpretation depends on the
            shape:

            - (M, N) `~numpy.ndarray` or masked array: values to be colormapped
            - (M, N, 3): RGB array
            - (M, N, 4): RGBA array

        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The Colormap instance or registered colormap name used to map
            scalar data to colors.
        norm : str or `~matplotlib.colors.Normalize`
            Maps luminance to 0-1.
        **kwargs : `~matplotlib.artist.Artist` properties
        """
    _imcache: Incomplete
    def make_image(self, renderer, magnification: float = 1.0, unsampled: bool = False): ...
    def _check_unsampled_image(self): ...
    _A: Incomplete
    _Ax: Incomplete
    _Ay: Incomplete
    stale: bool
    def set_data(self, x, y, A) -> None:
        """
        Set the grid for the rectangle boundaries, and the data values.

        Parameters
        ----------
        x, y : 1D array-like, optional
            Monotonic arrays of length N+1 and M+1, respectively, specifying
            rectangle boundaries.  If not given, will default to
            ``range(N + 1)`` and ``range(M + 1)``, respectively.
        A : array-like
            The data to be color-coded. The interpretation depends on the
            shape:

            - (M, N) `~numpy.ndarray` or masked array: values to be colormapped
            - (M, N, 3): RGB array
            - (M, N, 4): RGBA array
        """
    def set_array(self, *args) -> None: ...
    def get_cursor_data(self, event): ...

class FigureImage(_ImageBase):
    """An image attached to a figure."""
    zorder: int
    _interpolation: str
    ox: Incomplete
    oy: Incomplete
    magnification: float
    def __init__(self, fig, *, cmap: Incomplete | None = None, norm: Incomplete | None = None, colorizer: Incomplete | None = None, offsetx: int = 0, offsety: int = 0, origin: Incomplete | None = None, **kwargs) -> None:
        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        kwargs are an optional list of Artist keyword args
        """
    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
    def make_image(self, renderer, magnification: float = 1.0, unsampled: bool = False): ...
    stale: bool
    def set_data(self, A) -> None:
        """Set the image array."""

class BboxImage(_ImageBase):
    """The Image class whose size is determined by the given bbox."""
    bbox: Incomplete
    def __init__(self, bbox, *, cmap: Incomplete | None = None, norm: Incomplete | None = None, colorizer: Incomplete | None = None, interpolation: Incomplete | None = None, origin: Incomplete | None = None, filternorm: bool = True, filterrad: float = 4.0, resample: bool = False, **kwargs) -> None:
        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        kwargs are an optional list of Artist keyword args
        """
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def contains(self, mouseevent):
        """Test whether the mouse event occurred within the image."""
    _transform: Incomplete
    def make_image(self, renderer, magnification: float = 1.0, unsampled: bool = False): ...

def imread(fname, format: Incomplete | None = None):
    '''
    Read an image from a file into an array.

    .. note::

        This function exists for historical reasons.  It is recommended to
        use `PIL.Image.open` instead for loading images.

    Parameters
    ----------
    fname : str or file-like
        The image file to read: a filename, a URL or a file-like object opened
        in read-binary mode.

        Passing a URL is deprecated.  Please open the URL
        for reading and pass the result to Pillow, e.g. with
        ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.
    format : str, optional
        The image file format assumed for reading the data.  The image is
        loaded as a PNG file if *format* is set to "png", if *fname* is a path
        or opened file with a ".png" extension, or if it is a URL.  In all
        other cases, *format* is ignored and the format is auto-detected by
        `PIL.Image.open`.

    Returns
    -------
    `numpy.array`
        The image data. The returned array has shape

        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.

        PNG images are returned as float arrays (0-1).  All other formats are
        returned as int arrays, with a bit depth determined by the file\'s
        contents.
    '''
def imsave(fname, arr, vmin: Incomplete | None = None, vmax: Incomplete | None = None, cmap: Incomplete | None = None, format: Incomplete | None = None, origin: Incomplete | None = None, dpi: int = 100, *, metadata: Incomplete | None = None, pil_kwargs: Incomplete | None = None) -> None:
    '''
    Colormap and save an array as an image file.

    RGB(A) images are passed through.  Single channel images will be
    colormapped according to *cmap* and *norm*.

    .. note::

       If you want to save a single channel image as gray scale please use an
       image I/O library (such as pillow, tifffile, or imageio) directly.

    Parameters
    ----------
    fname : str or path-like or file-like
        A path or a file-like object to store the image in.
        If *format* is not set, then the output format is inferred from the
        extension of *fname*, if any, and from :rc:`savefig.format` otherwise.
        If *format* is set, it determines the output format.
    arr : array-like
        The image data. The shape can be one of
        MxN (luminance), MxNx3 (RGB) or MxNx4 (RGBA).
    vmin, vmax : float, optional
        *vmin* and *vmax* set the color scaling for the image by fixing the
        values that map to the colormap color limits. If either *vmin*
        or *vmax* is None, that limit is determined from the *arr*
        min/max value.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        A Colormap instance or registered colormap name. The colormap
        maps scalar data to colors. It is ignored for RGB(A) data.
    format : str, optional
        The file format, e.g. \'png\', \'pdf\', \'svg\', ...  The behavior when this
        is unset is documented under *fname*.
    origin : {\'upper\', \'lower\'}, default: :rc:`image.origin`
        Indicates whether the ``(0, 0)`` index of the array is in the upper
        left or lower left corner of the Axes.
    dpi : float
        The DPI to store in the metadata of the file.  This does not affect the
        resolution of the output image.  Depending on file format, this may be
        rounded to the nearest integer.
    metadata : dict, optional
        Metadata in the image file.  The supported keys depend on the output
        format, see the documentation of the respective backends for more
        information.
        Currently only supported for "png", "pdf", "ps", "eps", and "svg".
    pil_kwargs : dict, optional
        Keyword arguments passed to `PIL.Image.Image.save`.  If the \'pnginfo\'
        key is present, it completely overrides *metadata*, including the
        default \'Software\' key.
    '''
def pil_to_array(pilImage):
    """
    Load a `PIL image`_ and return it as a numpy int array.

    .. _PIL image: https://pillow.readthedocs.io/en/latest/reference/Image.html

    Returns
    -------
    numpy.array

        The array shape depends on the image type:

        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.
    """
def _pil_png_to_float_array(pil_png):
    """Convert a PIL `PNGImageFile` to a 0-1 float array."""
def thumbnail(infile, thumbfile, scale: float = 0.1, interpolation: str = 'bilinear', preview: bool = False):
    """
    Make a thumbnail of image in *infile* with output filename *thumbfile*.

    See :doc:`/gallery/misc/image_thumbnail_sgskip`.

    Parameters
    ----------
    infile : str or file-like
        The image file. Matplotlib relies on Pillow_ for image reading, and
        thus supports a wide range of file formats, including PNG, JPG, TIFF
        and others.

        .. _Pillow: https://python-pillow.github.io

    thumbfile : str or file-like
        The thumbnail filename.

    scale : float, default: 0.1
        The scale factor for the thumbnail.

    interpolation : str, default: 'bilinear'
        The interpolation scheme used in the resampling. See the
        *interpolation* parameter of `~.Axes.imshow` for possible values.

    preview : bool, default: False
        If True, the default backend (presumably a user interface
        backend) will be used which will cause a figure to be raised if
        `~matplotlib.pyplot.show` is called.  If it is False, the figure is
        created using `.FigureCanvasBase` and the drawing backend is selected
        as `.Figure.savefig` would normally do.

    Returns
    -------
    `.Figure`
        The figure instance containing the thumbnail.
    """

from _typeshed import Incomplete
from matplotlib import _path as _path, cbook as cbook
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, RendererBase as RendererBase, _Backend as _Backend
from matplotlib.backends.backend_mixed import MixedModeRenderer as MixedModeRenderer
from matplotlib.colors import rgb2hex as rgb2hex
from matplotlib.dates import UTC as UTC
from matplotlib.path import Path as Path
from matplotlib.transforms import Affine2D as Affine2D, Affine2DBase as Affine2DBase

_log: Incomplete

def _escape_cdata(s): ...

_escape_xml_comment: Incomplete

def _escape_comment(s): ...
def _escape_attrib(s): ...
def _quote_escape_attrib(s): ...
def _short_float_fmt(x):
    """
    Create a short string representation of a float, which is %f
    formatting with trailing zeros and the decimal point removed.
    """

class XMLWriter:
    """
    Parameters
    ----------
    file : writable text file-like object
    """
    __write: Incomplete
    __open: int
    __tags: Incomplete
    __data: Incomplete
    __indentation: Incomplete
    def __init__(self, file) -> None: ...
    def __flush(self, indent: bool = True) -> None: ...
    def start(self, tag, attrib={}, **extra):
        """
        Open a new element.  Attributes can be given as keyword
        arguments, or as a string/string dictionary. The method returns
        an opaque identifier that can be passed to the :meth:`close`
        method, to close all open elements up to and including this one.

        Parameters
        ----------
        tag
            Element tag.
        attrib
            Attribute dictionary.  Alternatively, attributes can be given as
            keyword arguments.

        Returns
        -------
        An element identifier.
        """
    def comment(self, comment) -> None:
        """
        Add a comment to the output stream.

        Parameters
        ----------
        comment : str
            Comment text.
        """
    def data(self, text) -> None:
        """
        Add character data to the output stream.

        Parameters
        ----------
        text : str
            Character data.
        """
    def end(self, tag: Incomplete | None = None, indent: bool = True) -> None:
        """
        Close the current element (opened by the most recent call to
        :meth:`start`).

        Parameters
        ----------
        tag
            Element tag.  If given, the tag must match the start tag.  If
            omitted, the current element is closed.
        indent : bool, default: True
        """
    def close(self, id) -> None:
        """
        Close open elements, up to (and including) the element identified
        by the given identifier.

        Parameters
        ----------
        id
            Element identifier, as returned by the :meth:`start` method.
        """
    def element(self, tag, text: Incomplete | None = None, attrib={}, **extra) -> None:
        """
        Add an entire element.  This is the same as calling :meth:`start`,
        :meth:`data`, and :meth:`end` in sequence. The *text* argument can be
        omitted.
        """
    def flush(self) -> None:
        """Flush the output stream."""

def _generate_transform(transform_list): ...
def _generate_css(attrib): ...

_capstyle_d: Incomplete

def _check_is_str(info, key) -> None: ...
def _check_is_iterable_of_str(infos, key) -> None: ...

class RendererSVG(RendererBase):
    width: Incomplete
    height: Incomplete
    writer: Incomplete
    image_dpi: Incomplete
    basename: Incomplete
    _groupd: Incomplete
    _image_counter: Incomplete
    _clip_path_ids: Incomplete
    _clipd: Incomplete
    _markers: Incomplete
    _path_collection_id: int
    _hatchd: Incomplete
    _has_gouraud: bool
    _n_gradients: int
    _glyph_map: Incomplete
    _start_id: Incomplete
    def __init__(self, width, height, svgwriter, basename: Incomplete | None = None, image_dpi: int = 72, *, metadata: Incomplete | None = None) -> None: ...
    def _get_clippath_id(self, clippath):
        """
        Returns a stable and unique identifier for the *clippath* argument
        object within the current rendering context.

        This allows plots that include custom clip paths to produce identical
        SVG output on each render, provided that the :rc:`svg.hashsalt` config
        setting and the ``SOURCE_DATE_EPOCH`` build-time environment variable
        are set to fixed values.
        """
    def finalize(self) -> None: ...
    def _write_metadata(self, metadata): ...
    def _write_default_style(self) -> None: ...
    def _make_id(self, type, content): ...
    def _make_flip_transform(self, transform): ...
    def _get_hatch(self, gc, rgbFace):
        """
        Create a new hatch pattern
        """
    def _write_hatches(self) -> None: ...
    def _get_style_dict(self, gc, rgbFace):
        """Generate a style string from the GraphicsContext and rgbFace."""
    def _get_style(self, gc, rgbFace): ...
    def _get_clip_attrs(self, gc): ...
    def _write_clips(self) -> None: ...
    def open_group(self, s, gid: Incomplete | None = None) -> None: ...
    def close_group(self, s) -> None: ...
    def option_image_nocomposite(self): ...
    def _convert_path(self, path, transform: Incomplete | None = None, clip: Incomplete | None = None, simplify: Incomplete | None = None, sketch: Incomplete | None = None): ...
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace: Incomplete | None = None) -> None: ...
    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position): ...
    def _draw_gouraud_triangle(self, transformed_points, colors) -> None: ...
    def draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform) -> None: ...
    def option_scale_image(self): ...
    def get_image_magnification(self): ...
    def draw_image(self, gc, x, y, im, transform: Incomplete | None = None) -> None: ...
    def _update_glyph_map_defs(self, glyph_map_new) -> None:
        """
        Emit definitions for not-yet-defined glyphs, and record them as having
        been defined.
        """
    def _adjust_char_id(self, char_id): ...
    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath, mtext: Incomplete | None = None) -> None: ...
    def _draw_text_as_text(self, gc, x, y, s, prop, angle, ismath, mtext: Incomplete | None = None): ...
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None) -> None: ...
    def flipy(self): ...
    def get_canvas_width_height(self): ...
    def get_text_width_height_descent(self, s, prop, ismath): ...

class FigureCanvasSVG(FigureCanvasBase):
    filetypes: Incomplete
    fixed_dpi: int
    def print_svg(self, filename, *, bbox_inches_restore: Incomplete | None = None, metadata: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict[str, Any], optional
            Metadata in the SVG file defined as key-value pairs of strings,
            datetimes, or lists of strings, e.g., ``{'Creator': 'My software',
            'Contributor': ['Me', 'My Friend'], 'Title': 'Awesome'}``.

            The standard keys and their value types are:

            * *str*: ``'Coverage'``, ``'Description'``, ``'Format'``,
              ``'Identifier'``, ``'Language'``, ``'Relation'``, ``'Source'``,
              ``'Title'``, and ``'Type'``.
            * *str* or *list of str*: ``'Contributor'``, ``'Creator'``,
              ``'Keywords'``, ``'Publisher'``, and ``'Rights'``.
            * *str*, *date*, *datetime*, or *tuple* of same: ``'Date'``. If a
              non-*str*, then it will be formatted as ISO 8601.

            Values have been predefined for ``'Creator'``, ``'Date'``,
            ``'Format'``, and ``'Type'``. They can be removed by setting them
            to `None`.

            Information is encoded as `Dublin Core Metadata`__.

            .. _DC: https://www.dublincore.org/specifications/dublin-core/

            __ DC_
        """
    def print_svgz(self, filename, **kwargs): ...
    def get_default_filetype(self): ...
    def draw(self): ...
FigureManagerSVG = FigureManagerBase
svgProlog: str

class _BackendSVG(_Backend):
    backend_version: Incomplete
    FigureCanvas = FigureCanvasSVG

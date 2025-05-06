from . import _backend_pdf_ps as _backend_pdf_ps
from _typeshed import Incomplete
from enum import Enum
from matplotlib import _api as _api, _path as _path, _text_helpers as _text_helpers, _type1font as _type1font, cbook as cbook, dviread as dviread
from matplotlib._afm import AFM as AFM
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, GraphicsContextBase as GraphicsContextBase, RendererBase as RendererBase, _Backend as _Backend
from matplotlib.backends.backend_mixed import MixedModeRenderer as MixedModeRenderer
from matplotlib.dates import UTC as UTC
from matplotlib.figure import Figure as Figure
from matplotlib.font_manager import get_font as get_font
from matplotlib.ft2font import FT2Font as FT2Font, FaceFlags as FaceFlags, Kerning as Kerning, LoadFlags as LoadFlags, StyleFlags as StyleFlags
from matplotlib.path import Path as Path
from matplotlib.transforms import Affine2D as Affine2D, BboxBase as BboxBase

_log: Incomplete

def _fill(strings, linelen: int = 75):
    """
    Make one string from sequence of strings, with whitespace in between.

    The whitespace is chosen to form lines of at most *linelen* characters,
    if possible.
    """
def _create_pdf_info_dict(backend, metadata):
    """
    Create a PDF infoDict based on user-supplied metadata.

    A default ``Creator``, ``Producer``, and ``CreationDate`` are added, though
    the user metadata may override it. The date may be the current time, or a
    time set by the ``SOURCE_DATE_EPOCH`` environment variable.

    Metadata is verified to have the correct keys and their expected types. Any
    unknown keys/types will raise a warning.

    Parameters
    ----------
    backend : str
        The name of the backend to use in the Producer value.

    metadata : dict[str, Union[str, datetime, Name]]
        A dictionary of metadata supplied by the user with information
        following the PDF specification, also defined in
        `~.backend_pdf.PdfPages` below.

        If any value is *None*, then the key will be removed. This can be used
        to remove any pre-defined values.

    Returns
    -------
    dict[str, Union[str, datetime, Name]]
        A validated dictionary of metadata.
    """
def _datetime_to_pdf(d):
    """
    Convert a datetime to a PDF string representing it.

    Used for PDF and PGF.
    """
def _calculate_quad_point_coordinates(x, y, width, height, angle: int = 0):
    """
    Calculate the coordinates of rectangle when rotated by angle around x, y
    """
def _get_coordinates_of_block(x, y, width, height, angle: int = 0):
    """
    Get the coordinates of rotated rectangle and rectangle that covers the
    rotated rectangle.
    """
def _get_link_annotation(gc, x, y, width, height, angle: int = 0):
    """
    Create a link annotation object for embedding URLs.
    """

_str_escapes: Incomplete

def pdfRepr(obj):
    """Map Python objects to PDF syntax."""
def _font_supports_glyph(fonttype, glyph):
    """
    Returns True if the font is able to provide codepoint *glyph* in a PDF.

    For a Type 3 font, this method returns True only for single-byte
    characters. For Type 42 fonts this method return True if the character is
    from the Basic Multilingual Plane.
    """

class Reference:
    """
    PDF reference object.

    Use PdfFile.reserveObject() to create References.
    """
    id: Incomplete
    def __init__(self, id) -> None: ...
    def __repr__(self) -> str: ...
    def pdfRepr(self): ...
    def write(self, contents, file) -> None: ...

class Name:
    """PDF name object."""
    __slots__: Incomplete
    _hexify: Incomplete
    name: Incomplete
    def __init__(self, name) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other): ...
    def __lt__(self, other): ...
    def __hash__(self): ...
    def pdfRepr(self): ...

class Verbatim:
    """Store verbatim PDF command content for later inclusion in the stream."""
    _x: Incomplete
    def __init__(self, x) -> None: ...
    def pdfRepr(self): ...

class Op(Enum):
    """PDF operators (not an exhaustive list)."""
    close_fill_stroke = b'b'
    fill_stroke = b'B'
    fill = b'f'
    closepath = b'h'
    close_stroke = b's'
    stroke = b'S'
    endpath = b'n'
    begin_text = b'BT'
    end_text = b'ET'
    curveto = b'c'
    rectangle = b're'
    lineto = b'l'
    moveto = b'm'
    concat_matrix = b'cm'
    use_xobject = b'Do'
    setgray_stroke = b'G'
    setgray_nonstroke = b'g'
    setrgb_stroke = b'RG'
    setrgb_nonstroke = b'rg'
    setcolorspace_stroke = b'CS'
    setcolorspace_nonstroke = b'cs'
    setcolor_stroke = b'SCN'
    setcolor_nonstroke = b'scn'
    setdash = b'd'
    setlinejoin = b'j'
    setlinecap = b'J'
    setgstate = b'gs'
    gsave = b'q'
    grestore = b'Q'
    textpos = b'Td'
    selectfont = b'Tf'
    textmatrix = b'Tm'
    show = b'Tj'
    showkern = b'TJ'
    setlinewidth = b'w'
    clip = b'W'
    shading = b'sh'
    def pdfRepr(self): ...
    @classmethod
    def paint_path(cls, fill, stroke):
        """
        Return the PDF operator to paint a path.

        Parameters
        ----------
        fill : bool
            Fill the path with the fill color.
        stroke : bool
            Stroke the outline of the path with the line color.
        """

class Stream:
    """
    PDF stream object.

    This has no pdfRepr method. Instead, call begin(), then output the
    contents of the stream by calling write(), and finally call end().
    """
    __slots__: Incomplete
    id: Incomplete
    len: Incomplete
    pdfFile: Incomplete
    file: Incomplete
    compressobj: Incomplete
    extra: Incomplete
    pos: Incomplete
    def __init__(self, id, len, file, extra: Incomplete | None = None, png: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        id : int
            Object id of the stream.
        len : Reference or None
            An unused Reference object for the length of the stream;
            None means to use a memory buffer so the length can be inlined.
        file : PdfFile
            The underlying object to write the stream to.
        extra : dict from Name to anything, or None
            Extra key-value pairs to include in the stream header.
        png : dict or None
            If the data is already png encoded, the decode parameters.
        """
    def _writeHeader(self) -> None: ...
    def end(self) -> None:
        """Finalize stream."""
    def write(self, data) -> None:
        """Write some data on the stream."""
    def _flush(self) -> None:
        """Flush the compression object."""

def _get_pdf_charprocs(font_path, glyph_ids): ...

class PdfFile:
    """PDF file object."""
    _object_seq: Incomplete
    xrefTable: Incomplete
    passed_in_file_object: bool
    original_file_like: Incomplete
    tell_base: int
    fh: Incomplete
    currentstream: Incomplete
    rootObject: Incomplete
    pagesObject: Incomplete
    pageList: Incomplete
    fontObject: Incomplete
    _extGStateObject: Incomplete
    hatchObject: Incomplete
    gouraudObject: Incomplete
    XObjectObject: Incomplete
    resourceObject: Incomplete
    infoDict: Incomplete
    fontNames: Incomplete
    _internal_font_seq: Incomplete
    dviFontInfo: Incomplete
    type1Descriptors: Incomplete
    _character_tracker: Incomplete
    alphaStates: Incomplete
    _alpha_state_seq: Incomplete
    _soft_mask_states: Incomplete
    _soft_mask_seq: Incomplete
    _soft_mask_groups: Incomplete
    _hatch_patterns: Incomplete
    _hatch_pattern_seq: Incomplete
    gouraudTriangles: Incomplete
    _images: Incomplete
    _image_seq: Incomplete
    markers: Incomplete
    multi_byte_charprocs: Incomplete
    paths: Incomplete
    _annotations: Incomplete
    pageAnnotations: Incomplete
    def __init__(self, filename, metadata: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict from strings to strings and dates
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.
        """
    def newPage(self, width, height) -> None: ...
    def newTextnote(self, text, positionRect=[-100, -100, 0, 0]) -> None: ...
    def _get_subsetted_psname(self, ps_name, charmap): ...
    def finalize(self) -> None:
        """Write out the various deferred objects and the pdf end matter."""
    def close(self) -> None:
        """Flush all buffers and free all resources."""
    def write(self, data) -> None: ...
    def output(self, *data) -> None: ...
    def beginStream(self, id, len, extra: Incomplete | None = None, png: Incomplete | None = None) -> None: ...
    def endStream(self) -> None: ...
    def outputStream(self, ref, data, *, extra: Incomplete | None = None) -> None: ...
    def _write_annotations(self) -> None: ...
    def fontName(self, fontprop):
        """
        Select a font based on fontprop and return a name suitable for
        Op.selectfont. If fontprop is a string, it will be interpreted
        as the filename of the font.
        """
    def dviFontName(self, dvifont):
        """
        Given a dvi font object, return a name suitable for Op.selectfont.
        This registers the font information in ``self.dviFontInfo`` if not yet
        registered.
        """
    def writeFonts(self) -> None: ...
    def _write_afm_font(self, filename): ...
    def _embedTeXFont(self, fontinfo): ...
    def createType1Descriptor(self, t1font, fontfile): ...
    def _get_xobject_glyph_name(self, filename, glyph_name): ...
    _identityToUnicodeCMap: bytes
    def embedTTF(self, filename, characters):
        """Embed the TTF font from the named file into the document."""
    def alphaState(self, alpha):
        """Return name of an ExtGState that sets alpha to the given value."""
    def _soft_mask_state(self, smask):
        """
        Return an ExtGState that sets the soft mask to the given shading.

        Parameters
        ----------
        smask : Reference
            Reference to a shading in DeviceGray color space, whose luminosity
            is to be used as the alpha channel.

        Returns
        -------
        Name
        """
    def writeExtGSTates(self) -> None: ...
    def _write_soft_mask_groups(self) -> None: ...
    def hatchPattern(self, hatch_style): ...
    hatchPatterns: Incomplete
    def writeHatches(self) -> None: ...
    def addGouraudTriangles(self, points, colors):
        """
        Add a Gouraud triangle shading.

        Parameters
        ----------
        points : np.ndarray
            Triangle vertices, shape (n, 3, 2)
            where n = number of triangles, 3 = vertices, 2 = x, y.
        colors : np.ndarray
            Vertex colors, shape (n, 3, 1) or (n, 3, 4)
            as with points, but last dimension is either (gray,)
            or (r, g, b, alpha).

        Returns
        -------
        Name, Reference
        """
    def writeGouraudTriangles(self) -> None: ...
    def imageObject(self, image):
        """Return name of an image XObject representing the given image."""
    def _unpack(self, im):
        """
        Unpack image array *im* into ``(data, alpha)``, which have shape
        ``(height, width, 3)`` (RGB) or ``(height, width, 1)`` (grayscale or
        alpha), except that alpha is None if the image is fully opaque.
        """
    def _writePng(self, img):
        """
        Write the image *img* into the pdf file using png
        predictors with Flate compression.
        """
    def _writeImg(self, data, id, smask: Incomplete | None = None) -> None:
        """
        Write the image *data*, of shape ``(height, width, 1)`` (grayscale) or
        ``(height, width, 3)`` (RGB), as pdf object *id* and with the soft mask
        (alpha channel) *smask*, which should be either None or a ``(height,
        width, 1)`` array.
        """
    def writeImages(self) -> None: ...
    def markerObject(self, path, trans, fill, stroke, lw, joinstyle, capstyle):
        """Return name of a marker XObject representing the given path."""
    def writeMarkers(self) -> None: ...
    def pathCollectionObject(self, gc, path, trans, padding, filled, stroked): ...
    def writePathCollectionTemplates(self) -> None: ...
    @staticmethod
    def pathOperations(path, transform, clip: Incomplete | None = None, simplify: Incomplete | None = None, sketch: Incomplete | None = None): ...
    def writePath(self, path, transform, clip: bool = False, sketch: Incomplete | None = None) -> None: ...
    def reserveObject(self, name: str = ''):
        """
        Reserve an ID for an indirect object.

        The name is used for debugging in case we forget to print out
        the object with writeObject.
        """
    def recordXref(self, id) -> None: ...
    def writeObject(self, object, contents) -> None: ...
    startxref: Incomplete
    def writeXref(self) -> None:
        """Write out the xref table."""
    infoObject: Incomplete
    def writeInfoDict(self) -> None:
        """Write out the info dictionary, checking it for good form"""
    def writeTrailer(self) -> None:
        """Write out the PDF trailer."""

class RendererPdf(_backend_pdf_ps.RendererPDFPSBase):
    _afm_font_dir: Incomplete
    _use_afm_rc_name: str
    file: Incomplete
    gc: Incomplete
    image_dpi: Incomplete
    def __init__(self, file, image_dpi, height, width) -> None: ...
    def finalize(self) -> None: ...
    def check_gc(self, gc, fillcolor: Incomplete | None = None) -> None: ...
    def get_image_magnification(self): ...
    def draw_image(self, gc, x, y, im, transform: Incomplete | None = None) -> None: ...
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position): ...
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace: Incomplete | None = None) -> None: ...
    def draw_gouraud_triangles(self, gc, points, colors, trans) -> None: ...
    def _setup_textpos(self, x, y, angle, oldx: int = 0, oldy: int = 0, oldangle: int = 0) -> None: ...
    def draw_mathtext(self, gc, x, y, s, prop, angle) -> None: ...
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext: Incomplete | None = None) -> None: ...
    def encode_string(self, s, fonttype): ...
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None): ...
    def _draw_xobject_glyph(self, font, fontsize, glyph_idx, x, y) -> None:
        """Draw a multibyte character from a Type 3 font as an XObject."""
    def new_gc(self): ...

class GraphicsContextPdf(GraphicsContextBase):
    _fillcolor: Incomplete
    _effective_alphas: Incomplete
    file: Incomplete
    parent: Incomplete
    def __init__(self, file) -> None: ...
    def __repr__(self) -> str: ...
    def stroke(self):
        """
        Predicate: does the path need to be stroked (its outline drawn)?
        This tests for the various conditions that disable stroking
        the path, in which case it would presumably be filled.
        """
    def fill(self, *args):
        """
        Predicate: does the path need to be filled?

        An optional argument can be used to specify an alternative
        _fillcolor, as needed by RendererPdf.draw_markers.
        """
    def paint(self):
        """
        Return the appropriate pdf operator to cause the path to be
        stroked, filled, or both.
        """
    capstyles: Incomplete
    joinstyles: Incomplete
    def capstyle_cmd(self, style): ...
    def joinstyle_cmd(self, style): ...
    def linewidth_cmd(self, width): ...
    def dash_cmd(self, dashes): ...
    def alpha_cmd(self, alpha, forced, effective_alphas): ...
    def hatch_cmd(self, hatch, hatch_color, hatch_linewidth): ...
    def rgb_cmd(self, rgb): ...
    def fillcolor_cmd(self, rgb): ...
    def push(self): ...
    def pop(self): ...
    def clip_cmd(self, cliprect, clippath):
        """Set clip rectangle. Calls `.pop()` and `.push()`."""
    commands: Incomplete
    def delta(self, other):
        """
        Copy properties of other into self and return PDF commands
        needed to transform *self* into *other*.
        """
    def copy_properties(self, other) -> None:
        """
        Copy properties of other into self.
        """
    def finalize(self):
        """
        Make sure every pushed graphics state is popped.
        """

class PdfPages:
    """
    A multi-page PDF file.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Initialize:
    >>> with PdfPages('foo.pdf') as pdf:
    ...     # As many times as you like, create a figure fig and save it:
    ...     fig = plt.figure()
    ...     pdf.savefig(fig)
    ...     # When no figure is specified the current figure is saved
    ...     pdf.savefig()

    Notes
    -----
    In reality `PdfPages` is a thin wrapper around `PdfFile`, in order to avoid
    confusion when using `~.pyplot.savefig` and forgetting the format argument.
    """
    _filename: Incomplete
    _metadata: Incomplete
    _file: Incomplete
    def __init__(self, filename, keep_empty: Incomplete | None = None, metadata: Incomplete | None = None) -> None:
        """
        Create a new PdfPages object.

        Parameters
        ----------
        filename : str or path-like or file-like
            Plots using `PdfPages.savefig` will be written to a file at this location.
            The file is opened when a figure is saved for the first time (overwriting
            any older file with the same name).

        metadata : dict, optional
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.
        """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def _ensure_file(self): ...
    def close(self) -> None:
        """
        Finalize this object, making the underlying file a complete
        PDF file.
        """
    def infodict(self):
        """
        Return a modifiable information dictionary object
        (see PDF reference section 10.2.1 'Document Information
        Dictionary').
        """
    def savefig(self, figure: Incomplete | None = None, **kwargs) -> None:
        """
        Save a `.Figure` to this file as a new page.

        Any other keyword arguments are passed to `~.Figure.savefig`.

        Parameters
        ----------
        figure : `.Figure` or int, default: the active figure
            The figure, or index of the figure, that is saved to the file.
        """
    def get_pagecount(self):
        """Return the current number of pages in the multipage pdf file."""
    def attach_note(self, text, positionRect=[-100, -100, 0, 0]) -> None:
        """
        Add a new text note to the page to be saved next. The optional
        positionRect specifies the position of the new note on the
        page. It is outside the page per default to make sure it is
        invisible on printouts.
        """

class FigureCanvasPdf(FigureCanvasBase):
    fixed_dpi: int
    filetypes: Incomplete
    def get_default_filetype(self): ...
    def print_pdf(self, filename, *, bbox_inches_restore: Incomplete | None = None, metadata: Incomplete | None = None) -> None: ...
    def draw(self): ...
FigureManagerPdf = FigureManagerBase

class _BackendPdf(_Backend):
    FigureCanvas = FigureCanvasPdf

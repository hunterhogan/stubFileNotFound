from . import _backend_pdf_ps as _backend_pdf_ps
from _typeshed import Incomplete
from enum import Enum
from matplotlib import _api as _api, _path as _path, _text_helpers as _text_helpers, cbook as cbook
from matplotlib._afm import AFM as AFM
from matplotlib._mathtext_data import uni2type1 as uni2type1
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, RendererBase as RendererBase, _Backend as _Backend
from matplotlib.backends.backend_mixed import MixedModeRenderer as MixedModeRenderer
from matplotlib.cbook import file_requires_unicode as file_requires_unicode, is_writable_file_like as is_writable_file_like
from matplotlib.font_manager import get_font as get_font
from matplotlib.ft2font import LoadFlags as LoadFlags
from matplotlib.path import Path as Path
from matplotlib.texmanager import TexManager as TexManager
from matplotlib.transforms import Affine2D as Affine2D

_log: Incomplete
debugPS: bool
papersize: Incomplete

def _nums_to_str(*args, sep: str = ' '): ...
def _move_path_to_path_or_stream(src, dst) -> None:
    """
    Move the contents of file at *src* to path-or-filelike *dst*.

    If *dst* is a path, the metadata of *src* are *not* copied.
    """
def _font_to_ps_type3(font_path, chars):
    """
    Subset *chars* from the font at *font_path* into a Type 3 font.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.

    Returns
    -------
    str
        The string representation of a Type 3 font, which can be included
        verbatim into a PostScript file.
    """
def _font_to_ps_type42(font_path, chars, fh) -> None:
    """
    Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.
    fh : file-like
        Where to write the font.
    """
def _serialize_type42(font, subset, fontdata):
    """
    Output a PostScript Type-42 format representation of font

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        The original font object
    subset : fontTools.ttLib.ttFont.TTFont
        The subset font object
    fontdata : bytes
        The raw font data in TTF format

    Returns
    -------
    str
        The Type-42 formatted font
    """
def _version_and_breakpoints(loca, fontdata):
    """
    Read the version number of the font and determine sfnts breakpoints.

    When a TrueType font file is written as a Type 42 font, it has to be
    broken into substrings of at most 65535 bytes. These substrings must
    begin at font table boundaries or glyph boundaries in the glyf table.
    This function determines all possible breakpoints and it is the caller's
    responsibility to do the splitting.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    loca : fontTools.ttLib._l_o_c_a.table__l_o_c_a or None
        The loca table of the font if available
    fontdata : bytes
        The raw data of the font

    Returns
    -------
    version : tuple[int, int]
        A 2-tuple of the major version number and minor version number.
    breakpoints : list[int]
        The breakpoints is a sorted list of offsets into fontdata; if loca is not
        available, just the table boundaries.
    """
def _bounds(font):
    """
    Compute the font bounding box, as if all glyphs were written
    at the same start position.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        The font

    Returns
    -------
    tuple
        (xMin, yMin, xMax, yMax) of the combined bounding box
        of all the glyphs in the font
    """
def _generate_charstrings(font):
    """
    Transform font glyphs into CharStrings

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        The font

    Returns
    -------
    str
        A definition of the CharStrings dictionary in PostScript
    """
def _generate_sfnts(fontdata, font, breakpoints):
    """
    Transform font data into PostScript sfnts format.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    fontdata : bytes
        The raw data of the font
    font : fontTools.ttLib.ttFont.TTFont
        The fontTools font object
    breakpoints : list
        Sorted offsets of possible breakpoints

    Returns
    -------
    str
        The sfnts array for the font definition, consisting
        of hex-encoded strings in PostScript format
    """
def _log_if_debug_on(meth):
    """
    Wrap `RendererPS` method *meth* to emit a PS comment with the method name,
    if the global flag `debugPS` is set.
    """

class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """
    _afm_font_dir: Incomplete
    _use_afm_rc_name: str
    _pswriter: Incomplete
    textcnt: int
    psfrag: Incomplete
    imagedpi: Incomplete
    color: Incomplete
    linewidth: Incomplete
    linejoin: Incomplete
    linecap: Incomplete
    linedash: Incomplete
    fontname: Incomplete
    fontsize: Incomplete
    _hatches: Incomplete
    image_magnification: Incomplete
    _clip_paths: Incomplete
    _path_collection_id: int
    _character_tracker: Incomplete
    _logwarn_once: Incomplete
    def __init__(self, width, height, pswriter, imagedpi: int = 72) -> None: ...
    def _is_transparent(self, rgb_or_rgba): ...
    def set_color(self, r, g, b, store: bool = True) -> None: ...
    def set_linewidth(self, linewidth, store: bool = True) -> None: ...
    @staticmethod
    def _linejoin_cmd(linejoin): ...
    def set_linejoin(self, linejoin, store: bool = True) -> None: ...
    @staticmethod
    def _linecap_cmd(linecap): ...
    def set_linecap(self, linecap, store: bool = True) -> None: ...
    def set_linedash(self, offset, seq, store: bool = True) -> None: ...
    def set_font(self, fontname, fontsize, store: bool = True) -> None: ...
    def create_hatch(self, hatch, linewidth): ...
    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
    def _convert_path(self, path, transform, clip: bool = False, simplify: Incomplete | None = None): ...
    def _get_clip_cmd(self, gc): ...
    def draw_image(self, gc, x, y, im, transform: Incomplete | None = None) -> None: ...
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace: Incomplete | None = None) -> None: ...
    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position): ...
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext: Incomplete | None = None) -> None: ...
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None): ...
    def draw_mathtext(self, gc, x, y, s, prop, angle) -> None:
        """Draw the math text using matplotlib.mathtext."""
    def draw_gouraud_triangles(self, gc, points, colors, trans) -> None: ...
    def _draw_ps(self, ps, gc, rgbFace, *, fill: bool = True, stroke: bool = True) -> None:
        """
        Emit the PostScript snippet *ps* with all the attributes from *gc*
        applied.  *ps* must consist of PostScript commands to construct a path.

        The *fill* and/or *stroke* kwargs can be set to False if the *ps*
        string already includes filling and/or stroking, in which case
        `_draw_ps` is just supplying properties and clipping.
        """

class _Orientation(Enum):
    portrait = ...
    landscape = ...
    def swap_if_landscape(self, shape): ...

class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi: int
    filetypes: Incomplete
    def get_default_filetype(self): ...
    def _print_ps(self, fmt, outfile, *, metadata: Incomplete | None = None, papertype: Incomplete | None = None, orientation: str = 'portrait', bbox_inches_restore: Incomplete | None = None, **kwargs) -> None: ...
    _pswriter: Incomplete
    def _print_figure(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore: Incomplete | None = None) -> None:
        """
        Render the figure to a filesystem path or a file-like object.

        Parameters are as for `.print_figure`, except that *dsc_comments* is a
        string containing Document Structuring Convention comments,
        generated from the *metadata* parameter to `.print_figure`.
        """
    def _print_figure_tex(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore: Incomplete | None = None) -> None:
        """
        If :rc:`text.usetex` is True, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.

        The rest of the behavior is as for `._print_figure`.
        """
    print_ps: Incomplete
    print_eps: Incomplete
    def draw(self): ...

def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
def _try_distill(func, tmppath, *args, **kwargs) -> None: ...
def gs_distill(tmpfile, eps: bool = False, ptype: str = 'letter', bbox: Incomplete | None = None, rotated: bool = False) -> None:
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """
def xpdf_distill(tmpfile, eps: bool = False, ptype: str = 'letter', bbox: Incomplete | None = None, rotated: bool = False) -> None:
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
def get_bbox_header(lbrt, rotated: bool = False):
    """
    Return a postscript header string for the given bbox lbrt=(l, b, r, t).
    Optionally, return rotate command.
    """
def _get_bbox_header(lbrt):
    """Return a PostScript header string for bounding box *lbrt*=(l, b, r, t)."""
def _get_rotate_command(lbrt):
    """Return a PostScript 90° rotation command for bounding box *lbrt*=(l, b, r, t)."""
def pstoeps(tmpfile, bbox: Incomplete | None = None, rotated: bool = False) -> None:
    """
    Convert the postscript to encapsulated postscript.  The bbox of
    the eps file will be replaced with the given *bbox* argument. If
    None, original bbox will be used.
    """
FigureManagerPS = FigureManagerBase
_psDefs: Incomplete

class _BackendPS(_Backend):
    backend_version: str
    FigureCanvas = FigureCanvasPS

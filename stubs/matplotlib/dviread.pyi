from _typeshed import Incomplete
from matplotlib import _api as _api, cbook as cbook
from typing import NamedTuple

_log: Incomplete
_dvistate: Incomplete

class Page(NamedTuple):
    text: Incomplete
    boxes: Incomplete
    height: Incomplete
    width: Incomplete
    descent: Incomplete

class Box(NamedTuple):
    x: Incomplete
    y: Incomplete
    height: Incomplete
    width: Incomplete

class Text(NamedTuple('Text', [('x', Incomplete), ('y', Incomplete), ('font', Incomplete), ('glyph', Incomplete), ('width', Incomplete)])):
    """
    A glyph in the dvi file.

    The *x* and *y* attributes directly position the glyph.  The *font*,
    *glyph*, and *width* attributes are kept public for back-compatibility,
    but users wanting to draw the glyph themselves are encouraged to instead
    load the font specified by `font_path` at `font_size`, warp it with the
    effects specified by `font_effects`, and load the glyph specified by
    `glyph_name_or_index`.
    """
    def _get_pdftexmap_entry(self): ...
    @property
    def font_path(self):
        """The `~pathlib.Path` to the font for this glyph."""
    @property
    def font_size(self):
        """The font size."""
    @property
    def font_effects(self):
        '''
        The "font effects" dict for this glyph.

        This dict contains the values for this glyph of SlantFont and
        ExtendFont (if any), read off :file:`pdftex.map`.
        '''
    @property
    def glyph_name_or_index(self):
        '''
        Either the glyph name or the native charmap glyph index.

        If :file:`pdftex.map` specifies an encoding for this glyph\'s font, that
        is a mapping of glyph indices to Adobe glyph names; use it to convert
        dvi indices to glyph names.  Callers can then convert glyph names to
        glyph indices (with FT_Get_Name_Index/get_name_index), and load the
        glyph using FT_Load_Glyph/load_glyph.

        If :file:`pdftex.map` specifies no encoding, the indices directly map
        to the font\'s "native" charmap; glyphs should directly load using
        FT_Load_Char/load_char after selecting the native charmap.
        '''

_arg_mapping: Incomplete

def _dispatch(table, min, max: Incomplete | None = None, state: Incomplete | None = None, args=('raw',)):
    """
    Decorator for dispatch by opcode. Sets the values in *table*
    from *min* to *max* to this method, adds a check that the Dvi state
    matches *state* if not None, reads arguments from the file according
    to *args*.

    Parameters
    ----------
    table : dict[int, callable]
        The dispatch table to be filled in.

    min, max : int
        Range of opcodes that calls the registered function; *max* defaults to
        *min*.

    state : _dvistate, optional
        State of the Dvi object in which these opcodes are allowed.

    args : list[str], default: ['raw']
        Sequence of argument specifications:

        - 'raw': opcode minus minimum
        - 'u1': read one unsigned byte
        - 'u4': read four bytes, treat as an unsigned number
        - 's4': read four bytes, treat as a signed number
        - 'slen': read (opcode - minimum) bytes, treat as signed
        - 'slen1': read (opcode - minimum + 1) bytes, treat as signed
        - 'ulen1': read (opcode - minimum + 1) bytes, treat as unsigned
        - 'olen1': read (opcode - minimum + 1) bytes, treat as unsigned
          if under four bytes, signed if four bytes
    """

class Dvi:
    '''
    A reader for a dvi ("device-independent") file, as produced by TeX.

    The current implementation can only iterate through pages in order,
    and does not even attempt to verify the postamble.

    This class can be used as a context manager to close the underlying
    file upon exit. Pages can be read via iteration. Here is an overly
    simple way to extract text without trying to detect whitespace::

        >>> with matplotlib.dviread.Dvi(\'input.dvi\', 72) as dvi:
        ...     for page in dvi:
        ...         print(\'\'.join(chr(t.glyph) for t in page.text))
    '''
    _dtable: Incomplete
    _dispatch: Incomplete
    file: Incomplete
    dpi: Incomplete
    fonts: Incomplete
    state: Incomplete
    _missing_font: Incomplete
    def __init__(self, filename, dpi) -> None:
        """
        Read the data from the file named *filename* and convert
        TeX's internal units to units of *dpi* per inch.
        *dpi* only sets the units and does not limit the resolution.
        Use None to return TeX's internal units.
        """
    def __enter__(self):
        """Context manager enter method, does nothing."""
    def __exit__(self, etype: type[BaseException] | None, evalue: BaseException | None, etrace: types.TracebackType | None) -> None:
        """
        Context manager exit method, closes the underlying file if it is open.
        """
    def __iter__(self):
        """
        Iterate through the pages of the file.

        Yields
        ------
        Page
            Details of all the text and box objects on the page.
            The Page tuple contains lists of Text and Box tuples and
            the page dimensions, and the Text and Box tuples contain
            coordinates transformed into a standard Cartesian
            coordinate system at the dpi value given when initializing.
            The coordinates are floating point numbers, but otherwise
            precision is not lost and coordinate values are not clipped to
            integers.
        """
    def close(self) -> None:
        """Close the underlying file if it is open."""
    _baseline_v: Incomplete
    def _output(self):
        """
        Output the text and boxes belonging to the most recent page.
        page = dvi._output()
        """
    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
    def _read_arg(self, nbytes, signed: bool = False):
        """
        Read and return a big-endian integer *nbytes* long.
        Signedness is determined by the *signed* keyword.
        """
    def _set_char_immediate(self, char) -> None: ...
    def _set_char(self, char) -> None: ...
    def _set_rule(self, a, b) -> None: ...
    def _put_char(self, char) -> None: ...
    def _put_char_real(self, char) -> None: ...
    def _put_rule(self, a, b) -> None: ...
    def _put_rule_real(self, a, b) -> None: ...
    def _nop(self, _) -> None: ...
    h: int
    stack: Incomplete
    text: Incomplete
    boxes: Incomplete
    def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p) -> None: ...
    def _eop(self, _) -> None: ...
    def _push(self, _) -> None: ...
    def _pop(self, _) -> None: ...
    def _right(self, b) -> None: ...
    w: Incomplete
    def _right_w(self, new_w) -> None: ...
    x: Incomplete
    def _right_x(self, new_x) -> None: ...
    def _down(self, a) -> None: ...
    y: Incomplete
    def _down_y(self, new_y) -> None: ...
    z: Incomplete
    def _down_z(self, new_z) -> None: ...
    f: Incomplete
    def _fnt_num_immediate(self, k) -> None: ...
    def _fnt_num(self, new_f) -> None: ...
    def _xxx(self, datalen) -> None: ...
    def _fnt_def(self, k, c, s, d, a, l) -> None: ...
    def _fnt_def_real(self, k, c, s, d, a, l) -> None: ...
    def _pre(self, i, num, den, mag, k) -> None: ...
    def _post(self, _) -> None: ...
    def _post_post(self, _) -> None: ...
    def _malformed(self, offset) -> None: ...

class DviFont:
    '''
    Encapsulation of a font that a DVI file can refer to.

    This class holds a font\'s texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    Parameters
    ----------
    scale : float
        Factor by which the font is scaled from its natural size.
    tfm : Tfm
        TeX font metrics for this font
    texname : bytes
       Name of the font as used internally by TeX and friends, as an ASCII
       bytestring.  This is usually very different from any external font
       names; `PsfontsMap` can be used to find the external name of the font.
    vf : Vf
       A TeX "virtual font" file, or None if this font is not virtual.

    Attributes
    ----------
    texname : bytes
    size : float
       Size of the font in Adobe points, converted from the slightly
       smaller TeX points.
    widths : list
       Widths of glyphs in glyph-space units, typically 1/1000ths of
       the point size.

    '''
    __slots__: Incomplete
    _scale: Incomplete
    _tfm: Incomplete
    texname: Incomplete
    _vf: Incomplete
    size: Incomplete
    widths: Incomplete
    def __init__(self, scale, tfm, texname, vf) -> None: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __repr__(self) -> str: ...
    def _width_of(self, char):
        """Width of char in dvi units."""
    def _height_depth_of(self, char):
        """Height and depth of char in dvi units."""

class Vf(Dvi):
    """
    A virtual font (\\*.vf file) containing subroutines for dvi files.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    The virtual font format is a derivative of dvi:
    http://mirrors.ctan.org/info/knuth/virtual-fonts
    This class reuses some of the machinery of `Dvi`
    but replaces the `_read` loop and dispatch mechanism.

    Examples
    --------
    ::

        vf = Vf(filename)
        glyph = vf[code]
        glyph.text, glyph.boxes, glyph.width
    """
    _first_font: Incomplete
    _chars: Incomplete
    def __init__(self, filename) -> None: ...
    def __getitem__(self, code): ...
    state: Incomplete
    def _read(self) -> None:
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
    h: int
    stack: Incomplete
    text: Incomplete
    boxes: Incomplete
    f: Incomplete
    _missing_font: Incomplete
    def _init_packet(self, pl): ...
    def _finalize_packet(self, packet_char, packet_width) -> None: ...
    def _pre(self, i, x, cs, ds) -> None: ...

def _mul2012(num1, num2):
    """Multiply two numbers in 20.12 fixed point format."""

class Tfm:
    """
    A TeX Font Metric file.

    This implementation covers only the bare minimum needed by the Dvi class.

    Parameters
    ----------
    filename : str or path-like

    Attributes
    ----------
    checksum : int
       Used for verifying against the dvi file.
    design_size : int
       Design size of the font (unknown units)
    width, height, depth : dict
       Dimensions of each character, need to be scaled by the factor
       specified in the dvi file. These are dicts because indexing may
       not start from 0.
    """
    __slots__: Incomplete
    width: Incomplete
    height: Incomplete
    depth: Incomplete
    def __init__(self, filename) -> None: ...

class PsFont(NamedTuple):
    texname: Incomplete
    psname: Incomplete
    effects: Incomplete
    encoding: Incomplete
    filename: Incomplete

class PsfontsMap:
    '''
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    For historical reasons, TeX knows many Type-1 fonts by different
    names than the outside world. (For one thing, the names have to
    fit in eight characters.) Also, TeX\'s native fonts are not Type-1
    but Metafont, which is nontrivial to convert to PostScript except
    as a bitmap. While high-quality conversions to Type-1 format exist
    and are shipped with modern TeX distributions, we need to know
    which Type-1 fonts are the counterparts of which native fonts. For
    these reasons a mapping is needed from internal font names to font
    file names.

    A texmf tree typically includes mapping files called e.g.
    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
    The file :file:`psfonts.map` is used by :program:`dvips`,
    :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
    by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
    the 35 PostScript fonts (i.e., have no filename for them, as in
    the Times-Bold example above), while the pdf-related files perhaps
    only avoid the "Base 14" pdf fonts. But the user may have
    configured these files differently.

    Examples
    --------
    >>> map = PsfontsMap(find_tex_file(\'pdftex.map\'))
    >>> entry = map[b\'ptmbo8r\']
    >>> entry.texname
    b\'ptmbo8r\'
    >>> entry.psname
    b\'Times-Bold\'
    >>> entry.encoding
    \'/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc\'
    >>> entry.effects
    {\'slant\': 0.16700000000000001}
    >>> entry.filename
    '''
    __slots__: Incomplete
    _filename: Incomplete
    _unparsed: Incomplete
    _parsed: Incomplete
    def __new__(cls, filename): ...
    def __getitem__(self, texname): ...
    def _parse_and_cache_line(self, line):
        """
        Parse a line in the font mapping file.

        The format is (partially) documented at
        http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
        https://tug.org/texinfohtml/dvips.html#psfonts_002emap
        Each line can have the following fields:

        - tfmname (first, only required field),
        - psname (defaults to tfmname, must come immediately after tfmname if
          present),
        - fontflags (integer, must come immediately after psname if present,
          ignored by us),
        - special (SlantFont and ExtendFont, only field that is double-quoted),
        - fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
          precedes a font, <[ always precedes an encoding, < can precede either
          but then an encoding file must have extension .enc; < and << also
          request different font subsetting behaviors but we ignore that; < can
          be separated from the filename by whitespace).

        special, fontfile, and encodingfile can appear in any order.
        """

def _parse_enc(path):
    """
    Parse a \\*.enc file referenced from a psfonts.map style file.

    The format supported by this function is a tiny subset of PostScript.

    Parameters
    ----------
    path : `os.PathLike`

    Returns
    -------
    list
        The nth entry of the list is the PostScript glyph name of the nth
        glyph.
    """

class _LuatexKpsewhich:
    _proc: Incomplete
    def __new__(cls): ...
    def _new_proc(self): ...
    def search(self, filename): ...

def find_tex_file(filename):
    """
    Find a file in the texmf tree using kpathsea_.

    The kpathsea library, provided by most existing TeX distributions, both
    on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
    luatex process if luatex is installed, or via kpsewhich otherwise.

    .. _kpathsea: https://www.tug.org/kpathsea/

    Parameters
    ----------
    filename : str or path-like

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
def _fontfile(cls, suffix, texname): ...

_tfmfile: Incomplete
_vffile: Incomplete

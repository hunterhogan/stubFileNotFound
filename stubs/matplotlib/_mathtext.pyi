import abc
import enum
import typing as T
from ._mathtext_data import latex_to_bakoma as latex_to_bakoma, stix_glyph_fixes as stix_glyph_fixes, stix_virtual_fonts as stix_virtual_fonts, tex2uni as tex2uni
from .font_manager import FontProperties as FontProperties, findfont as findfont, get_font as get_font
from .ft2font import FT2Font as FT2Font, FT2Image as FT2Image, Glyph as Glyph, Kerning as Kerning, LoadFlags as LoadFlags
from _typeshed import Incomplete
from pyparsing import ParseResults, ParserElement
from typing import NamedTuple

_log: Incomplete

def get_unicode_index(symbol: str) -> int:
    """
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single (Unicode) character, a TeX command (e.g. r'\\pi') or a Type1
        symbol name (e.g. 'phi').
    """

class VectorParse(NamedTuple):
    '''
    The namedtuple type returned by ``MathTextParser("path").parse(...)``.

    Attributes
    ----------
    width, height, depth : float
        The global metrics.
    glyphs : list
        The glyphs including their positions.
    rect : list
        The list of rectangles.
    '''
    width: float
    height: float
    depth: float
    glyphs: list[tuple[FT2Font, float, int, float, float]]
    rects: list[tuple[float, float, float, float]]

class RasterParse(NamedTuple):
    '''
    The namedtuple type returned by ``MathTextParser("agg").parse(...)``.

    Attributes
    ----------
    ox, oy : float
        The offsets are always zero.
    width, height, depth : float
        The global metrics.
    image : FT2Image
        A raster image.
    '''
    ox: float
    oy: float
    width: float
    height: float
    depth: float
    image: FT2Image

class Output:
    """
    Result of `ship`\\ping a box: lists of positioned glyphs and rectangles.

    This class is not exposed to end users, but converted to a `VectorParse` or
    a `RasterParse` by `.MathTextParser.parse`.
    """
    box: Incomplete
    glyphs: list[tuple[float, float, FontInfo]]
    rects: list[tuple[float, float, float, float]]
    def __init__(self, box: Box) -> None: ...
    def to_vector(self) -> VectorParse: ...
    def to_raster(self, *, antialiased: bool) -> RasterParse: ...

class FontMetrics(NamedTuple):
    '''
    Metrics of a font.

    Attributes
    ----------
    advance : float
        The advance distance (in points) of the glyph.
    height : float
        The height of the glyph in points.
    width : float
        The width of the glyph in points.
    xmin, xmax, ymin, ymax : float
        The ink rectangle of the glyph.
    iceberg : float
        The distance from the baseline to the top of the glyph. (This corresponds to
        TeX\'s definition of "height".)
    slanted : bool
        Whether the glyph should be considered as "slanted" (currently used for kerning
        sub/superscripts).
    '''
    advance: float
    height: float
    width: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    iceberg: float
    slanted: bool

class FontInfo(NamedTuple):
    font: FT2Font
    fontsize: float
    postscript_name: str
    metrics: FontMetrics
    num: int
    glyph: Glyph
    offset: float

class Fonts(abc.ABC):
    """
    An abstract base class for a system of fonts to use for mathtext.

    The class must be able to take symbol keys and font file names and
    return the character metrics.  It also delegates to a backend class
    to do the actual drawing.
    """
    default_font_prop: Incomplete
    load_glyph_flags: Incomplete
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags) -> None:
        """
        Parameters
        ----------
        default_font_prop : `~.font_manager.FontProperties`
            The default non-math font, or the base font for Unicode (generic)
            font rendering.
        load_glyph_flags : `.ft2font.LoadFlags`
            Flags passed to the glyph loader (e.g. ``FT_Load_Glyph`` and
            ``FT_Load_Char`` for FreeType-based fonts).
        """
    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float, font2: str, fontclass2: str, sym2: str, fontsize2: float, dpi: float) -> float:
        """
        Get the kerning distance for font between *sym1* and *sym2*.

        See `~.Fonts.get_metrics` for a detailed description of the parameters.
        """
    def _get_font(self, font: str) -> FT2Font: ...
    def _get_info(self, font: str, font_class: str, sym: str, fontsize: float, dpi: float) -> FontInfo: ...
    def get_metrics(self, font: str, font_class: str, sym: str, fontsize: float, dpi: float) -> FontMetrics:
        '''
        Parameters
        ----------
        font : str
            One of the TeX font names: "tt", "it", "rm", "cal", "sf", "bf",
            "default", "regular", "bb", "frak", "scr".  "default" and "regular"
            are synonyms and use the non-math font.
        font_class : str
            One of the TeX font names (as for *font*), but **not** "bb",
            "frak", or "scr".  This is used to combine two font classes.  The
            only supported combination currently is ``get_metrics("frak", "bf",
            ...)``.
        sym : str
            A symbol in raw TeX form, e.g., "1", "x", or "\\sigma".
        fontsize : float
            Font size in points.
        dpi : float
            Rendering dots-per-inch.

        Returns
        -------
        FontMetrics
        '''
    def render_glyph(self, output: Output, ox: float, oy: float, font: str, font_class: str, sym: str, fontsize: float, dpi: float) -> None:
        """
        At position (*ox*, *oy*), draw the glyph specified by the remaining
        parameters (see `get_metrics` for their detailed description).
        """
    def render_rect_filled(self, output: Output, x1: float, y1: float, x2: float, y2: float) -> None:
        """
        Draw a filled rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
    def get_xheight(self, font: str, fontsize: float, dpi: float) -> float:
        """
        Get the xheight for the given *font* and *fontsize*.
        """
    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float:
        """
        Get the line thickness that matches the given font.  Used as a
        base unit for drawing lines such as in a fraction or radical.
        """
    def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]]:
        """
        Override if your font provides multiple sizes of the same
        symbol.  Should return a list of symbols matching *sym* in
        various sizes.  The expression renderer will select the most
        appropriate size for a given situation from this list.
        """

class TruetypeFonts(Fonts, metaclass=abc.ABCMeta):
    """
    A generic base class for all font setups that use Truetype fonts
    (through FT2Font).
    """
    _fonts: Incomplete
    fontmap: dict[str | int, str]
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags) -> None: ...
    def _get_font(self, font: str | int) -> FT2Font: ...
    def _get_offset(self, font: FT2Font, glyph: Glyph, fontsize: float, dpi: float) -> float: ...
    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]: ...
    def _get_info(self, fontname: str, font_class: str, sym: str, fontsize: float, dpi: float) -> FontInfo: ...
    def get_xheight(self, fontname: str, fontsize: float, dpi: float) -> float: ...
    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float: ...
    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float, font2: str, fontclass2: str, sym2: str, fontsize2: float, dpi: float) -> float: ...

class BakomaFonts(TruetypeFonts):
    """
    Use the Bakoma TrueType fonts for rendering.

    Symbols are strewn about a number of font files, each of which has
    its own proprietary 8-bit encoding.
    """
    _fontmap: Incomplete
    _stix_fallback: Incomplete
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags) -> None: ...
    _slanted_symbols: Incomplete
    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]: ...
    _size_alternatives: Incomplete
    def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]]: ...

class UnicodeFonts(TruetypeFonts):
    '''
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I\'m aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol cannot be found in the font.
    '''
    _cmr10_substitutions: Incomplete
    _fallback_font: Incomplete
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags) -> None: ...
    _slanted_symbols: Incomplete
    def _map_virtual_font(self, fontname: str, font_class: str, uniindex: int) -> tuple[str, int]: ...
    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]: ...
    def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]]: ...

class DejaVuFonts(UnicodeFonts, metaclass=abc.ABCMeta):
    _fontmap: dict[str | int, str]
    _fallback_font: Incomplete
    bakoma: Incomplete
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags) -> None: ...
    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]: ...

class DejaVuSerifFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Serif fonts

    If a glyph is not found it will fallback to Stix Serif
    """
    _fontmap: Incomplete

class DejaVuSansFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Sans fonts

    If a glyph is not found it will fallback to Stix Sans
    """
    _fontmap: Incomplete

class StixFonts(UnicodeFonts):
    '''
    A font handling class for the STIX fonts.

    In addition to what UnicodeFonts provides, this class:

    - supports "virtual fonts" which are complete alpha numeric
      character sets with different font styles at special Unicode
      code points, such as "Blackboard".

    - handles sized alternative characters for the STIXSizeX fonts.
    '''
    _fontmap: dict[str | int, str]
    _fallback_font: Incomplete
    _sans: bool
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags) -> None: ...
    def _map_virtual_font(self, fontname: str, font_class: str, uniindex: int) -> tuple[str, int]: ...
    def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]] | list[tuple[int, str]]: ...

class StixSansFonts(StixFonts):
    """
    A font handling class for the STIX fonts (that uses sans-serif
    characters by default).
    """
    _sans: bool

SHRINK_FACTOR: float
NUM_SIZE_LEVELS: int

class FontConstantsBase:
    """
    A set of constants that controls how certain things, such as sub-
    and superscripts are laid out.  These are all metrics that can't
    be reliably retrieved from the font metrics in the font itself.
    """
    script_space: T.ClassVar[float]
    subdrop: T.ClassVar[float]
    sup1: T.ClassVar[float]
    sub1: T.ClassVar[float]
    sub2: T.ClassVar[float]
    delta: T.ClassVar[float]
    delta_slanted: T.ClassVar[float]
    delta_integral: T.ClassVar[float]

class ComputerModernFontConstants(FontConstantsBase):
    script_space: float
    subdrop: float
    sup1: float
    sub1: float
    sub2: float
    delta: float
    delta_slanted: float
    delta_integral: float

class STIXFontConstants(FontConstantsBase):
    script_space: float
    sup1: float
    sub2: float
    delta: float
    delta_slanted: float
    delta_integral: float

class STIXSansFontConstants(FontConstantsBase):
    script_space: float
    sup1: float
    delta_slanted: float
    delta_integral: float

class DejaVuSerifFontConstants(FontConstantsBase): ...
class DejaVuSansFontConstants(FontConstantsBase): ...

_font_constant_mapping: Incomplete

def _get_font_constant_set(state: ParserState) -> type[FontConstantsBase]: ...

class Node:
    """A node in the TeX box model."""
    size: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def get_kerning(self, next: Node | None) -> float: ...
    def shrink(self) -> None:
        """
        Shrinks one level smaller.  There are only three levels of
        sizes, after which things will no longer get smaller.
        """
    def render(self, output: Output, x: float, y: float) -> None:
        """Render this node."""

class Box(Node):
    """A node with a physical location."""
    width: Incomplete
    height: Incomplete
    depth: Incomplete
    def __init__(self, width: float, height: float, depth: float) -> None: ...
    def shrink(self) -> None: ...
    def render(self, output: Output, x1: float, y1: float, x2: float, y2: float) -> None: ...

class Vbox(Box):
    """A box with only height (zero width)."""
    def __init__(self, height: float, depth: float) -> None: ...

class Hbox(Box):
    """A box with only width (zero height and depth)."""
    def __init__(self, width: float) -> None: ...

class Char(Node):
    """
    A single character.

    Unlike TeX, the font information and metrics are stored with each `Char`
    to make it easier to lookup the font metrics when needed.  Note that TeX
    boxes have a width, height, and depth, unlike Type1 and TrueType which use
    a full bounding box and an advance in the x-direction.  The metrics must
    be converted to the TeX model, and the advance (if different from width)
    must be converted into a `Kern` node when the `Char` is added to its parent
    `Hlist`.
    """
    c: Incomplete
    fontset: Incomplete
    font: Incomplete
    font_class: Incomplete
    fontsize: Incomplete
    dpi: Incomplete
    def __init__(self, c: str, state: ParserState) -> None: ...
    def __repr__(self) -> str: ...
    width: Incomplete
    height: Incomplete
    depth: Incomplete
    def _update_metrics(self) -> None: ...
    def is_slanted(self) -> bool: ...
    def get_kerning(self, next: Node | None) -> float:
        """
        Return the amount of kerning between this and the given character.

        This method is called when characters are strung together into `Hlist`
        to create `Kern` nodes.
        """
    def render(self, output: Output, x: float, y: float) -> None: ...
    def shrink(self) -> None: ...

class Accent(Char):
    """
    The font metrics need to be dealt with differently for accents,
    since they are already offset correctly from the baseline in
    TrueType fonts.
    """
    width: Incomplete
    height: Incomplete
    depth: int
    def _update_metrics(self) -> None: ...
    def shrink(self) -> None: ...
    def render(self, output: Output, x: float, y: float) -> None: ...

class List(Box):
    """A list of nodes (either horizontal or vertical)."""
    shift_amount: float
    children: Incomplete
    glue_set: float
    glue_sign: int
    glue_order: int
    def __init__(self, elements: T.Sequence[Node]) -> None: ...
    def __repr__(self) -> str: ...
    glue_ratio: float
    def _set_glue(self, x: float, sign: int, totals: list[float], error_type: str) -> None: ...
    def shrink(self) -> None: ...

class Hlist(List):
    """A horizontal list of boxes."""
    def __init__(self, elements: T.Sequence[Node], w: float = 0.0, m: T.Literal['additional', 'exactly'] = 'additional', do_kern: bool = True) -> None: ...
    children: Incomplete
    def kern(self) -> None:
        """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
    height: Incomplete
    depth: Incomplete
    width: Incomplete
    glue_sign: int
    glue_order: int
    glue_ratio: float
    def hpack(self, w: float = 0.0, m: T.Literal['additional', 'exactly'] = 'additional') -> None:
        """
        Compute the dimensions of the resulting boxes, and adjust the glue if
        one of those dimensions is pre-specified.  The computed sizes normally
        enclose all of the material inside the new box; but some items may
        stick out if negative glue is used, if the box is overfull, or if a
        ``\\vbox`` includes other boxes that have been shifted left.

        Parameters
        ----------
        w : float, default: 0
            A width.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose width is 'exactly' *w*; or a box
            with the natural width of the contents, plus *w* ('additional').

        Notes
        -----
        The defaults produce a box with the natural width of the contents.
        """

class Vlist(List):
    """A vertical list of boxes."""
    def __init__(self, elements: T.Sequence[Node], h: float = 0.0, m: T.Literal['additional', 'exactly'] = 'additional') -> None: ...
    width: Incomplete
    depth: Incomplete
    height: Incomplete
    glue_sign: int
    glue_order: int
    glue_ratio: float
    def vpack(self, h: float = 0.0, m: T.Literal['additional', 'exactly'] = 'additional', l: float = ...) -> None:
        """
        Compute the dimensions of the resulting boxes, and to adjust the glue
        if one of those dimensions is pre-specified.

        Parameters
        ----------
        h : float, default: 0
            A height.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose height is 'exactly' *h*; or a box
            with the natural height of the contents, plus *h* ('additional').
        l : float, default: np.inf
            The maximum height.

        Notes
        -----
        The defaults produce a box with the natural height of the contents.
        """

class Rule(Box):
    '''
    A solid black rectangle.

    It has *width*, *depth*, and *height* fields just as in an `Hlist`.
    However, if any of these dimensions is inf, the actual value will be
    determined by running the rule up to the boundary of the innermost
    enclosing box.  This is called a "running dimension".  The width is never
    running in an `Hlist`; the height and depth are never running in a `Vlist`.
    '''
    fontset: Incomplete
    def __init__(self, width: float, height: float, depth: float, state: ParserState) -> None: ...
    def render(self, output: Output, x: float, y: float, w: float, h: float) -> None: ...

class Hrule(Rule):
    """Convenience class to create a horizontal rule."""
    def __init__(self, state: ParserState, thickness: float | None = None) -> None: ...

class Vrule(Rule):
    """Convenience class to create a vertical rule."""
    def __init__(self, state: ParserState) -> None: ...

class _GlueSpec(NamedTuple):
    width: float
    stretch: float
    stretch_order: int
    shrink: float
    shrink_order: int

class Glue(Node):
    """
    Most of the information in this object is stored in the underlying
    ``_GlueSpec`` class, which is shared between multiple glue objects.
    (This is a memory optimization which probably doesn't matter anymore, but
    it's easier to stick to what TeX does.)
    """
    glue_spec: Incomplete
    def __init__(self, glue_type: _GlueSpec | T.Literal['fil', 'fill', 'filll', 'neg_fil', 'neg_fill', 'neg_filll', 'empty', 'ss']) -> None: ...
    def shrink(self) -> None: ...

class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """
    def __init__(self, elements: list[Node]) -> None: ...

class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """
    def __init__(self, elements: list[Node]) -> None: ...

class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """
    height: int
    depth: int
    width: Incomplete
    def __init__(self, width: float) -> None: ...
    def __repr__(self) -> str: ...
    def shrink(self) -> None: ...

class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """
    shift_amount: Incomplete
    def __init__(self, c: str, height: float, depth: float, state: ParserState, always: bool = False, factor: float | None = None) -> None: ...

class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """
    width: Incomplete
    def __init__(self, c: str, width: float, state: ParserState, always: bool = False, char_class: type[Char] = ...) -> None: ...

def ship(box: Box, xy: tuple[float, float] = (0, 0)) -> Output:
    """
    Ship out *box* at offset *xy*, converting it to an `Output`.

    Since boxes can be inside of boxes inside of boxes, the main work of `ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become local variables here.
    """
def Error(msg: str) -> ParserElement:
    """Helper class to raise parser errors."""

class ParserState:
    '''
    Parser state.

    States are pushed and popped from a stack as necessary, and the "current"
    state is always at the top of the stack.

    Upon entering and leaving a group { } or math/non-math, the stack is pushed
    and popped accordingly.
    '''
    fontset: Incomplete
    _font: Incomplete
    font_class: Incomplete
    fontsize: Incomplete
    dpi: Incomplete
    def __init__(self, fontset: Fonts, font: str, font_class: str, fontsize: float, dpi: float) -> None: ...
    def copy(self) -> ParserState: ...
    @property
    def font(self) -> str: ...
    @font.setter
    def font(self, name: str) -> None: ...
    def get_current_underline_thickness(self) -> float:
        """Return the underline thickness for this state."""

def cmd(expr: str, args: ParserElement) -> ParserElement:
    '''
    Helper to define TeX commands.

    ``cmd("\\cmd", args)`` is equivalent to
    ``"\\cmd" - (args | Error("Expected \\cmd{arg}{...}"))`` where the names in
    the error message are taken from element names in *args*.  If *expr*
    already includes arguments (e.g. "\\cmd{arg}{...}"), then they are stripped
    when constructing the parse element, but kept (and *expr* is used as is) in
    the error message.
    '''

class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """
    class _MathStyle(enum.Enum):
        DISPLAYSTYLE = 0
        TEXTSTYLE = 1
        SCRIPTSTYLE = 2
        SCRIPTSCRIPTSTYLE = 3
    _binary_operators: Incomplete
    _relation_symbols: Incomplete
    _arrow_symbols: Incomplete
    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols
    _punctuation_symbols: Incomplete
    _overunder_symbols: Incomplete
    _overunder_functions: Incomplete
    _dropsub_symbols: Incomplete
    _fontnames: Incomplete
    _function_names: Incomplete
    _ambi_delims: Incomplete
    _left_delims: Incomplete
    _right_delims: Incomplete
    _delims = _left_delims | _right_delims | _ambi_delims
    _small_greek: Incomplete
    _latin_alphabets: Incomplete
    _expression: Incomplete
    _math_expression: Incomplete
    _in_subscript_or_superscript: bool
    def __init__(self) -> None: ...
    _state_stack: Incomplete
    _em_width_cache: dict[tuple[str, float, float], float]
    def parse(self, s: str, fonts_object: Fonts, fontsize: float, dpi: float) -> Hlist:
        """
        Parse expression *s* using the given *fonts_object* for
        output, at the given *fontsize* and *dpi*.

        Returns the parse tree of `Node` instances.
        """
    def get_state(self) -> ParserState:
        """Get the current `State` of the parser."""
    def pop_state(self) -> None:
        """Pop a `State` off of the stack."""
    def push_state(self) -> None:
        """Push a new `State` onto the stack, copying the current state."""
    def main(self, toks: ParseResults) -> list[Hlist]: ...
    def math_string(self, toks: ParseResults) -> ParseResults: ...
    def math(self, toks: ParseResults) -> T.Any: ...
    def non_math(self, toks: ParseResults) -> T.Any: ...
    float_literal: Incomplete
    def text(self, toks: ParseResults) -> T.Any: ...
    def _make_space(self, percentage: float) -> Kern: ...
    _space_widths: Incomplete
    def space(self, toks: ParseResults) -> T.Any: ...
    def customspace(self, toks: ParseResults) -> T.Any: ...
    def symbol(self, s: str, loc: int, toks: ParseResults | dict[str, str]) -> T.Any: ...
    def unknown_symbol(self, s: str, loc: int, toks: ParseResults) -> T.Any: ...
    _accent_map: Incomplete
    _wide_accents: Incomplete
    def accent(self, toks: ParseResults) -> T.Any: ...
    def function(self, s: str, loc: int, toks: ParseResults) -> T.Any: ...
    def operatorname(self, s: str, loc: int, toks: ParseResults) -> T.Any: ...
    def start_group(self, toks: ParseResults) -> T.Any: ...
    def group(self, toks: ParseResults) -> T.Any: ...
    def required_group(self, toks: ParseResults) -> T.Any: ...
    optional_group = required_group
    def end_group(self) -> T.Any: ...
    def unclosed_group(self, s: str, loc: int, toks: ParseResults) -> T.Any: ...
    def font(self, toks: ParseResults) -> T.Any: ...
    def is_overunder(self, nucleus: Node) -> bool: ...
    def is_dropsub(self, nucleus: Node) -> bool: ...
    def is_slanted(self, nucleus: Node) -> bool: ...
    def subsuper(self, s: str, loc: int, toks: ParseResults) -> T.Any: ...
    def _genfrac(self, ldelim: str, rdelim: str, rule: float | None, style: _MathStyle, num: Hlist, den: Hlist) -> T.Any: ...
    def style_literal(self, toks: ParseResults) -> T.Any: ...
    def genfrac(self, toks: ParseResults) -> T.Any: ...
    def frac(self, toks: ParseResults) -> T.Any: ...
    def dfrac(self, toks: ParseResults) -> T.Any: ...
    def binom(self, toks: ParseResults) -> T.Any: ...
    def _genset(self, s: str, loc: int, toks: ParseResults) -> T.Any: ...
    overset = _genset
    underset = _genset
    def sqrt(self, toks: ParseResults) -> T.Any: ...
    def overline(self, toks: ParseResults) -> T.Any: ...
    def _auto_sized_delimiter(self, front: str, middle: list[Box | Char | str], back: str) -> T.Any: ...
    def auto_delim(self, toks: ParseResults) -> T.Any: ...
    def boldsymbol(self, toks: ParseResults) -> T.Any: ...
    def substack(self, toks: ParseResults) -> T.Any: ...

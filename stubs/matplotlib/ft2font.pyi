import enum
import numpy
from typing import Callable, ClassVar

__freetype_build_type__: str
__freetype_version__: str

class FT2Font:
    def __init__(self, *args, **kwargs) -> None:
        """__init__(self: matplotlib.ft2font.FT2Font, filename: object, hinting_factor: int = 8, *, _fallback_list: Optional[list[matplotlib.ft2font.FT2Font]] = None, _kerning_factor: int = 0) -> None


            Parameters
            ----------
            filename : str or file-like
                The source of the font data in a format (ttf or ttc) that FreeType can read.

            hinting_factor : int, optional
                Must be positive. Used to scale the hinting in the x-direction.

            _fallback_list : list of FT2Font, optional
                A list of FT2Font objects used to find missing glyphs.

                .. warning::
                    This API is both private and provisional: do not use it directly.

            _kerning_factor : int, optional
                Used to adjust the degree of kerning.

                .. warning::
                    This API is private: do not use it directly.

        """
    def _get_fontmap(self, string: str) -> dict:
        """_get_fontmap(self: matplotlib.ft2font.FT2Font, string: str) -> dict


            Get a mapping between characters and the font that includes them.

            .. warning::
                This API uses the fallback list and is both private and provisional: do not use
                it directly.

            Parameters
            ----------
            text : str
                The characters for which to find fonts.

            Returns
            -------
            dict[str, FT2Font]
                A dictionary mapping unicode characters to `.FT2Font` objects.

        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def clear(self) -> None:
        """clear(self: matplotlib.ft2font.FT2Font) -> None

        Clear all the glyphs, reset for a new call to `.set_text`.
        """
    def draw_glyph_to_bitmap(self, *args, **kwargs):
        """draw_glyph_to_bitmap(self: matplotlib.ft2font.FT2Font, image: matplotlib.ft2font.FT2Image, x: Union[float, int], y: Union[float, int], glyph: matplotlib.ft2font.Glyph, *, antialiased: bool = True) -> None


            Draw a single glyph to the bitmap at pixel locations x, y.

            Note it is your responsibility to create the image manually with the correct size
            before this call is made.

            If you want automatic layout, use `.set_text` in combinations with
            `.draw_glyphs_to_bitmap`. This function is instead intended for people who want to
            render individual glyphs (e.g., returned by `.load_char`) at precise locations.

            Parameters
            ----------
            image : FT2Image
                The image buffer on which to draw the glyph.
            x, y : int
                The pixel location at which to draw the glyph.
            glyph : Glyph
                The glyph to draw.
            antialiased : bool, default: True
                Whether to render glyphs 8-bit antialiased or in pure black-and-white.

            See Also
            --------
            .draw_glyphs_to_bitmap

        """
    def draw_glyphs_to_bitmap(self, *args, **kwargs):
        """draw_glyphs_to_bitmap(self: matplotlib.ft2font.FT2Font, *, antialiased: bool = True) -> None


            Draw the glyphs that were loaded by `.set_text` to the bitmap.

            The bitmap size will be automatically set to include the glyphs.

            Parameters
            ----------
            antialiased : bool, default: True
                Whether to render glyphs 8-bit antialiased or in pure black-and-white.

            See Also
            --------
            .draw_glyph_to_bitmap

        """
    def get_bitmap_offset(self) -> tuple:
        """get_bitmap_offset(self: matplotlib.ft2font.FT2Font) -> tuple


            Get the (x, y) offset for the bitmap if ink hangs left or below (0, 0).

            Since Matplotlib only supports left-to-right text, y is always 0.

            Returns
            -------
            x, y : float
                The x and y offset in 26.6 subpixels of the bitmap. To get x and y in pixels,
                divide these values by 64.

            See Also
            --------
            .get_width_height
            .get_descent

        """
    def get_char_index(self, codepoint: int) -> int:
        """get_char_index(self: matplotlib.ft2font.FT2Font, codepoint: int) -> int


            Return the glyph index corresponding to a character code point.

            Parameters
            ----------
            codepoint : int
                A character code point in the current charmap (which defaults to Unicode.)

            Returns
            -------
            int
                The corresponding glyph index.

            See Also
            --------
            .set_charmap
            .select_charmap
            .get_glyph_name
            .get_name_index

        """
    def get_charmap(self) -> dict:
        """get_charmap(self: matplotlib.ft2font.FT2Font) -> dict


            Return a mapping of character codes to glyph indices in the font.

            The charmap is Unicode by default, but may be changed by `.set_charmap` or
            `.select_charmap`.

            Returns
            -------
            dict[int, int]
                A dictionary of the selected charmap mapping character codes to their
                corresponding glyph indices.

        """
    def get_descent(self) -> int:
        """get_descent(self: matplotlib.ft2font.FT2Font) -> int


            Get the descent of the current string set by `.set_text`.

            The rotation of the string is accounted for.

            Returns
            -------
            int
                The descent in 26.6 subpixels of the bitmap. To get the descent in pixels,
                divide these values by 64.

            See Also
            --------
            .get_bitmap_offset
            .get_width_height

        """
    def get_glyph_name(self, index: int) -> str:
        """get_glyph_name(self: matplotlib.ft2font.FT2Font, index: int) -> str


            Retrieve the ASCII name of a given glyph *index* in a face.

            Due to Matplotlib's internal design, for fonts that do not contain glyph names (per
            ``FT_FACE_FLAG_GLYPH_NAMES``), this returns a made-up name which does *not*
            roundtrip through `.get_name_index`.

            Parameters
            ----------
            index : int
                The glyph number to query.

            Returns
            -------
            str
                The name of the glyph, or if the font does not contain names, a name synthesized
                by Matplotlib.

            See Also
            --------
            .get_name_index

        """
    def get_image(self) -> numpy.ndarray:
        """get_image(self: matplotlib.ft2font.FT2Font) -> numpy.ndarray


            Return the underlying image buffer for this font object.

            Returns
            -------
            np.ndarray[int]

            See Also
            --------
            .get_path

        """
    def get_kerning(self, left: int, right: int, mode: Kerning | int) -> int:
        """get_kerning(self: matplotlib.ft2font.FT2Font, left: int, right: int, mode: Union[Kerning, int]) -> int


            Get the kerning between two glyphs.

            Parameters
            ----------
            left, right : int
                The glyph indices. Note these are not characters nor character codes.
                Use `.get_char_index` to convert character codes to glyph indices.

            mode : Kerning
                A kerning mode constant:

                - ``DEFAULT``  - Return scaled and grid-fitted kerning distances.
                - ``UNFITTED`` - Return scaled but un-grid-fitted kerning distances.
                - ``UNSCALED`` - Return the kerning vector in original font units.

                .. versionchanged:: 3.10
                    This now takes a `.ft2font.Kerning` value instead of an `int`.

            Returns
            -------
            int
                The kerning adjustment between the two glyphs.

        """
    def get_name_index(self, name: str) -> int:
        """get_name_index(self: matplotlib.ft2font.FT2Font, name: str) -> int


            Return the glyph index of a given glyph *name*.

            Parameters
            ----------
            name : str
                The name of the glyph to query.

            Returns
            -------
            int
                The corresponding glyph index; 0 means 'undefined character code'.

            See Also
            --------
            .get_char_index
            .get_glyph_name

        """
    def get_num_glyphs(self) -> int:
        """get_num_glyphs(self: matplotlib.ft2font.FT2Font) -> int

        Return the number of loaded glyphs.
        """
    def get_path(self) -> tuple:
        """get_path(self: matplotlib.ft2font.FT2Font) -> tuple


            Get the path data from the currently loaded glyph.

            Returns
            -------
            vertices : np.ndarray[double]
                The (N, 2) array of vertices describing the current glyph.
            codes : np.ndarray[np.uint8]
                The (N, ) array of codes corresponding to the vertices.

            See Also
            --------
            .get_image
            .load_char
            .load_glyph
            .set_text

        """
    def get_ps_font_info(self) -> tuple:
        """get_ps_font_info(self: matplotlib.ft2font.FT2Font) -> tuple


            Return the information in the PS Font Info structure.

            For more information, see the `FreeType documentation on this structure
            <https://freetype.org/freetype2/docs/reference/ft2-type1_tables.html#ps_fontinforec>`_.

            Returns
            -------
            version : str
            notice : str
            full_name : str
            family_name : str
            weight : str
            italic_angle : int
            is_fixed_pitch : bool
            underline_position : int
            underline_thickness : int

        """
    def get_sfnt(self) -> dict:
        """get_sfnt(self: matplotlib.ft2font.FT2Font) -> dict


            Load the entire SFNT names table.

            Returns
            -------
            dict[tuple[int, int, int, int], bytes]
                The SFNT names table; the dictionary keys are tuples of:

                    (platform-ID, ISO-encoding-scheme, language-code, description)

                and the values are the direct information from the font table.

        """
    def get_sfnt_table(self, name: str) -> dict | None:
        '''get_sfnt_table(self: matplotlib.ft2font.FT2Font, name: str) -> Optional[dict]


            Return one of the SFNT tables.

            Parameters
            ----------
            name : {"head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt"}
                Which table to return.

            Returns
            -------
            dict[str, Any]
                The corresponding table; for more information, see `the FreeType documentation
                <https://freetype.org/freetype2/docs/reference/ft2-truetype_tables.html>`_.

        '''
    def get_width_height(self) -> tuple:
        """get_width_height(self: matplotlib.ft2font.FT2Font) -> tuple


            Get the dimensions of the current string set by `.set_text`.

            The rotation of the string is accounted for.

            Returns
            -------
            width, height : float
                The width and height in 26.6 subpixels of the current string. To get width and
                height in pixels, divide these values by 64.

            See Also
            --------
            .get_bitmap_offset
            .get_descent

        """
    def load_char(self, charcode: int, flags: LoadFlags | int = ...) -> Glyph:
        """load_char(self: matplotlib.ft2font.FT2Font, charcode: int, flags: Union[LoadFlags, int] = <LoadFlags.FORCE_AUTOHINT: 32>) -> matplotlib.ft2font.Glyph


            Load character in current fontfile and set glyph.

            Parameters
            ----------
            charcode : int
                The character code to prepare rendering information for. This code must be in
                the charmap, or else a ``.notdef`` glyph may be returned instead.
            flags : LoadFlags, default: `.LoadFlags.FORCE_AUTOHINT`
                Any bitwise-OR combination of the `.LoadFlags` flags.

                .. versionchanged:: 3.10
                    This now takes an `.ft2font.LoadFlags` instead of an int.

            Returns
            -------
            Glyph
                The glyph information corresponding to the specified character.

            See Also
            --------
            .load_glyph
            .select_charmap
            .set_charmap

        """
    def load_glyph(self, glyph_index: int, flags: LoadFlags | int = ...) -> Glyph:
        """load_glyph(self: matplotlib.ft2font.FT2Font, glyph_index: int, flags: Union[LoadFlags, int] = <LoadFlags.FORCE_AUTOHINT: 32>) -> matplotlib.ft2font.Glyph


            Load glyph index in current fontfile and set glyph.

            Note that the glyph index is specific to a font, and not universal like a Unicode
            code point.

            Parameters
            ----------
            glyph_index : int
                The glyph index to prepare rendering information for.
            flags : LoadFlags, default: `.LoadFlags.FORCE_AUTOHINT`
                Any bitwise-OR combination of the `.LoadFlags` flags.

                .. versionchanged:: 3.10
                    This now takes an `.ft2font.LoadFlags` instead of an int.

            Returns
            -------
            Glyph
                The glyph information corresponding to the specified index.

            See Also
            --------
            .load_char

        """
    def select_charmap(self, i: int) -> None:
        """select_charmap(self: matplotlib.ft2font.FT2Font, i: int) -> None


            Select a charmap by its FT_Encoding number.

            For more details on character mapping, see the `FreeType documentation
            <https://freetype.org/freetype2/docs/reference/ft2-character_mapping.html>`_.

            Parameters
            ----------
            i : int
                The charmap in the form defined by FreeType:
                https://freetype.org/freetype2/docs/reference/ft2-character_mapping.html#ft_encoding

            See Also
            --------
            .set_charmap
            .get_charmap

        """
    def set_charmap(self, i: int) -> None:
        """set_charmap(self: matplotlib.ft2font.FT2Font, i: int) -> None


            Make the i-th charmap current.

            For more details on character mapping, see the `FreeType documentation
            <https://freetype.org/freetype2/docs/reference/ft2-character_mapping.html>`_.

            Parameters
            ----------
            i : int
                The charmap number in the range [0, `.num_charmaps`).

            See Also
            --------
            .num_charmaps
            .select_charmap
            .get_charmap

        """
    def set_size(self, ptsize: float, dpi: float) -> None:
        """set_size(self: matplotlib.ft2font.FT2Font, ptsize: float, dpi: float) -> None


            Set the size of the text.

            Parameters
            ----------
            ptsize : float
                The size of the text in points.
            dpi : float
                The DPI used for rendering the text.

        """
    def set_text(self, string: str, angle: float = ..., flags: LoadFlags | int = ...) -> numpy.ndarray[numpy.float64]:
        """set_text(self: matplotlib.ft2font.FT2Font, string: str, angle: float = 0.0, flags: Union[LoadFlags, int] = <LoadFlags.FORCE_AUTOHINT: 32>) -> numpy.ndarray[numpy.float64]


            Set the text *string* and *angle*.

            You must call this before `.draw_glyphs_to_bitmap`.

            Parameters
            ----------
            string : str
                The text to prepare rendering information for.
            angle : float
                The angle at which to render the supplied text.
            flags : LoadFlags, default: `.LoadFlags.FORCE_AUTOHINT`
                Any bitwise-OR combination of the `.LoadFlags` flags.

                .. versionchanged:: 3.10
                    This now takes an `.ft2font.LoadFlags` instead of an int.

            Returns
            -------
            np.ndarray[double]
                A sequence of x,y glyph positions in 26.6 subpixels; divide by 64 for pixels.

        """
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""
    @property
    def ascender(self) -> int: ...
    @property
    def bbox(self) -> tuple: ...
    @property
    def descender(self) -> int: ...
    @property
    def face_flags(self) -> FaceFlags: ...
    @property
    def family_name(self) -> str: ...
    @property
    def fname(self) -> str: ...
    @property
    def height(self) -> int: ...
    @property
    def max_advance_height(self) -> int: ...
    @property
    def max_advance_width(self) -> int: ...
    @property
    def num_charmaps(self) -> int: ...
    @property
    def num_faces(self) -> int: ...
    @property
    def num_fixed_sizes(self) -> int: ...
    @property
    def num_glyphs(self) -> int: ...
    @property
    def num_named_instances(self) -> int: ...
    @property
    def postscript_name(self) -> str: ...
    @property
    def scalable(self) -> bool: ...
    @property
    def style_flags(self) -> StyleFlags: ...
    @property
    def style_name(self) -> str: ...
    @property
    def underline_position(self) -> int: ...
    @property
    def underline_thickness(self) -> int: ...
    @property
    def units_per_EM(self) -> int: ...

class FT2Image:
    def __init__(self, width: float | int, height: float | int) -> None:
        """__init__(self: matplotlib.ft2font.FT2Image, width: Union[float, int], height: Union[float, int]) -> None


            Parameters
            ----------
            width, height : int
                The dimensions of the image buffer.

        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def draw_rect_filled(self, x0: float | int, y0: float | int, x1: float | int, y1: float | int) -> None:
        """draw_rect_filled(self: matplotlib.ft2font.FT2Image, x0: Union[float, int], y0: Union[float, int], x1: Union[float, int], y1: Union[float, int]) -> None


            Draw a filled rectangle to the image.

            Parameters
            ----------
            x0, y0, x1, y1 : float
                The bounds of the rectangle from (x0, y0) to (x1, y1).

        """
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""

class FaceFlags(enum.Flag):
    __new__: ClassVar[Callable] = ...
    CID_KEYED: ClassVar[FaceFlags] = ...
    COLOR: ClassVar[FaceFlags] = ...
    EXTERNAL_STREAM: ClassVar[FaceFlags] = ...
    FAST_GLYPHS: ClassVar[FaceFlags] = ...
    FIXED_SIZES: ClassVar[FaceFlags] = ...
    FIXED_WIDTH: ClassVar[FaceFlags] = ...
    GLYPH_NAMES: ClassVar[FaceFlags] = ...
    HINTER: ClassVar[FaceFlags] = ...
    HORIZONTAL: ClassVar[FaceFlags] = ...
    KERNING: ClassVar[FaceFlags] = ...
    MULTIPLE_MASTERS: ClassVar[FaceFlags] = ...
    SCALABLE: ClassVar[FaceFlags] = ...
    SFNT: ClassVar[FaceFlags] = ...
    TRICKY: ClassVar[FaceFlags] = ...
    VERTICAL: ClassVar[FaceFlags] = ...
    _all_bits_: ClassVar[int] = ...
    _boundary_: ClassVar[enum.FlagBoundary] = ...
    _flag_mask_: ClassVar[int] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _hashable_values_: ClassVar[list] = ...
    _inverted_: ClassVar[None] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[object]] = ...
    _singles_mask_: ClassVar[int] = ...
    _unhashable_values_: ClassVar[list] = ...
    _unhashable_values_map_: ClassVar[dict] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    _value_repr_: ClassVar[None] = ...
    __and__: ClassVar[Callable] = ...
    __invert__: ClassVar[Callable] = ...
    __or__: ClassVar[Callable] = ...
    __rand__: ClassVar[Callable] = ...
    __ror__: ClassVar[Callable] = ...
    __rxor__: ClassVar[Callable] = ...
    __xor__: ClassVar[Callable] = ...
    @classmethod
    def _new_member_(cls, *args, **kwargs):
        """Create and return a new object.  See help(type) for accurate signature."""

class Glyph:
    def __init__(self) -> None:
        """__init__(self: matplotlib.ft2font.Glyph) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def bbox(self) -> tuple: ...
    @property
    def height(self) -> int: ...
    @property
    def horiAdvance(self) -> int: ...
    @property
    def horiBearingX(self) -> int: ...
    @property
    def horiBearingY(self) -> int: ...
    @property
    def linearHoriAdvance(self) -> int: ...
    @property
    def vertAdvance(self) -> int: ...
    @property
    def vertBearingX(self) -> int: ...
    @property
    def vertBearingY(self) -> int: ...
    @property
    def width(self) -> int: ...

class Kerning(enum.Enum):
    __new__: ClassVar[Callable] = ...
    DEFAULT: ClassVar[Kerning] = ...
    UNFITTED: ClassVar[Kerning] = ...
    UNSCALED: ClassVar[Kerning] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _hashable_values_: ClassVar[list] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[object]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _unhashable_values_map_: ClassVar[dict] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    _value_repr_: ClassVar[None] = ...
    @classmethod
    def _new_member_(cls, *args, **kwargs):
        """Create and return a new object.  See help(type) for accurate signature."""

class LoadFlags(enum.Flag):
    __new__: ClassVar[Callable] = ...
    COLOR: ClassVar[LoadFlags] = ...
    COMPUTE_METRICS: ClassVar[LoadFlags] = ...
    CROP_BITMAP: ClassVar[LoadFlags] = ...
    DEFAULT: ClassVar[LoadFlags] = ...
    FORCE_AUTOHINT: ClassVar[LoadFlags] = ...
    IGNORE_GLOBAL_ADVANCE_WIDTH: ClassVar[LoadFlags] = ...
    IGNORE_TRANSFORM: ClassVar[LoadFlags] = ...
    LINEAR_DESIGN: ClassVar[LoadFlags] = ...
    MONOCHROME: ClassVar[LoadFlags] = ...
    NO_AUTOHINT: ClassVar[LoadFlags] = ...
    NO_BITMAP: ClassVar[LoadFlags] = ...
    NO_HINTING: ClassVar[LoadFlags] = ...
    NO_RECURSE: ClassVar[LoadFlags] = ...
    NO_SCALE: ClassVar[LoadFlags] = ...
    PEDANTIC: ClassVar[LoadFlags] = ...
    RENDER: ClassVar[LoadFlags] = ...
    TARGET_LCD: ClassVar[LoadFlags] = ...
    TARGET_LCD_V: ClassVar[LoadFlags] = ...
    TARGET_LIGHT: ClassVar[LoadFlags] = ...
    TARGET_MONO: ClassVar[LoadFlags] = ...
    TARGET_NORMAL: ClassVar[LoadFlags] = ...
    VERTICAL_LAYOUT: ClassVar[LoadFlags] = ...
    _all_bits_: ClassVar[int] = ...
    _boundary_: ClassVar[enum.FlagBoundary] = ...
    _flag_mask_: ClassVar[int] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _hashable_values_: ClassVar[list] = ...
    _inverted_: ClassVar[None] = ...
    _iter_member_: ClassVar[method] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[object]] = ...
    _singles_mask_: ClassVar[int] = ...
    _unhashable_values_: ClassVar[list] = ...
    _unhashable_values_map_: ClassVar[dict] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    _value_repr_: ClassVar[None] = ...
    __and__: ClassVar[Callable] = ...
    __invert__: ClassVar[Callable] = ...
    __or__: ClassVar[Callable] = ...
    __rand__: ClassVar[Callable] = ...
    __ror__: ClassVar[Callable] = ...
    __rxor__: ClassVar[Callable] = ...
    __xor__: ClassVar[Callable] = ...
    @classmethod
    def _new_member_(cls, *args, **kwargs):
        """Create and return a new object.  See help(type) for accurate signature."""

class StyleFlags(enum.Flag):
    __new__: ClassVar[Callable] = ...
    BOLD: ClassVar[StyleFlags] = ...
    ITALIC: ClassVar[StyleFlags] = ...
    NORMAL: ClassVar[StyleFlags] = ...
    _all_bits_: ClassVar[int] = ...
    _boundary_: ClassVar[enum.FlagBoundary] = ...
    _flag_mask_: ClassVar[int] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _hashable_values_: ClassVar[list] = ...
    _inverted_: ClassVar[None] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[object]] = ...
    _singles_mask_: ClassVar[int] = ...
    _unhashable_values_: ClassVar[list] = ...
    _unhashable_values_map_: ClassVar[dict] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    _value_repr_: ClassVar[None] = ...
    __and__: ClassVar[Callable] = ...
    __invert__: ClassVar[Callable] = ...
    __or__: ClassVar[Callable] = ...
    __rand__: ClassVar[Callable] = ...
    __ror__: ClassVar[Callable] = ...
    __rxor__: ClassVar[Callable] = ...
    __xor__: ClassVar[Callable] = ...
    @classmethod
    def _new_member_(cls, *args, **kwargs):
        """Create and return a new object.  See help(type) for accurate signature."""

def __getattr__(arg0: str) -> object:
    """__getattr__(arg0: str) -> object"""

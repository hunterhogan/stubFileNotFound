import dataclasses
import json
from _typeshed import Incomplete
from matplotlib import _afm as _afm, _api as _api, cbook as cbook, ft2font as ft2font
from matplotlib._fontconfig_pattern import generate_fontconfig_pattern as generate_fontconfig_pattern, parse_fontconfig_pattern as parse_fontconfig_pattern

_log: Incomplete
font_scalings: Incomplete
stretch_dict: Incomplete
weight_dict: Incomplete
_weight_regexes: Incomplete
font_family_aliases: Incomplete
_HOME: Incomplete
MSFolders: str
MSFontDirectories: Incomplete
MSUserFontDirectories: Incomplete
X11FontDirectories: Incomplete
OSXFontDirectories: Incomplete

def get_fontext_synonyms(fontext):
    """
    Return a list of file extensions that are synonyms for
    the given file extension *fileext*.
    """
def list_fonts(directory, extensions):
    """
    Return a list of all fonts matching any of the extensions, found
    recursively under the directory.
    """
def win32FontDirectory():
    """
    Return the user-specified font directory for Win32.  This is
    looked up from the registry key ::

      \\\\HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\\Fonts

    If the key is not found, ``%WINDIR%\\Fonts`` will be returned.
    """
def _get_win32_installed_fonts():
    """List the font paths known to the Windows registry."""
def _get_fontconfig_fonts():
    """Cache and list the font paths known to ``fc-list``."""
def _get_macos_fonts():
    """Cache and list the font paths known to ``system_profiler SPFontsDataType``."""
def findSystemFonts(fontpaths: Incomplete | None = None, fontext: str = 'ttf'):
    """
    Search for fonts in the specified font paths.  If no paths are
    given, will use a standard set of system paths, as well as the
    list of fonts tracked by fontconfig if fontconfig is installed and
    available.  A list of TrueType fonts are returned by default with
    AFM fonts as an option.
    """

@dataclasses.dataclass(frozen=True)
class FontEntry:
    """
    A class for storing Font properties.

    It is used when populating the font lookup dictionary.
    """
    fname: str = ...
    name: str = ...
    style: str = ...
    variant: str = ...
    weight: str | int = ...
    stretch: str = ...
    size: str = ...
    def _repr_html_(self) -> str: ...
    def _repr_png_(self) -> bytes: ...

def ttfFontProperty(font):
    """
    Extract information from a TrueType font file.

    Parameters
    ----------
    font : `.FT2Font`
        The TrueType font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.

    """
def afmFontProperty(fontpath, font):
    """
    Extract information from an AFM font file.

    Parameters
    ----------
    fontpath : str
        The filename corresponding to *font*.
    font : AFM
        The AFM font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.
    """
def _cleanup_fontproperties_init(init_method):
    """
    A decorator to limit the call signature to single a positional argument
    or alternatively only keyword arguments.

    We still accept but deprecate all other call signatures.

    When the deprecation expires we can switch the signature to::

        __init__(self, pattern=None, /, *, family=None, style=None, ...)

    plus a runtime check that pattern is not used alongside with the
    keyword arguments. This results eventually in the two possible
    call signatures::

        FontProperties(pattern)
        FontProperties(family=..., size=..., ...)

    """

class FontProperties:
    """
    A class for storing and manipulating font properties.

    The font properties are the six properties described in the
    `W3C Cascading Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification and *math_fontfamily* for math fonts:

    - family: A list of font names in decreasing order of priority.
      The items may include a generic font family name, either 'sans-serif',
      'serif', 'cursive', 'fantasy', or 'monospace'.  In that case, the actual
      font to be used will be looked up from the associated rcParam during the
      search process in `.findfont`. Default: :rc:`font.family`

    - style: Either 'normal', 'italic' or 'oblique'.
      Default: :rc:`font.style`

    - variant: Either 'normal' or 'small-caps'.
      Default: :rc:`font.variant`

    - stretch: A numeric value in the range 0-1000 or one of
      'ultra-condensed', 'extra-condensed', 'condensed',
      'semi-condensed', 'normal', 'semi-expanded', 'expanded',
      'extra-expanded' or 'ultra-expanded'. Default: :rc:`font.stretch`

    - weight: A numeric value in the range 0-1000 or one of
      'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
      'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
      'extra bold', 'black'. Default: :rc:`font.weight`

    - size: Either a relative value of 'xx-small', 'x-small',
      'small', 'medium', 'large', 'x-large', 'xx-large' or an
      absolute font size, e.g., 10. Default: :rc:`font.size`

    - math_fontfamily: The family of fonts used to render math text.
      Supported values are: 'dejavusans', 'dejavuserif', 'cm',
      'stix', 'stixsans' and 'custom'. Default: :rc:`mathtext.fontset`

    Alternatively, a font may be specified using the absolute path to a font
    file, by using the *fname* kwarg.  However, in this case, it is typically
    simpler to just pass the path (as a `pathlib.Path`, not a `str`) to the
    *font* kwarg of the `.Text` object.

    The preferred usage of font sizes is to use the relative values,
    e.g.,  'large', instead of absolute font sizes, e.g., 12.  This
    approach allows all text sizes to be made larger or smaller based
    on the font manager's default font size.

    This class accepts a single positional string as fontconfig_ pattern_,
    or alternatively individual properties as keyword arguments::

        FontProperties(pattern)
        FontProperties(*, family=None, style=None, variant=None, ...)

    This support does not depend on fontconfig; we are merely borrowing its
    pattern syntax for use here.

    .. _fontconfig: https://www.freedesktop.org/wiki/Software/fontconfig/
    .. _pattern:
       https://www.freedesktop.org/software/fontconfig/fontconfig-user.html

    Note that Matplotlib's internal font manager and fontconfig use a
    different algorithm to lookup fonts, so the results of the same pattern
    may be different in Matplotlib than in other applications that use
    fontconfig.
    """
    def __init__(self, family: Incomplete | None = None, style: Incomplete | None = None, variant: Incomplete | None = None, weight: Incomplete | None = None, stretch: Incomplete | None = None, size: Incomplete | None = None, fname: Incomplete | None = None, math_fontfamily: Incomplete | None = None) -> None: ...
    @classmethod
    def _from_any(cls, arg):
        """
        Generic constructor which can build a `.FontProperties` from any of the
        following:

        - a `.FontProperties`: it is passed through as is;
        - `None`: a `.FontProperties` using rc values is used;
        - an `os.PathLike`: it is used as path to the font file;
        - a `str`: it is parsed as a fontconfig pattern;
        - a `dict`: it is passed as ``**kwargs`` to `.FontProperties`.
        """
    def __hash__(self): ...
    def __eq__(self, other): ...
    def __str__(self) -> str: ...
    def get_family(self):
        """
        Return a list of individual font family names or generic family names.

        The font families or generic font families (which will be resolved
        from their respective rcParams when searching for a matching font) in
        the order of preference.
        """
    def get_name(self):
        """
        Return the name of the font that best matches the font properties.
        """
    def get_style(self):
        """
        Return the font style.  Values are: 'normal', 'italic' or 'oblique'.
        """
    def get_variant(self):
        """
        Return the font variant.  Values are: 'normal' or 'small-caps'.
        """
    def get_weight(self):
        """
        Set the font weight.  Options are: A numeric value in the
        range 0-1000 or one of 'light', 'normal', 'regular', 'book',
        'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
        'heavy', 'extra bold', 'black'
        """
    def get_stretch(self):
        """
        Return the font stretch or width.  Options are: 'ultra-condensed',
        'extra-condensed', 'condensed', 'semi-condensed', 'normal',
        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.
        """
    def get_size(self):
        """
        Return the font size.
        """
    def get_file(self):
        """
        Return the filename of the associated font.
        """
    def get_fontconfig_pattern(self):
        """
        Get a fontconfig_ pattern_ suitable for looking up the font as
        specified with fontconfig's ``fc-match`` utility.

        This support does not depend on fontconfig; we are merely borrowing its
        pattern syntax for use here.
        """
    _family: Incomplete
    def set_family(self, family) -> None:
        """
        Change the font family.  Can be either an alias (generic name
        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
        'fantasy', or 'monospace', a real font name or a list of real
        font names.  Real font names are not supported when
        :rc:`text.usetex` is `True`. Default: :rc:`font.family`
        """
    _slant: Incomplete
    def set_style(self, style) -> None:
        """
        Set the font style.

        Parameters
        ----------
        style : {'normal', 'italic', 'oblique'}, default: :rc:`font.style`
        """
    _variant: Incomplete
    def set_variant(self, variant) -> None:
        """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}, default: :rc:`font.variant`
        """
    _weight: Incomplete
    def set_weight(self, weight) -> None:
        """
        Set the font weight.

        Parameters
        ----------
        weight : int or {'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}, default: :rc:`font.weight`
            If int, must be in the range  0-1000.
        """
    _stretch: Incomplete
    def set_stretch(self, stretch) -> None:
        """
        Set the font stretch or width.

        Parameters
        ----------
        stretch : int or {'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}, default: :rc:`font.stretch`
            If int, must be in the range  0-1000.
        """
    _size: Incomplete
    def set_size(self, size) -> None:
        """
        Set the font size.

        Parameters
        ----------
        size : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, default: :rc:`font.size`
            If a float, the font size in points. The string values denote
            sizes relative to the default font size.
        """
    _file: Incomplete
    def set_file(self, file) -> None:
        """
        Set the filename of the fontfile to use.  In this case, all
        other properties will be ignored.
        """
    def set_fontconfig_pattern(self, pattern) -> None:
        """
        Set the properties by parsing a fontconfig_ *pattern*.

        This support does not depend on fontconfig; we are merely borrowing its
        pattern syntax for use here.
        """
    def get_math_fontfamily(self):
        """
        Return the name of the font family used for math text.

        The default font is :rc:`mathtext.fontset`.
        """
    _math_fontfamily: Incomplete
    def set_math_fontfamily(self, fontfamily) -> None:
        """
        Set the font family for text in math mode.

        If not set explicitly, :rc:`mathtext.fontset` will be used.

        Parameters
        ----------
        fontfamily : str
            The name of the font family.

            Available font families are defined in the
            :ref:`default matplotlibrc file <customizing-with-matplotlibrc-files>`.

        See Also
        --------
        .text.Text.get_math_fontfamily
        """
    def copy(self):
        """Return a copy of self."""
    set_name = set_family
    get_slant = get_style
    set_slant = set_style
    get_size_in_points = get_size

class _JSONEncoder(json.JSONEncoder):
    def default(self, o): ...

def _json_decode(o): ...
def json_dump(data, filename) -> None:
    """
    Dump `FontManager` *data* as JSON to the file named *filename*.

    See Also
    --------
    json_load

    Notes
    -----
    File paths that are children of the Matplotlib data path (typically, fonts
    shipped with Matplotlib) are stored relative to that data path (to remain
    valid across virtualenvs).

    This function temporarily locks the output file to prevent multiple
    processes from overwriting one another's output.
    """
def json_load(filename):
    """
    Load a `FontManager` from the JSON file named *filename*.

    See Also
    --------
    json_dump
    """

class FontManager:
    '''
    On import, the `FontManager` singleton instance creates a list of ttf and
    afm fonts and caches their `FontProperties`.  The `FontManager.findfont`
    method does a nearest neighbor search to find the font that most closely
    matches the specification.  If no good enough match is found, the default
    font is returned.

    Fonts added with the `FontManager.addfont` method will not persist in the
    cache; therefore, `addfont` will need to be called every time Matplotlib is
    imported. This method should only be used if and when a font cannot be
    installed on your operating system by other means.

    Notes
    -----
    The `FontManager.addfont` method must be called on the global `FontManager`
    instance.

    Example usage::

        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        font_dirs = ["/resources/fonts"]  # The path to the custom font file.
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
    '''
    __version__: int
    _version: Incomplete
    __default_weight: Incomplete
    default_size: Incomplete
    defaultFamily: Incomplete
    afmlist: Incomplete
    ttflist: Incomplete
    def __init__(self, size: Incomplete | None = None, weight: str = 'normal') -> None: ...
    def addfont(self, path) -> None:
        """
        Cache the properties of the font at *path* to make it available to the
        `FontManager`.  The type of font is inferred from the path suffix.

        Parameters
        ----------
        path : str or path-like

        Notes
        -----
        This method is useful for adding a custom font without installing it in
        your operating system. See the `FontManager` singleton instance for
        usage and caveats about this function.
        """
    @property
    def defaultFont(self): ...
    def get_default_weight(self):
        """
        Return the default font weight.
        """
    @staticmethod
    def get_default_size():
        """
        Return the default font size.
        """
    def set_default_weight(self, weight) -> None:
        """
        Set the default font weight.  The initial value is 'normal'.
        """
    @staticmethod
    def _expand_aliases(family): ...
    def score_family(self, families, family2):
        """
        Return a match score between the list of font families in
        *families* and the font family name *family2*.

        An exact match at the head of the list returns 0.0.

        A match further down the list will return between 0 and 1.

        No match will return 1.0.
        """
    def score_style(self, style1, style2):
        """
        Return a match score between *style1* and *style2*.

        An exact match returns 0.0.

        A match between 'italic' and 'oblique' returns 0.1.

        No match returns 1.0.
        """
    def score_variant(self, variant1, variant2):
        """
        Return a match score between *variant1* and *variant2*.

        An exact match returns 0.0, otherwise 1.0.
        """
    def score_stretch(self, stretch1, stretch2):
        """
        Return a match score between *stretch1* and *stretch2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *stretch1* and *stretch2*, normalized
        between 0.0 and 1.0.
        """
    def score_weight(self, weight1, weight2):
        """
        Return a match score between *weight1* and *weight2*.

        The result is 0.0 if both weight1 and weight 2 are given as strings
        and have the same value.

        Otherwise, the result is the absolute value of the difference between
        the CSS numeric values of *weight1* and *weight2*, normalized between
        0.05 and 1.0.
        """
    def score_size(self, size1, size2):
        """
        Return a match score between *size1* and *size2*.

        If *size2* (the size specified in the font file) is 'scalable', this
        function always returns 0.0, since any font size can be generated.

        Otherwise, the result is the absolute distance between *size1* and
        *size2*, normalized so that the usual range of font sizes (6pt -
        72pt) will lie between 0.0 and 1.0.
        """
    def findfont(self, prop, fontext: str = 'ttf', directory: Incomplete | None = None, fallback_to_default: bool = True, rebuild_if_missing: bool = True):
        '''
        Find the path to the font file most closely matching the given font properties.

        Parameters
        ----------
        prop : str or `~matplotlib.font_manager.FontProperties`
            The font properties to search for. This can be either a
            `.FontProperties` object or a string defining a
            `fontconfig patterns`_.

        fontext : {\'ttf\', \'afm\'}, default: \'ttf\'
            The extension of the font file:

            - \'ttf\': TrueType and OpenType fonts (.ttf, .ttc, .otf)
            - \'afm\': Adobe Font Metrics (.afm)

        directory : str, optional
            If given, only search this directory and its subdirectories.

        fallback_to_default : bool
            If True, will fall back to the default font family (usually
            "DejaVu Sans" or "Helvetica") if the first lookup hard-fails.

        rebuild_if_missing : bool
            Whether to rebuild the font cache and search again if the first
            match appears to point to a nonexisting font (i.e., the font cache
            contains outdated entries).

        Returns
        -------
        str
            The filename of the best matching font.

        Notes
        -----
        This performs a nearest neighbor search.  Each font is given a
        similarity score to the target font properties.  The first font with
        the highest score is returned.  If no matches below a certain
        threshold are found, the default font (usually DejaVu Sans) is
        returned.

        The result is cached, so subsequent lookups don\'t have to
        perform the O(n) nearest neighbor search.

        See the `W3C Cascading Style Sheet, Level 1
        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
        for a description of the font finding algorithm.

        .. _fontconfig patterns:
           https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
        '''
    def get_font_names(self):
        """Return the list of available fonts."""
    def _find_fonts_by_props(self, prop, fontext: str = 'ttf', directory: Incomplete | None = None, fallback_to_default: bool = True, rebuild_if_missing: bool = True):
        '''
        Find the paths to the font files most closely matching the given properties.

        Parameters
        ----------
        prop : str or `~matplotlib.font_manager.FontProperties`
            The font properties to search for. This can be either a
            `.FontProperties` object or a string defining a
            `fontconfig patterns`_.

        fontext : {\'ttf\', \'afm\'}, default: \'ttf\'
            The extension of the font file:

            - \'ttf\': TrueType and OpenType fonts (.ttf, .ttc, .otf)
            - \'afm\': Adobe Font Metrics (.afm)

        directory : str, optional
            If given, only search this directory and its subdirectories.

        fallback_to_default : bool
            If True, will fall back to the default font family (usually
            "DejaVu Sans" or "Helvetica") if none of the families were found.

        rebuild_if_missing : bool
            Whether to rebuild the font cache and search again if the first
            match appears to point to a nonexisting font (i.e., the font cache
            contains outdated entries).

        Returns
        -------
        list[str]
            The paths of the fonts found.

        Notes
        -----
        This is an extension/wrapper of the original findfont API, which only
        returns a single font for given font properties. Instead, this API
        returns a list of filepaths of multiple fonts which closely match the
        given font properties.  Since this internally uses the original API,
        there\'s no change to the logic of performing the nearest neighbor
        search.  See `findfont` for more details.
        '''
    def _findfont_cached(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params): ...

def is_opentype_cff_font(filename):
    """
    Return whether the given font is a Postscript Compact Font Format Font
    embedded in an OpenType wrapper.  Used by the PostScript and PDF backends
    that cannot subset these fonts.
    """
def _get_font(font_filepaths, hinting_factor, *, _kerning_factor, thread_id): ...
def _cached_realpath(path): ...
def get_font(font_filepaths, hinting_factor: Incomplete | None = None):
    """
    Get an `.ft2font.FT2Font` object given a list of file paths.

    Parameters
    ----------
    font_filepaths : Iterable[str, Path, bytes], str, Path, bytes
        Relative or absolute paths to the font files to be used.

        If a single string, bytes, or `pathlib.Path`, then it will be treated
        as a list with that entry only.

        If more than one filepath is passed, then the returned FT2Font object
        will fall back through the fonts, in the order given, to find a needed
        glyph.

    Returns
    -------
    `.ft2font.FT2Font`

    """
def _load_fontmanager(*, try_read_cache: bool = True): ...

fontManager: Incomplete
findfont: Incomplete
get_font_names: Incomplete

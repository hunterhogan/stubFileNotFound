from _typeshed import Incomplete
from matplotlib import cbook as cbook, dviread as dviread

_log: Incomplete

def _usepackage_if_not_loaded(package, *, option: Incomplete | None = None):
    """
    Output LaTeX code that loads a package (possibly with an option) if it
    hasn't been loaded yet.

    LaTeX cannot load twice a package with different options, so this helper
    can be used to protect against users loading arbitrary packages/options in
    their custom preamble.
    """

class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a directory.

    The cache directory is called ``tex.cache`` and is located in the directory
    returned by `.get_cachedir`.

    Repeated calls to this constructor always return the same instance.
    """
    _texcache: Incomplete
    _grey_arrayd: Incomplete
    _font_families: Incomplete
    _font_preambles: Incomplete
    _font_types: Incomplete
    def __new__(cls): ...
    @classmethod
    def _get_font_family_and_reduced(cls):
        """Return the font family name and whether the font is reduced."""
    @classmethod
    def _get_font_preamble_and_command(cls): ...
    @classmethod
    def get_basefile(cls, tex, fontsize, dpi: Incomplete | None = None):
        """
        Return a filename based on a hash of the string, fontsize, and dpi.
        """
    @classmethod
    def get_font_preamble(cls):
        """
        Return a string containing font configuration for the tex preamble.
        """
    @classmethod
    def get_custom_preamble(cls):
        """Return a string containing user additions to the tex preamble."""
    @classmethod
    def _get_tex_source(cls, tex, fontsize):
        """Return the complete TeX source for processing a TeX string."""
    @classmethod
    def make_tex(cls, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size.

        Return the file name.
        """
    @classmethod
    def _run_checked_subprocess(cls, command, tex, *, cwd: Incomplete | None = None): ...
    @classmethod
    def make_dvi(cls, tex, fontsize):
        """
        Generate a dvi file containing latex's layout of tex string.

        Return the file name.
        """
    @classmethod
    def make_png(cls, tex, fontsize, dpi):
        """
        Generate a png file containing latex's rendering of tex string.

        Return the file name.
        """
    @classmethod
    def get_grey(cls, tex, fontsize: Incomplete | None = None, dpi: Incomplete | None = None):
        """Return the alpha channel."""
    @classmethod
    def get_rgba(cls, tex, fontsize: Incomplete | None = None, dpi: Incomplete | None = None, rgb=(0, 0, 0)):
        '''
        Return latex\'s rendering of the tex string as an RGBA array.

        Examples
        --------
        >>> texmanager = TexManager()
        >>> s = r"\\TeX\\ is $\\displaystyle\\sum_n\\frac{-e^{i\\pi}}{2^n}$!"
        >>> Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1, 0, 0))
        '''
    @classmethod
    def get_text_width_height_descent(cls, tex, fontsize, renderer: Incomplete | None = None):
        """Return width, height and descent of the text."""

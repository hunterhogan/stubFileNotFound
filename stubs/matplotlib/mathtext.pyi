from ._mathtext import RasterParse as RasterParse, VectorParse as VectorParse, get_unicode_index as get_unicode_index
from _typeshed import Incomplete
from matplotlib import _api as _api, _mathtext as _mathtext

_log: Incomplete

class MathTextParser:
    _parser: Incomplete
    _font_type_mapping: Incomplete
    _output_type: Incomplete
    def __init__(self, output) -> None:
        '''
        Create a MathTextParser for the given backend *output*.

        Parameters
        ----------
        output : {"path", "agg"}
            Whether to return a `VectorParse` ("path") or a
            `RasterParse` ("agg", or its synonym "macosx").
        '''
    def parse(self, s, dpi: int = 72, prop: Incomplete | None = None, *, antialiased: Incomplete | None = None):
        '''
        Parse the given math expression *s* at the given *dpi*.  If *prop* is
        provided, it is a `.FontProperties` object specifying the "default"
        font to use in the math expression, used for all non-math text.

        The results are cached, so multiple calls to `parse`
        with the same expression should be fast.

        Depending on the *output* type, this returns either a `VectorParse` or
        a `RasterParse`.
        '''
    def _parse_cached(self, s, dpi, prop, antialiased, load_glyph_flags): ...

def math_to_image(s, filename_or_obj, prop: Incomplete | None = None, dpi: Incomplete | None = None, format: Incomplete | None = None, *, color: Incomplete | None = None):
    """
    Given a math expression, renders it in a closely-clipped bounding
    box to an image file.

    Parameters
    ----------
    s : str
        A math expression.  The math portion must be enclosed in dollar signs.
    filename_or_obj : str or path-like or file-like
        Where to write the image data.
    prop : `.FontProperties`, optional
        The size and style of the text.
    dpi : float, optional
        The output dpi.  If not set, the dpi is determined as for
        `.Figure.savefig`.
    format : str, optional
        The output format, e.g., 'svg', 'pdf', 'ps' or 'png'.  If not set, the
        format is determined as for `.Figure.savefig`.
    color : str, optional
        Foreground color, defaults to :rc:`text.color`.
    """

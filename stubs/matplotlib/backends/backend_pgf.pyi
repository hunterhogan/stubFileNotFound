from _typeshed import Incomplete
from matplotlib import cbook as cbook
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, RendererBase as RendererBase, _Backend as _Backend
from matplotlib.backends.backend_mixed import MixedModeRenderer as MixedModeRenderer
from matplotlib.backends.backend_pdf import _create_pdf_info_dict as _create_pdf_info_dict, _datetime_to_pdf as _datetime_to_pdf
from matplotlib.figure import Figure as Figure
from matplotlib.font_manager import FontProperties as FontProperties
from matplotlib.path import Path as Path

_log: Incomplete
_DOCUMENTCLASS: str

def _get_preamble():
    """Prepare a LaTeX preamble based on the rcParams configuration."""

latex_pt_to_in: Incomplete
latex_in_to_pt: Incomplete
mpl_pt_to_in: Incomplete
mpl_in_to_pt: Incomplete

def _tex_escape(text):
    """
    Do some necessary and/or useful substitutions for texts to be included in
    LaTeX documents.
    """
def _writeln(fh, line) -> None: ...
def _escape_and_apply_props(s, prop):
    """
    Generate a TeX string that renders string *s* with font properties *prop*,
    also applying any required escapes to *s*.
    """
def _metadata_to_str(key, value):
    """Convert metadata key/value to a form that hyperref accepts."""
def make_pdf_to_png_converter():
    """Return a function that converts a pdf file to a png file."""

class LatexError(Exception):
    latex_output: Incomplete
    def __init__(self, message, latex_output: str = '') -> None: ...
    def __str__(self) -> str: ...

class LatexManager:
    """
    The LatexManager opens an instance of the LaTeX application for
    determining the metrics of text elements. The LaTeX environment can be
    modified by setting fonts and/or a custom preamble in `.rcParams`.
    """
    @staticmethod
    def _build_latex_header(): ...
    @classmethod
    def _get_cached_or_new(cls):
        """
        Return the previous LatexManager if the header and tex system did not
        change, or a new instance otherwise.
        """
    @classmethod
    def _get_cached_or_new_impl(cls, header): ...
    def _stdin_writeln(self, s) -> None: ...
    latex: Incomplete
    def _expect(self, s): ...
    def _expect_prompt(self): ...
    _tmpdir: Incomplete
    tmpdir: Incomplete
    _finalize_tmpdir: Incomplete
    def __init__(self) -> None: ...
    _finalize_latex: Incomplete
    def _setup_latex_process(self, *, expect_reply: bool = True) -> None: ...
    def get_width_height_descent(self, text, prop):
        """
        Get the width, total height, and descent (in TeX points) for a text
        typeset by the current LaTeX environment.
        """
    def _get_box_metrics(self, tex):
        """
        Get the width, total height and descent (in TeX points) for a TeX
        command's output in the current LaTeX environment.
        """

def _get_image_inclusion_command(): ...

class RendererPgf(RendererBase):
    dpi: Incomplete
    fh: Incomplete
    figure: Incomplete
    image_counter: int
    def __init__(self, figure, fh) -> None:
        """
        Create a new PGF renderer that translates any drawing instruction
        into text commands to be interpreted in a latex pgfpicture environment.

        Attributes
        ----------
        figure : `~matplotlib.figure.Figure`
            Matplotlib figure to initialize height, width and dpi from.
        fh : file-like
            File handle for the output of the drawing commands.
        """
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace: Incomplete | None = None) -> None: ...
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def _print_pgf_clip(self, gc) -> None: ...
    def _print_pgf_path_styles(self, gc, rgbFace) -> None: ...
    def _print_pgf_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def _pgf_path_draw(self, stroke: bool = True, fill: bool = False) -> None: ...
    def option_scale_image(self): ...
    def option_image_nocomposite(self): ...
    def draw_image(self, gc, x, y, im, transform: Incomplete | None = None) -> None: ...
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext: Incomplete | None = None) -> None: ...
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None) -> None: ...
    def get_text_width_height_descent(self, s, prop, ismath): ...
    def flipy(self): ...
    def get_canvas_width_height(self): ...
    def points_to_pixels(self, points): ...

class FigureCanvasPgf(FigureCanvasBase):
    filetypes: Incomplete
    def get_default_filetype(self): ...
    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore: Incomplete | None = None) -> None: ...
    def print_pgf(self, fname_or_fh, **kwargs) -> None:
        """
        Output pgf macros for drawing the figure so it can be included and
        rendered in latex documents.
        """
    def print_pdf(self, fname_or_fh, *, metadata: Incomplete | None = None, **kwargs) -> None:
        """Use LaTeX to compile a pgf generated figure to pdf."""
    def print_png(self, fname_or_fh, **kwargs) -> None:
        """Use LaTeX to compile a pgf figure to pdf and convert it to png."""
    def get_renderer(self): ...
    def draw(self): ...
FigureManagerPgf = FigureManagerBase

class _BackendPgf(_Backend):
    FigureCanvas = FigureCanvasPgf

class PdfPages:
    """
    A multi-page PDF file using the pgf backend

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
    """
    _output_name: Incomplete
    _n_figures: int
    _metadata: Incomplete
    _info_dict: Incomplete
    _file: Incomplete
    def __init__(self, filename, *, metadata: Incomplete | None = None) -> None:
        """
        Create a new PdfPages object.

        Parameters
        ----------
        filename : str or path-like
            Plots using `PdfPages.savefig` will be written to a file at this
            location. Any older file with the same name is overwritten.

        metadata : dict, optional
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.

            Note that some versions of LaTeX engines may ignore the 'Producer'
            key and set it to themselves.
        """
    def _write_header(self, width_inches, height_inches) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def close(self) -> None:
        """
        Finalize this object, running LaTeX in a temporary directory
        and moving the final pdf file to *filename*.
        """
    def _run_latex(self) -> None: ...
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

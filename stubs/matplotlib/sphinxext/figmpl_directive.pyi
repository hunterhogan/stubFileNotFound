from _typeshed import Incomplete
from docutils import nodes
from docutils.parsers.rst.directives.images import Figure

class figmplnode(nodes.General, nodes.Element): ...

class FigureMpl(Figure):
    '''
    Implements a directive to allow an optional hidpi image.

    Meant to be used with the *plot_srcset* configuration option in conf.py,
    and gets set in the TEMPLATE of plot_directive.py

    e.g.::

        .. figure-mpl:: plot_directive/some_plots-1.png
            :alt: bar
            :srcset: plot_directive/some_plots-1.png,
                     plot_directive/some_plots-1.2x.png 2.00x
            :class: plot-directive

    The resulting html (at ``some_plots.html``) is::

        <img src="sphx_glr_bar_001_hidpi.png"
            srcset="_images/some_plot-1.png,
                    _images/some_plots-1.2x.png 2.00x",
            alt="bar"
            class="plot_directive" />

    Note that the handling of subdirectories is different than that used by the sphinx
    figure directive::

        .. figure-mpl:: plot_directive/nestedpage/index-1.png
            :alt: bar
            :srcset: plot_directive/nestedpage/index-1.png
                     plot_directive/nestedpage/index-1.2x.png 2.00x
            :class: plot_directive

    The resulting html (at ``nestedpage/index.html``)::

        <img src="../_images/nestedpage-index-1.png"
            srcset="../_images/nestedpage-index-1.png,
                    ../_images/_images/nestedpage-index-1.2x.png 2.00x",
            alt="bar"
            class="sphx-glr-single-img" />

    where the subdirectory is included in the image name for uniqueness.
    '''
    has_content: bool
    required_arguments: int
    optional_arguments: int
    final_argument_whitespace: bool
    option_spec: Incomplete
    def run(self): ...

def _parse_srcsetNodes(st):
    """
    parse srcset...
    """
def _copy_images_figmpl(self, node): ...
def visit_figmpl_html(self, node) -> None: ...
def visit_figmpl_latex(self, node) -> None: ...
def depart_figmpl_html(self, node) -> None: ...
def depart_figmpl_latex(self, node) -> None: ...
def figurempl_addnode(app) -> None: ...
def setup(app): ...

from _typeshed import Incomplete
from matplotlib.transforms import Affine2D as Affine2D, Bbox as Bbox, TransformedBbox as TransformedBbox

def adjust_bbox(fig, bbox_inches, fixed_dpi: Incomplete | None = None):
    """
    Temporarily adjust the figure so that only the specified area
    (bbox_inches) is saved.

    It modifies fig.bbox, fig.bbox_inches,
    fig.transFigure._boxout, and fig.patch.  While the figure size
    changes, the scale of the original figure is conserved.  A
    function which restores the original values are returned.
    """
def process_figure_for_rasterizing(fig, bbox_inches_restore, fixed_dpi: Incomplete | None = None):
    """
    A function that needs to be called when figure dpi changes during the
    drawing (e.g., rasterizing).  It recovers the bbox and re-adjust it with
    the new dpi.
    """

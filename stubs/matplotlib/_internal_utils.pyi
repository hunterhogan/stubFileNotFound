from _typeshed import Incomplete
from matplotlib.transforms import TransformNode as TransformNode

def graphviz_dump_transform(transform, dest, *, highlight: Incomplete | None = None) -> None:
    """
    Generate a graphical representation of the transform tree for *transform*
    using the :program:`dot` program (which this function depends on).  The
    output format (png, dot, etc.) is determined from the suffix of *dest*.

    Parameters
    ----------
    transform : `~matplotlib.transform.Transform`
        The represented transform.
    dest : str
        Output filename.  The extension must be one of the formats supported
        by :program:`dot`, e.g. png, svg, dot, ...
        (see https://www.graphviz.org/doc/info/output.html).
    highlight : list of `~matplotlib.transform.Transform` or None
        The transforms in the tree to be drawn in bold.
        If *None*, *transform* is highlighted.
    """

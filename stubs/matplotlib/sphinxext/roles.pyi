from _typeshed import Incomplete
from docutils import nodes

class _QueryReference(nodes.Inline, nodes.TextElement):
    """
    Wraps a reference or pending reference to add a query string.

    The query string is generated from the attributes added to this node.

    Also equivalent to a `~docutils.nodes.literal` node.
    """
    def to_query_string(self):
        """Generate query string from node attributes."""

def _visit_query_reference_node(self, node) -> None:
    """
    Resolve *node* into query strings on its ``reference`` children.

    Then act as if this is a `~docutils.nodes.literal`.
    """
def _depart_query_reference_node(self, node) -> None:
    """
    Act as if this is a `~docutils.nodes.literal`.
    """
def _rcparam_role(name, rawtext, text, lineno, inliner, options: Incomplete | None = None, content: Incomplete | None = None):
    """
    Sphinx role ``:rc:`` to highlight and link ``rcParams`` entries.

    Usage: Give the desired ``rcParams`` key as parameter.

    :code:`:rc:`figure.dpi`` will render as: :rc:`figure.dpi`
    """
def _mpltype_role(name, rawtext, text, lineno, inliner, options: Incomplete | None = None, content: Incomplete | None = None):
    """
    Sphinx role ``:mpltype:`` for custom matplotlib types.

    In Matplotlib, there are a number of type-like concepts that do not have a
    direct type representation; example: color. This role allows to properly
    highlight them in the docs and link to their definition.

    Currently supported values:

    - :code:`:mpltype:`color`` will render as: :mpltype:`color`

    """
def setup(app): ...

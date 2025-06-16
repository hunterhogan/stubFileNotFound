from _typeshed import Incomplete

__all__ = ['to_latex_raw', 'to_latex', 'write_latex']

def to_latex_raw(G, pos: str = 'pos', tikz_options: str = '', default_node_options: str = '', node_options: str = 'node_options', node_label: str = 'label', default_edge_options: str = '', edge_options: str = 'edge_options', edge_label: str = 'label', edge_label_options: str = 'edge_label_options'):
    '''Return a string of the LaTeX/TikZ code to draw `G`

    This function produces just the code for the tikzpicture
    without any enclosing environment.

    Parameters
    ==========
    G : NetworkX graph
        The NetworkX graph to be drawn
    pos : string or dict (default "pos")
        The name of the node attribute on `G` that holds the position of each node.
        Positions can be sequences of length 2 with numbers for (x,y) coordinates.
        They can also be strings to denote positions in TikZ style, such as (x, y)
        or (angle:radius).
        If a dict, it should be keyed by node to a position.
        If an empty dict, a circular layout is computed by TikZ.
    tikz_options : string
        The tikzpicture options description defining the options for the picture.
        Often large scale options like `[scale=2]`.
    default_node_options : string
        The draw options for a path of nodes. Individual node options override these.
    node_options : string or dict
        The name of the node attribute on `G` that holds the options for each node.
        Or a dict keyed by node to a string holding the options for that node.
    node_label : string or dict
        The name of the node attribute on `G` that holds the node label (text)
        displayed for each node. If the attribute is "" or not present, the node
        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
        Or a dict keyed by node to a string holding the label for that node.
    default_edge_options : string
        The options for the scope drawing all edges. The default is "[-]" for
        undirected graphs and "[->]" for directed graphs.
    edge_options : string or dict
        The name of the edge attribute on `G` that holds the options for each edge.
        If the edge is a self-loop and ``"loop" not in edge_options`` the option
        "loop," is added to the options for the self-loop edge. Hence you can
        use "[loop above]" explicitly, but the default is "[loop]".
        Or a dict keyed by edge to a string holding the options for that edge.
    edge_label : string or dict
        The name of the edge attribute on `G` that holds the edge label (text)
        displayed for each edge. If the attribute is "" or not present, no edge
        label is drawn.
        Or a dict keyed by edge to a string holding the label for that edge.
    edge_label_options : string or dict
        The name of the edge attribute on `G` that holds the label options for
        each edge. For example, "[sloped,above,blue]". The default is no options.
        Or a dict keyed by edge to a string holding the label options for that edge.

    Returns
    =======
    latex_code : string
       The text string which draws the desired graph(s) when compiled by LaTeX.

    See Also
    ========
    to_latex
    write_latex
    '''
def to_latex(Gbunch, pos: str = 'pos', tikz_options: str = '', default_node_options: str = '', node_options: str = 'node_options', node_label: str = 'node_label', default_edge_options: str = '', edge_options: str = 'edge_options', edge_label: str = 'edge_label', edge_label_options: str = 'edge_label_options', caption: str = '', latex_label: str = '', sub_captions: Incomplete | None = None, sub_labels: Incomplete | None = None, n_rows: int = 1, as_document: bool = True, document_wrapper='\\documentclass{{report}}\n\\usepackage{{tikz}}\n\\usepackage{{subcaption}}\n\n\\begin{{document}}\n{content}\n\\end{{document}}', figure_wrapper='\\begin{{figure}}\n{content}{caption}{label}\n\\end{{figure}}', subfigure_wrapper='  \\begin{{subfigure}}{{{size}\\textwidth}}\n{content}{caption}{label}\n  \\end{{subfigure}}'):
    '''Return latex code to draw the graph(s) in `Gbunch`

    The TikZ drawing utility in LaTeX is used to draw the graph(s).
    If `Gbunch` is a graph, it is drawn in a figure environment.
    If `Gbunch` is an iterable of graphs, each is drawn in a subfigure environment
    within a single figure environment.

    If `as_document` is True, the figure is wrapped inside a document environment
    so that the resulting string is ready to be compiled by LaTeX. Otherwise,
    the string is ready for inclusion in a larger tex document using ``\\include``
    or ``\\input`` statements.

    Parameters
    ==========
    Gbunch : NetworkX graph or iterable of NetworkX graphs
        The NetworkX graph to be drawn or an iterable of graphs
        to be drawn inside subfigures of a single figure.
    pos : string or list of strings
        The name of the node attribute on `G` that holds the position of each node.
        Positions can be sequences of length 2 with numbers for (x,y) coordinates.
        They can also be strings to denote positions in TikZ style, such as (x, y)
        or (angle:radius).
        If a dict, it should be keyed by node to a position.
        If an empty dict, a circular layout is computed by TikZ.
        If you are drawing many graphs in subfigures, use a list of position dicts.
    tikz_options : string
        The tikzpicture options description defining the options for the picture.
        Often large scale options like `[scale=2]`.
    default_node_options : string
        The draw options for a path of nodes. Individual node options override these.
    node_options : string or dict
        The name of the node attribute on `G` that holds the options for each node.
        Or a dict keyed by node to a string holding the options for that node.
    node_label : string or dict
        The name of the node attribute on `G` that holds the node label (text)
        displayed for each node. If the attribute is "" or not present, the node
        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
        Or a dict keyed by node to a string holding the label for that node.
    default_edge_options : string
        The options for the scope drawing all edges. The default is "[-]" for
        undirected graphs and "[->]" for directed graphs.
    edge_options : string or dict
        The name of the edge attribute on `G` that holds the options for each edge.
        If the edge is a self-loop and ``"loop" not in edge_options`` the option
        "loop," is added to the options for the self-loop edge. Hence you can
        use "[loop above]" explicitly, but the default is "[loop]".
        Or a dict keyed by edge to a string holding the options for that edge.
    edge_label : string or dict
        The name of the edge attribute on `G` that holds the edge label (text)
        displayed for each edge. If the attribute is "" or not present, no edge
        label is drawn.
        Or a dict keyed by edge to a string holding the label for that edge.
    edge_label_options : string or dict
        The name of the edge attribute on `G` that holds the label options for
        each edge. For example, "[sloped,above,blue]". The default is no options.
        Or a dict keyed by edge to a string holding the label options for that edge.
    caption : string
        The caption string for the figure environment
    latex_label : string
        The latex label used for the figure for easy referral from the main text
    sub_captions : list of strings
        The sub_caption string for each subfigure in the figure
    sub_latex_labels : list of strings
        The latex label for each subfigure in the figure
    n_rows : int
        The number of rows of subfigures to arrange for multiple graphs
    as_document : bool
        Whether to wrap the latex code in a document environment for compiling
    document_wrapper : formatted text string with variable ``content``.
        This text is called to evaluate the content embedded in a document
        environment with a preamble setting up TikZ.
    figure_wrapper : formatted text string
        This text is evaluated with variables ``content``, ``caption`` and ``label``.
        It wraps the content and if a caption is provided, adds the latex code for
        that caption, and if a label is provided, adds the latex code for a label.
    subfigure_wrapper : formatted text string
        This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.
        It wraps the content and if a caption is provided, adds the latex code for
        that caption, and if a label is provided, adds the latex code for a label.
        The size is the vertical size of each row of subfigures as a fraction.

    Returns
    =======
    latex_code : string
        The text string which draws the desired graph(s) when compiled by LaTeX.

    See Also
    ========
    write_latex
    to_latex_raw
    '''
def write_latex(Gbunch, path, **options) -> None:
    '''Write the latex code to draw the graph(s) onto `path`.

    This convenience function creates the latex drawing code as a string
    and writes that to a file ready to be compiled when `as_document` is True
    or ready to be ``import`` ed or ``include`` ed into your main LaTeX document.

    The `path` argument can be a string filename or a file handle to write to.

    Parameters
    ----------
    Gbunch : NetworkX graph or iterable of NetworkX graphs
        If Gbunch is a graph, it is drawn in a figure environment.
        If Gbunch is an iterable of graphs, each is drawn in a subfigure
        environment within a single figure environment.
    path : filename
        Filename or file handle to write to
    options : dict
        By default, TikZ is used with options: (others are ignored)::

            pos : string or dict or list
                The name of the node attribute on `G` that holds the position of each node.
                Positions can be sequences of length 2 with numbers for (x,y) coordinates.
                They can also be strings to denote positions in TikZ style, such as (x, y)
                or (angle:radius).
                If a dict, it should be keyed by node to a position.
                If an empty dict, a circular layout is computed by TikZ.
                If you are drawing many graphs in subfigures, use a list of position dicts.
            tikz_options : string
                The tikzpicture options description defining the options for the picture.
                Often large scale options like `[scale=2]`.
            default_node_options : string
                The draw options for a path of nodes. Individual node options override these.
            node_options : string or dict
                The name of the node attribute on `G` that holds the options for each node.
                Or a dict keyed by node to a string holding the options for that node.
            node_label : string or dict
                The name of the node attribute on `G` that holds the node label (text)
                displayed for each node. If the attribute is "" or not present, the node
                itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
                Or a dict keyed by node to a string holding the label for that node.
            default_edge_options : string
                The options for the scope drawing all edges. The default is "[-]" for
                undirected graphs and "[->]" for directed graphs.
            edge_options : string or dict
                The name of the edge attribute on `G` that holds the options for each edge.
                If the edge is a self-loop and ``"loop" not in edge_options`` the option
                "loop," is added to the options for the self-loop edge. Hence you can
                use "[loop above]" explicitly, but the default is "[loop]".
                Or a dict keyed by edge to a string holding the options for that edge.
            edge_label : string or dict
                The name of the edge attribute on `G` that holds the edge label (text)
                displayed for each edge. If the attribute is "" or not present, no edge
                label is drawn.
                Or a dict keyed by edge to a string holding the label for that edge.
            edge_label_options : string or dict
                The name of the edge attribute on `G` that holds the label options for
                each edge. For example, "[sloped,above,blue]". The default is no options.
                Or a dict keyed by edge to a string holding the label options for that edge.
            caption : string
                The caption string for the figure environment
            latex_label : string
                The latex label used for the figure for easy referral from the main text
            sub_captions : list of strings
                The sub_caption string for each subfigure in the figure
            sub_latex_labels : list of strings
                The latex label for each subfigure in the figure
            n_rows : int
                The number of rows of subfigures to arrange for multiple graphs
            as_document : bool
                Whether to wrap the latex code in a document environment for compiling
            document_wrapper : formatted text string with variable ``content``.
                This text is called to evaluate the content embedded in a document
                environment with a preamble setting up the TikZ syntax.
            figure_wrapper : formatted text string
                This text is evaluated with variables ``content``, ``caption`` and ``label``.
                It wraps the content and if a caption is provided, adds the latex code for
                that caption, and if a label is provided, adds the latex code for a label.
            subfigure_wrapper : formatted text string
                This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.
                It wraps the content and if a caption is provided, adds the latex code for
                that caption, and if a label is provided, adds the latex code for a label.
                The size is the vertical size of each row of subfigures as a fraction.

    See Also
    ========
    to_latex
    '''

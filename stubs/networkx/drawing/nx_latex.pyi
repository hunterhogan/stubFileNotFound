__all__ = ["to_latex_raw", "to_latex", "write_latex"]

def to_latex_raw(
    G,
    pos: str = "pos",
    tikz_options: str = "",
    default_node_options: str = "",
    node_options: str = "node_options",
    node_label: str = "label",
    default_edge_options: str = "",
    edge_options: str = "edge_options",
    edge_label: str = "label",
    edge_label_options: str = "edge_label_options",
): ...
def to_latex(
    Gbunch,
    pos: str = "pos",
    tikz_options: str = "",
    default_node_options: str = "",
    node_options: str = "node_options",
    node_label: str = "node_label",
    default_edge_options: str = "",
    edge_options: str = "edge_options",
    edge_label: str = "edge_label",
    edge_label_options: str = "edge_label_options",
    caption: str = "",
    latex_label: str = "",
    sub_captions=None,
    sub_labels=None,
    n_rows: int = 1,
    as_document: bool = True,
    document_wrapper: str = '\\documentclass{{report}}\n\\usepackage{{tikz}}\n\\usepackage{{subcaption}}\n\n\\begin{{document}}\n{content}\n\\end{{document}}',
    figure_wrapper: str = '\\begin{{figure}}\n{content}{caption}{label}\n\\end{{figure}}',
    subfigure_wrapper: str = '  \\begin{{subfigure}}{{{size}\\textwidth}}\n{content}{caption}{label}\n  \\end{{subfigure}}',
): ...
def write_latex(Gbunch, path, **options) -> None: ...

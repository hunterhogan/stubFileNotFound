from _typeshed import Incomplete

def make_flex_doc(op_name: str, typ: str) -> str:
    """
    Make the appropriate substitutions for the given operation and class-typ
    into either _flex_doc_SERIES or _flex_doc_FRAME to return the docstring
    to attach to a generated method.

    Parameters
    ----------
    op_name : str {'__add__', '__sub__', ... '__eq__', '__ne__', ...}
    typ : str {series, 'dataframe']}

    Returns
    -------
    doc : str
    """

_common_examples_algebra_SERIES: str
_common_examples_comparison_SERIES: str
_add_example_SERIES: Incomplete
_sub_example_SERIES: Incomplete
_mul_example_SERIES: Incomplete
_div_example_SERIES: Incomplete
_floordiv_example_SERIES: Incomplete
_divmod_example_SERIES: Incomplete
_mod_example_SERIES: Incomplete
_pow_example_SERIES: Incomplete
_ne_example_SERIES: Incomplete
_eq_example_SERIES: Incomplete
_lt_example_SERIES: Incomplete
_le_example_SERIES: Incomplete
_gt_example_SERIES: Incomplete
_ge_example_SERIES: Incomplete
_returns_series: str
_returns_tuple: str
_op_descriptions: dict[str, dict[str, str | None]]
_py_num_ref: str
_op_names: Incomplete
reverse_op: Incomplete
_flex_doc_SERIES: str
_see_also_reverse_SERIES: str
_flex_doc_FRAME: str
_flex_comp_doc_FRAME: str

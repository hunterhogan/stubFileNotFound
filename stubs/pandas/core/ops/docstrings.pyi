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
_add_example_SERIES: str
_sub_example_SERIES: str
_mul_example_SERIES: str
_div_example_SERIES: str
_floordiv_example_SERIES: str
_divmod_example_SERIES: str
_mod_example_SERIES: str
_pow_example_SERIES: str
_ne_example_SERIES: str
_eq_example_SERIES: str
_lt_example_SERIES: str
_le_example_SERIES: str
_gt_example_SERIES: str
_ge_example_SERIES: str
_returns_series: str
_returns_tuple: str
_op_descriptions: dict
_py_num_ref: str
_op_names: list
key: str
reverse_op: None
_flex_doc_SERIES: str
_see_also_reverse_SERIES: str
_flex_doc_FRAME: str
_flex_comp_doc_FRAME: str

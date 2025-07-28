from collections.abc import Callable, Mapping
from typing import Any
import _cython_3_0_11

__pyx_capi__: dict
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
assoc: _cython_3_0_11.cython_function_or_method
assoc_in: _cython_3_0_11.cython_function_or_method
dissoc: _cython_3_0_11.cython_function_or_method
get_in: _cython_3_0_11.cython_function_or_method
itemfilter: _cython_3_0_11.cython_function_or_method
itemmap: _cython_3_0_11.cython_function_or_method
keyfilter: _cython_3_0_11.cython_function_or_method
keymap: _cython_3_0_11.cython_function_or_method
merge: _cython_3_0_11.cython_function_or_method

def merge_with[文件, 文义](func: Callable[[list[Any]], Any], *dicts: Mapping[文件, 文义], **kwargs: Any) -> dict[文件, 文义]:
    """Merge dictionaries and apply function to combined values.

    A key may occur in more than one dictionary, and all values mapped from the key will be passed to the `func` parameter as a
    list. For example, `func([val1, val2, ...])` is called for each key.

    Parameters
    ----------
    func : Callable[[list[Any]], Any]
        Function applied to the list of values for each key.
    dicts : Mapping[Any, Any]
        Dictionaries to merge.
    keywordArguments : Any
        Additional keyword arguments (unused).

    Returns
    -------
    mergedDictionary : dict[Any, Any]
        Dictionary with merged keys and values processed by `func`.

    Examples
    --------
    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    {1: 11, 2: 22}

    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})
    {1: 1, 2: 2, 3: 30}

    See Also
    --------
    merge
    """

update_in: _cython_3_0_11.cython_function_or_method
valfilter: _cython_3_0_11.cython_function_or_method
valmap: _cython_3_0_11.cython_function_or_method

class _iter_mapping:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_iter_mapping.__reduce_cython__(self)"""

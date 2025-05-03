import pandas.core.common as com
from functools import partial
from pandas.core.base import PandasObject as PandasObject
from pandas.core.computation.common import result_type_many as result_type_many
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Callable

TYPE_CHECKING: bool
def _align_core_single_unary_op(term) -> tuple[partial | type[NDFrame], dict[str, Index] | None]: ...
def _zip_axes_from_type(typ: type[NDFrame], new_axes: Sequence[Index]) -> dict[str, Index]: ...
def _any_pandas_objects(terms) -> bool:
    """
    Check a sequence of terms for instances of PandasObject.
    """
def _filter_special_cases(f) -> Callable[[F], F]: ...
def _align_core(terms): ...
def align_terms(terms):
    """
    Align a set of terms.
    """
def reconstruct_object(typ, obj, axes, dtype):
    """
    Reconstruct an object given its type, raw value, and possibly empty
    (None) axes.

    Parameters
    ----------
    typ : object
        A type
    obj : object
        The value to use in the type constructor
    axes : dict
        The axes to use to construct the resulting pandas object

    Returns
    -------
    ret : typ
        An object of type ``typ`` with the value `obj` and possible axes
        `axes`.
    """

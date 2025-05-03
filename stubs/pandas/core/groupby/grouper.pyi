import lib as lib
import np
import npt
import ops as ops
import pandas.core.algorithms as algorithms
import pandas.core.common as com
from _typeshed import Incomplete
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs.lib import is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.groupby.categorical import recode_for_groupby as recode_for_groupby
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.category import CategoricalIndex as CategoricalIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.series import Series as Series
from pandas.errors import InvalidIndexError as InvalidIndexError
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

TYPE_CHECKING: bool

class Grouper:
    _attributes: ClassVar[tuple] = ...
    def __init__(self, key, level, freq, axis: Axis | lib.NoDefault = ..., sort: bool = ..., dropna: bool = ...) -> None: ...
    def _get_grouper(self, obj: NDFrameT, validate: bool = ...) -> tuple[ops.BaseGrouper, NDFrameT]:
        """
        Parameters
        ----------
        obj : Series or DataFrame
        validate : bool, default True
            if True, validate the grouper

        Returns
        -------
        a tuple of grouper, obj (possibly sorted)
        """
    def _set_grouper(self, obj: NDFrameT, sort: bool = ..., *, gpr_index: Index | None) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        """
        given an object and the specifications, setup the internal grouper
        for this particular specification

        Parameters
        ----------
        obj : Series or DataFrame
        sort : bool, default False
            whether the resulting grouper should be sorted
        gpr_index : Index or None, default None

        Returns
        -------
        NDFrame
        Index
        np.ndarray[np.intp] | None
        """
    @property
    def ax(self): ...
    @property
    def indexer(self): ...
    @property
    def obj(self): ...
    @property
    def grouper(self): ...
    @property
    def groups(self): ...

class Grouping:
    _codes: ClassVar[None] = ...
    __final__: ClassVar[bool] = ...
    _passed_categorical: Incomplete
    name: Incomplete
    _ilevel: Incomplete
    indices: Incomplete
    _group_arraylike: Incomplete
    _result_index: Incomplete
    _group_index: Incomplete
    _codes_and_uniques: Incomplete
    groups: Incomplete
    def __init__(self, index: Index, grouper, obj: NDFrame | None, level, sort: bool = ..., observed: bool = ..., in_axis: bool = ..., dropna: bool = ..., uniques: ArrayLike | None) -> None: ...
    def __iter__(self) -> Iterator: ...
    @property
    def ngroups(self): ...
    @property
    def codes(self): ...
    @property
    def group_arraylike(self): ...
    @property
    def result_index(self): ...
    @property
    def group_index(self): ...
def get_grouper(obj: NDFrameT, key, axis: Axis = ..., level, sort: bool = ..., observed: bool = ..., validate: bool = ..., dropna: bool = ...) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to axis,level,sort, while
    the passed in axis, level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
def _is_label_like(val) -> bool: ...
def _convert_grouper(axis: Index, grouper): ...

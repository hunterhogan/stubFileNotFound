import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs import lib as lib
from pandas._typing import ArrayLike as ArrayLike, Axis as Axis, NDFrameT as NDFrameT, npt as npt
from pandas.core.arrays import Categorical as Categorical, ExtensionArray as ExtensionArray
from pandas.core.dtypes.common import is_list_like as is_list_like, is_scalar as is_scalar
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby import ops as ops
from pandas.core.indexes.api import CategoricalIndex as CategoricalIndex, Index as Index, MultiIndex as MultiIndex

class Grouper:
    '''
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level and/or axis parameters are given, a level of the index of the target
    object.

    If `axis` and/or `level` are passed as keywords to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    axis : str, int, defaults to 0
        Number/name of the axis.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {\'left\' or \'right\'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {\'left\' or \'right\'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {\'start\', \'end\', \'e\', \'s\'}
        If grouper is PeriodIndex and `freq` parameter is passed.

    origin : Timestamp or str, default \'start_day\'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If string, must be one of the following:

        - \'epoch\': `origin` is 1970-01-01
        - \'start\': `origin` is the first value of the timeseries
        - \'start_day\': `origin` is the first day at midnight of the timeseries

        - \'end\': `origin` is the last value of the timeseries
        - \'end_day\': `origin` is the ceiling midnight of the last day

        .. versionadded:: 1.3.0

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

    dropna : bool, default True
        If True, and if group keys contain NA values, NA values together with
        row/column will be dropped. If False, NA values will also be treated as
        the key in groups.

    Returns
    -------
    Grouper or pandas.api.typing.TimeGrouper
        A TimeGrouper is returned if ``freq`` is not ``None``. Otherwise, a Grouper
        is returned.

    Examples
    --------
    ``df.groupby(pd.Grouper(key="Animal"))`` is equivalent to ``df.groupby(\'Animal\')``

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column \'Publish date\'

    >>> df = pd.DataFrame(
    ...    {
    ...        "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16")
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40]
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = \'2000-10-01 23:30:00\', \'2000-10-02 00:30:00\'
    >>> rng = pd.date_range(start, end, freq=\'7min\')
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
    2000-10-02 00:05:00    15
    2000-10-02 00:12:00    18
    2000-10-02 00:19:00    21
    2000-10-02 00:26:00    24
    Freq: 7min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq=\'17min\')).sum()
    2000-10-01 23:14:00     0
    2000-10-01 23:31:00     9
    2000-10-01 23:48:00    21
    2000-10-02 00:05:00    54
    2000-10-02 00:22:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq=\'17min\', origin=\'epoch\')).sum()
    2000-10-01 23:18:00     0
    2000-10-01 23:35:00    18
    2000-10-01 23:52:00    27
    2000-10-02 00:09:00    39
    2000-10-02 00:26:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq=\'17min\', origin=\'2000-01-01\')).sum()
    2000-10-01 23:24:00     3
    2000-10-01 23:41:00    15
    2000-10-01 23:58:00    45
    2000-10-02 00:15:00    45
    Freq: 17min, dtype: int64

    If you want to adjust the start of the bins with an `offset` Timedelta, the two
    following lines are equivalent:

    >>> ts.groupby(pd.Grouper(freq=\'17min\', origin=\'start\')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq=\'17min\', offset=\'23h30min\')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    To replace the use of the deprecated `base` argument, you can now use `offset`,
    in this example it is equivalent to have `base=2`:

    >>> ts.groupby(pd.Grouper(freq=\'17min\', offset=\'2min\')).sum()
    2000-10-01 23:16:00     0
    2000-10-01 23:33:00     9
    2000-10-01 23:50:00    36
    2000-10-02 00:07:00    39
    2000-10-02 00:24:00    24
    Freq: 17min, dtype: int64
    '''
    sort: bool
    dropna: bool
    _gpr_index: Index | None
    _grouper: Index | None
    _attributes: tuple[str, ...]
    def __new__(cls, *args, **kwargs): ...
    key: Incomplete
    level: Incomplete
    freq: Incomplete
    axis: Incomplete
    _grouper_deprecated: Incomplete
    _indexer_deprecated: npt.NDArray[np.intp] | None
    _obj_deprecated: Incomplete
    binner: Incomplete
    _indexer: npt.NDArray[np.intp] | None
    def __init__(self, key: Incomplete | None = None, level: Incomplete | None = None, freq: Incomplete | None = None, axis: Axis | lib.NoDefault = ..., sort: bool = False, dropna: bool = True) -> None: ...
    def _get_grouper(self, obj: NDFrameT, validate: bool = True) -> tuple[ops.BaseGrouper, NDFrameT]:
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
    def _set_grouper(self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
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
    def ax(self) -> Index: ...
    @property
    def indexer(self): ...
    @property
    def obj(self): ...
    @property
    def grouper(self): ...
    @property
    def groups(self): ...
    def __repr__(self) -> str: ...

class Grouping:
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
    grouper :
    obj : DataFrame or Series
    name : Label
    level :
    observed : bool, default False
        If we are a Categorical, use the observed values
    in_axis : if the Grouping is a column in self.obj and hence among
        Groupby.exclusions list
    dropna : bool, default True
        Whether to drop NA groups.
    uniques : Array-like, optional
        When specified, will be used for unique values. Enables including empty groups
        in the result for a BinGrouper. Must not contain duplicates.

    Attributes
    -------
    indices : dict
        Mapping of {group -> index_list}
    codes : ndarray
        Group codes
    group_index : Index or None
        unique groups
    groups : dict
        Mapping of {group -> label_list}
    """
    _codes: npt.NDArray[np.signedinteger] | None
    _all_grouper: Categorical | None
    _orig_cats: Index | None
    _index: Index
    level: Incomplete
    _orig_grouper: Incomplete
    _sort: Incomplete
    obj: Incomplete
    _observed: Incomplete
    in_axis: Incomplete
    _dropna: Incomplete
    _uniques: Incomplete
    grouping_vector: Incomplete
    def __init__(self, index: Index, grouper: Incomplete | None = None, obj: NDFrame | None = None, level: Incomplete | None = None, sort: bool = True, observed: bool = False, in_axis: bool = False, dropna: bool = True, uniques: ArrayLike | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __iter__(self) -> Iterator: ...
    def _passed_categorical(self) -> bool: ...
    def name(self) -> Hashable: ...
    def _ilevel(self) -> int | None:
        """
        If necessary, converted index level name to index level position.
        """
    @property
    def ngroups(self) -> int: ...
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]: ...
    @property
    def codes(self) -> npt.NDArray[np.signedinteger]: ...
    def _group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
    @property
    def group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
    def _result_index(self) -> Index: ...
    @property
    def result_index(self) -> Index: ...
    def _group_index(self) -> Index: ...
    @property
    def group_index(self) -> Index: ...
    def _codes_and_uniques(self) -> tuple[npt.NDArray[np.signedinteger], ArrayLike]: ...
    def groups(self) -> dict[Hashable, np.ndarray]: ...

def get_grouper(obj: NDFrameT, key: Incomplete | None = None, axis: Axis = 0, level: Incomplete | None = None, sort: bool = True, observed: bool = False, validate: bool = True, dropna: bool = True) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
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

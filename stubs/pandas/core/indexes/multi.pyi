import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator, Hashable, Iterable, Sequence
from pandas import CategoricalIndex as CategoricalIndex, DataFrame as DataFrame, Series as Series
from pandas._config import get_option as get_option
from pandas._libs import index as libindex, lib as lib
from pandas._libs.hashtable import duplicated as duplicated
from pandas._typing import AnyAll as AnyAll, AnyArrayLike as AnyArrayLike, Axis as Axis, DropKeep as DropKeep, DtypeObj as DtypeObj, F as F, IgnoreRaise as IgnoreRaise, IndexLabel as IndexLabel, Scalar as Scalar, Self as Self, Shape as Shape, npt as npt
from pandas.core.array_algos.putmask import validate_putmask as validate_putmask
from pandas.core.arrays import Categorical as Categorical, ExtensionArray as ExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterables as factorize_from_iterables, recode_for_categories as recode_for_categories
from pandas.core.construction import sanitize_array as sanitize_array
from pandas.core.dtypes.cast import coerce_indexer_dtype as coerce_indexer_dtype
from pandas.core.dtypes.common import ensure_int64 as ensure_int64, ensure_platform_int as ensure_platform_int, is_hashable as is_hashable, is_integer as is_integer, is_iterator as is_iterator, is_list_like as is_list_like, is_object_dtype as is_object_dtype, is_scalar as is_scalar, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.inference import is_array_like as is_array_like
from pandas.core.dtypes.missing import array_equivalent as array_equivalent, isna as isna
from pandas.core.indexes.base import Index as Index, _index_shared_docs as _index_shared_docs, ensure_index as ensure_index, get_unanimous_names as get_unanimous_names
from pandas.core.indexes.frozen import FrozenList as FrozenList
from pandas.core.ops.invalid import make_invalid_op as make_invalid_op
from pandas.core.sorting import get_group_index as get_group_index, lexsort_indexer as lexsort_indexer
from pandas.errors import InvalidIndexError as InvalidIndexError, PerformanceWarning as PerformanceWarning, UnsortedIndexError as UnsortedIndexError
from pandas.io.formats.printing import get_adjustment as get_adjustment, pprint_thing as pprint_thing
from pandas.util._decorators import Appender as Appender, cache_readonly as cache_readonly, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Literal

from collections.abc import Callable

_index_doc_kwargs: Incomplete

class MultiIndexUIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    """
    This class manages a MultiIndex by mapping label combinations to positive
    integers.
    """
    _base: Incomplete
    def _codes_to_ints(self, codes):
        """
        Transform combination(s) of uint64 in one uint64 (each), in a strictly
        monotonic way (i.e. respecting the lexicographic order of integer
        combinations): see BaseMultiIndexCodesEngine documentation.

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint64
            Combinations of integers (one per row)

        Returns
        -------
        scalar or 1-dimensional array, of dtype uint64
            Integer(s) representing one combination (each).
        """

class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    """
    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """
    _base: Incomplete
    def _codes_to_ints(self, codes):
        """
        Transform combination(s) of uint64 in one Python integer (each), in a
        strictly monotonic way (i.e. respecting the lexicographic order of
        integer combinations): see BaseMultiIndexCodesEngine documentation.

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint64
            Combinations of integers (one per row)

        Returns
        -------
        int, or 1-dimensional array of dtype object
            Integer(s) representing one combination (each).
        """

def names_compat(meth: F) -> F:
    """
    A decorator to allow either `name` or `names` keyword but not both.

    This makes it easier to share code with base class.
    """

class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape
    dtypes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_indexer
    get_loc
    get_locs
    get_loc_level
    drop

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : The base pandas Index type.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
    for more.

    Examples
    --------
    A new ``MultiIndex`` is typically constructed using one of the helper
    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`
    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):

    >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
    >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               names=['number', 'color'])

    See further examples for how to construct a MultiIndex in the doc strings
    of the mentioned helper methods.
    """
    _hidden_attrs: Incomplete
    _typ: str
    _names: list[Hashable | None]
    _levels: Incomplete
    _codes: Incomplete
    _comparables: Incomplete
    sortorder: int | None
    def __new__(cls, levels: Incomplete | None = None, codes: Incomplete | None = None, sortorder: Incomplete | None = None, names: Incomplete | None = None, dtype: Incomplete | None = None, copy: bool = False, name: Incomplete | None = None, verify_integrity: bool = True) -> Self: ...
    def _validate_codes(self, level: list, code: list):
        """
        Reassign code values as -1 if their corresponding levels are NaN.

        Parameters
        ----------
        code : list
            Code to reassign.
        level : list
            Level to check for missing values (NaN, NaT, None).

        Returns
        -------
        new code where code value = -1 if it corresponds
        to a level with missing values (NaN, NaT, None).
        """
    def _verify_integrity(self, codes: list | None = None, levels: list | None = None, levels_to_verify: list[int] | range | None = None):
        """
        Parameters
        ----------
        codes : optional list
            Codes to check for validity. Defaults to current codes.
        levels : optional list
            Levels to check for validity. Defaults to current levels.
        levels_to_validate: optional list
            Specifies the levels to verify.

        Raises
        ------
        ValueError
            If length of levels and codes don't match, if the codes for any
            level would exceed level bounds, or there are any duplicate levels.

        Returns
        -------
        new codes where code value = -1 if it corresponds to a
        NaN level.
        """
    @classmethod
    def from_arrays(cls, arrays, sortorder: int | None = None, names: Sequence[Hashable] | Hashable | lib.NoDefault = ...) -> MultiIndex:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
    @classmethod
    def from_tuples(cls, tuples: Iterable[tuple[Hashable, ...]], sortorder: int | None = None, names: Sequence[Hashable] | Hashable | None = None) -> MultiIndex:
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
    @classmethod
    def from_product(cls, iterables: Sequence[Iterable[Hashable]], sortorder: int | None = None, names: Sequence[Hashable] | Hashable | lib.NoDefault = ...) -> MultiIndex:
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> pd.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
    @classmethod
    def from_frame(cls, df: DataFrame, sortorder: int | None = None, names: Sequence[Hashable] | Hashable | None = None) -> MultiIndex:
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = pd.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> pd.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
    def _values(self) -> np.ndarray: ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def array(self) -> None:
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
    def dtypes(self) -> Series:
        """
        Return the dtypes as a Series for the underlying MultiIndex.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=['number', 'color'])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.dtypes
        number     int64
        color     object
        dtype: object
        """
    def __len__(self) -> int: ...
    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.
        """
    def levels(self) -> FrozenList:
        '''
        Levels of the MultiIndex.

        Levels refer to the different hierarchical levels or layers in a MultiIndex.
        In a MultiIndex, each level represents a distinct dimension or category of
        the index.

        To access the levels, you can use the levels attribute of the MultiIndex,
        which returns a tuple of Index objects. Each Index object represents a
        level in the MultiIndex and contains the unique values found in that
        specific level.

        If a MultiIndex is created with levels A, B, C, and the DataFrame using
        it filters out all rows of the level C, MultiIndex.levels will still
        return A, B, C.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product([[\'mammal\'],
        ...                                     (\'goat\', \'human\', \'cat\', \'dog\')],
        ...                                    names=[\'Category\', \'Animals\'])
        >>> leg_num = pd.DataFrame(data=(4, 2, 4, 4), index=index, columns=[\'Legs\'])
        >>> leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 human       2
                 cat         4
                 dog         4

        >>> leg_num.index.levels
        FrozenList([[\'mammal\'], [\'cat\', \'dog\', \'goat\', \'human\']])

        MultiIndex levels will not change even if the DataFrame using the MultiIndex
        does not contain all them anymore.
        See how "human" is not in the DataFrame, but it is still in levels:

        >>> large_leg_num = leg_num[leg_num.Legs > 2]
        >>> large_leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 cat         4
                 dog         4

        >>> large_leg_num.index.levels
        FrozenList([[\'mammal\'], [\'cat\', \'dog\', \'goat\', \'human\']])
        '''
    def _set_levels(self, levels, *, level: Incomplete | None = None, copy: bool = False, validate: bool = True, verify_integrity: bool = False) -> None: ...
    def set_levels(self, levels, *, level: Incomplete | None = None, verify_integrity: bool = True) -> MultiIndex:
        '''
        Set new levels on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        levels : sequence or list of sequence
            New level(s) to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, "one"),
        ...         (1, "two"),
        ...         (2, "one"),
        ...         (2, "two"),
        ...         (3, "one"),
        ...         (3, "two")
        ...     ],
        ...     names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, \'one\'),
            (1, \'two\'),
            (2, \'one\'),
            (2, \'two\'),
            (3, \'one\'),
            (3, \'two\')],
           names=[\'foo\', \'bar\'])

        >>> idx.set_levels([[\'a\', \'b\', \'c\'], [1, 2]])
        MultiIndex([(\'a\', 1),
                    (\'a\', 2),
                    (\'b\', 1),
                    (\'b\', 2),
                    (\'c\', 1),
                    (\'c\', 2)],
                   names=[\'foo\', \'bar\'])
        >>> idx.set_levels([\'a\', \'b\', \'c\'], level=0)
        MultiIndex([(\'a\', \'one\'),
                    (\'a\', \'two\'),
                    (\'b\', \'one\'),
                    (\'b\', \'two\'),
                    (\'c\', \'one\'),
                    (\'c\', \'two\')],
                   names=[\'foo\', \'bar\'])
        >>> idx.set_levels([\'a\', \'b\'], level=\'bar\')
        MultiIndex([(1, \'a\'),
                    (1, \'b\'),
                    (2, \'a\'),
                    (2, \'b\'),
                    (3, \'a\'),
                    (3, \'b\')],
                   names=[\'foo\', \'bar\'])

        If any of the levels passed to ``set_levels()`` exceeds the
        existing length, all of the values from that argument will
        be stored in the MultiIndex levels, though the values will
        be truncated in the MultiIndex output.

        >>> idx.set_levels([[\'a\', \'b\', \'c\'], [1, 2, 3, 4]], level=[0, 1])
        MultiIndex([(\'a\', 1),
            (\'a\', 2),
            (\'b\', 1),
            (\'b\', 2),
            (\'c\', 1),
            (\'c\', 2)],
           names=[\'foo\', \'bar\'])
        >>> idx.set_levels([[\'a\', \'b\', \'c\'], [1, 2, 3, 4]], level=[0, 1]).levels
        FrozenList([[\'a\', \'b\', \'c\'], [1, 2, 3, 4]])
        '''
    @property
    def nlevels(self) -> int:
        """
        Integer number of levels in this MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.nlevels
        3
        """
    @property
    def levshape(self) -> Shape:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.levshape
        (1, 1, 1)
        """
    @property
    def codes(self) -> FrozenList: ...
    def _set_codes(self, codes, *, level: Incomplete | None = None, copy: bool = False, validate: bool = True, verify_integrity: bool = False) -> None: ...
    def set_codes(self, codes, *, level: Incomplete | None = None, verify_integrity: bool = True) -> MultiIndex:
        '''
        Set new codes on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        codes : sequence or list of sequence
            New codes to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, \'one\'),
            (1, \'two\'),
            (2, \'one\'),
            (2, \'two\')],
           names=[\'foo\', \'bar\'])

        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])
        MultiIndex([(2, \'one\'),
                    (1, \'one\'),
                    (2, \'two\'),
                    (1, \'two\')],
                   names=[\'foo\', \'bar\'])
        >>> idx.set_codes([1, 0, 1, 0], level=0)
        MultiIndex([(2, \'one\'),
                    (1, \'two\'),
                    (2, \'one\'),
                    (1, \'two\')],
                   names=[\'foo\', \'bar\'])
        >>> idx.set_codes([0, 0, 1, 1], level=\'bar\')
        MultiIndex([(1, \'one\'),
                    (1, \'one\'),
                    (2, \'two\'),
                    (2, \'two\')],
                   names=[\'foo\', \'bar\'])
        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])
        MultiIndex([(2, \'one\'),
                    (1, \'one\'),
                    (2, \'two\'),
                    (1, \'two\')],
                   names=[\'foo\', \'bar\'])
        '''
    def _engine(self): ...
    @property
    def _constructor(self) -> Callable[..., MultiIndex]: ...
    def _shallow_copy(self, values: np.ndarray, name=...) -> MultiIndex: ...
    def _view(self) -> MultiIndex: ...
    def copy(self, names: Incomplete | None = None, deep: bool = False, name: Incomplete | None = None) -> Self:
        """
        Make a copy of this object.

        Names, dtype, levels and codes can be passed and will be set on new copy.

        Parameters
        ----------
        names : sequence, optional
        deep : bool, default False
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.copy()
        MultiIndex([('a', 'b', 'c')],
                   )
        """
    def __array__(self, dtype: Incomplete | None = None, copy: Incomplete | None = None) -> np.ndarray:
        """the array interface, return my values"""
    def view(self, cls: Incomplete | None = None) -> Self:
        """this is defined as a copy with the same identity"""
    def __contains__(self, key: Any) -> bool: ...
    def dtype(self) -> np.dtype: ...
    def _is_memory_usage_qualified(self) -> bool:
        """return a boolean if we need a qualified .info display"""
    def memory_usage(self, deep: bool = False) -> int: ...
    def nbytes(self) -> int:
        """return the number of bytes in the underlying data"""
    def _nbytes(self, deep: bool = False) -> int:
        """
        return the number of bytes in the underlying data
        deeply introspect the level data if deep=True

        include the engine hashtable

        *this is in internal routine*

        """
    def _formatter_func(self, tup):
        """
        Formats each item in tup according to its level's formatter function.
        """
    def _get_values_for_csv(self, *, na_rep: str = 'nan', **kwargs) -> npt.NDArray[np.object_]: ...
    def format(self, name: bool | None = None, formatter: Callable | None = None, na_rep: str | None = None, names: bool = False, space: int = 2, sparsify: Incomplete | None = None, adjoin: bool = True) -> list: ...
    def _format_multi(self, *, include_names: bool, sparsify: bool | None | lib.NoDefault, formatter: Callable | None = None) -> list: ...
    def _get_names(self) -> FrozenList: ...
    def _set_names(self, names, *, level: Incomplete | None = None, validate: bool = True):
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None
        validate : bool, default True
            validate that the names match level lengths

        Raises
        ------
        TypeError if each name is not hashable.

        Notes
        -----
        sets names on levels. WARNING: mutates!

        Note that you generally want to set this *after* changing levels, so
        that it only acts on copies
        """
    names: Incomplete
    def inferred_type(self) -> str: ...
    def _get_level_number(self, level) -> int: ...
    def is_monotonic_increasing(self) -> bool:
        """
        Return a boolean if the values are equal or increasing.
        """
    def is_monotonic_decreasing(self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.
        """
    def _inferred_type_levels(self) -> list[str]:
        """return a list of the inferred types, one for each level"""
    def duplicated(self, keep: DropKeep = 'first') -> npt.NDArray[np.bool_]: ...
    _duplicated = duplicated
    def fillna(self, value: Incomplete | None = None, downcast: Incomplete | None = None) -> None:
        """
        fillna is not implemented for MultiIndex
        """
    def dropna(self, how: AnyAll = 'any') -> MultiIndex: ...
    def _get_level_values(self, level: int, unique: bool = False) -> Index:
        """
        Return vector of label values for requested level,
        equal to the length of the index

        **this is an internal method**

        Parameters
        ----------
        level : int
        unique : bool, default False
            if True, drop duplicated values

        Returns
        -------
        Index
        """
    def get_level_values(self, level) -> Index:
        """
        Return vector of label values for requested level.

        Length of returned vector is equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Notes
        -----
        If the level contains missing values, the result may be casted to
        ``float`` with missing values specified as ``NaN``. This is because
        the level is converted to a regular ``Index``.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))
        >>> mi.names = ['level_1', 'level_2']

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.get_level_values('level_2')
        Index(['d', 'e', 'f'], dtype='object', name='level_2')

        If a level contains missing values, the return type of the level
        may be cast to ``float``.

        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).dtypes
        level_0    int64
        level_1    int64
        dtype: object
        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).get_level_values(0)
        Index([1.0, nan, 2.0], dtype='float64')
        """
    def unique(self, level: Incomplete | None = None): ...
    def to_frame(self, index: bool = True, name=..., allow_duplicates: bool = False) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b'], ['c', 'd']])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=['x', 'y'])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
    def to_flat_index(self) -> Index:
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        See Also
        --------
        MultiIndex.from_tuples : Convert flat index back to MultiIndex.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [['foo', 'bar'], ['baz', 'qux']],
        ...     names=['a', 'b'])
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
    def _is_lexsorted(self) -> bool:
        """
        Return True if the codes are lexicographically sorted.

        Returns
        -------
        bool

        Examples
        --------
        In the below examples, the first level of the MultiIndex is sorted because
        a<b<c, so there is no need to look at the next level.

        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'],
        ...                            ['d', 'e', 'f']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'],
        ...                            ['d', 'f', 'e']])._is_lexsorted()
        True

        In case there is a tie, the lexicographical sorting looks
        at the next level of the MultiIndex.

        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'b', 'c']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'c', 'b']])._is_lexsorted()
        False
        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],
        ...                            ['aa', 'bb', 'aa', 'bb']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],
        ...                            ['bb', 'aa', 'aa', 'bb']])._is_lexsorted()
        False
        """
    def _lexsort_depth(self) -> int:
        """
        Compute and return the lexsort_depth, the number of levels of the
        MultiIndex that are sorted lexically

        Returns
        -------
        int
        """
    def _sort_levels_monotonic(self, raise_if_incomparable: bool = False) -> MultiIndex:
        """
        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )

        >>> mi.sort_values()
        MultiIndex([('a', 'aa'),
                    ('a', 'bb'),
                    ('b', 'aa'),
                    ('b', 'bb')],
                   )
        """
    def remove_unused_levels(self) -> MultiIndex:
        """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list('ab')])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """
    def __reduce__(self):
        """Necessary for making this object picklable"""
    def __getitem__(self, key): ...
    def _getitem_slice(self, slobj: slice) -> MultiIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
    def take(self, indices, axis: Axis = 0, allow_fill: bool = True, fill_value: Incomplete | None = None, **kwargs) -> MultiIndex: ...
    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index
            The combined index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b']])
        >>> mi
        MultiIndex([('a', 'b')],
                   )
        >>> mi.append(mi)
        MultiIndex([('a', 'b'), ('a', 'b')],
                   )
        """
    def argsort(self, *args, na_position: str = 'last', **kwargs) -> npt.NDArray[np.intp]: ...
    def repeat(self, repeats: int, axis: Incomplete | None = None) -> MultiIndex: ...
    def drop(self, codes, level: Index | np.ndarray | Iterable[Hashable] | None = None, errors: IgnoreRaise = 'raise') -> MultiIndex:
        '''
        Make a new :class:`pandas.MultiIndex` with the passed list of codes deleted.

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples when ``level`` is not specified.
        level : int or level name, default None
        errors : str, default \'raise\'

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), (\'green\', \'purple\')],
        ...                                  names=["number", "color"])
        >>> idx
        MultiIndex([(0,  \'green\'),
                    (0, \'purple\'),
                    (1,  \'green\'),
                    (1, \'purple\'),
                    (2,  \'green\'),
                    (2, \'purple\')],
                   names=[\'number\', \'color\'])
        >>> idx.drop([(1, \'green\'), (2, \'purple\')])
        MultiIndex([(0,  \'green\'),
                    (0, \'purple\'),
                    (1, \'purple\'),
                    (2,  \'green\')],
                   names=[\'number\', \'color\'])

        We can also drop from a specific level.

        >>> idx.drop(\'green\', level=\'color\')
        MultiIndex([(0, \'purple\'),
                    (1, \'purple\'),
                    (2, \'purple\')],
                   names=[\'number\', \'color\'])

        >>> idx.drop([1, 2], level=0)
        MultiIndex([(0,  \'green\'),
                    (0, \'purple\')],
                   names=[\'number\', \'color\'])
        '''
    def _drop_from_level(self, codes, level, errors: IgnoreRaise = 'raise') -> MultiIndex: ...
    def swaplevel(self, i: int = -2, j: int = -1) -> MultiIndex:
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        DataFrame.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
                    ('aa', 'a'),
                    ('bb', 'b'),
                    ('aa', 'b')],
                   )
        """
    def reorder_levels(self, order) -> MultiIndex:
        """
        Rearrange levels using input order. May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])
        >>> mi
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.reorder_levels(order=[1, 0])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])

        >>> mi.reorder_levels(order=['y', 'x'])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])
        """
    def _reorder_ilevels(self, order) -> MultiIndex: ...
    def _recode_for_new_levels(self, new_levels, copy: bool = True) -> Generator[np.ndarray, None, None]: ...
    def _get_codes_for_sorting(self) -> list[Categorical]:
        """
        we are categorizing our codes by using the
        available categories (all, not just observed)
        excluding any missing ones (-1); this is in preparation
        for sorting, where we need to disambiguate that -1 is not
        a valid valid
        """
    def sortlevel(self, level: IndexLabel = 0, ascending: bool | list[bool] = True, sort_remaining: bool = True, na_position: str = 'first') -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        """
        Sort MultiIndex at the requested level.

        The result will respect the original ordering of the associated
        factor at that level.

        Parameters
        ----------
        level : list-like, int or str, default 0
            If a string is given, must be a name of the level.
            If list-like must be names or ints of levels.
        ascending : bool, default True
            False to sort in descending order.
            Can also be a list to specify a directed ordering.
        sort_remaining : sort by the remaining levels after level
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        Returns
        -------
        sorted_index : pd.MultiIndex
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[0, 0], [2, 1]])
        >>> mi
        MultiIndex([(0, 2),
                    (0, 1)],
                   )

        >>> mi.sortlevel()
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(sort_remaining=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))

        >>> mi.sortlevel(1)
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(1, ascending=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))
        """
    def _wrap_reindex_result(self, target, indexer, preserve_names: bool): ...
    def _maybe_preserve_names(self, target: Index, preserve_names: bool) -> Index: ...
    def _check_indexing_error(self, key) -> None: ...
    def _should_fallback_to_positional(self) -> bool:
        """
        Should integer key(s) be treated as positional?
        """
    def _get_indexer_strict(self, key, axis_name: str) -> tuple[Index, npt.NDArray[np.intp]]: ...
    def _raise_if_missing(self, key, indexer, axis_name: str) -> None: ...
    def _get_indexer_level_0(self, target) -> npt.NDArray[np.intp]:
        """
        Optimized equivalent to `self.get_level_values(0).get_indexer_for(target)`.
        """
    def get_slice_bound(self, label: Hashable | Sequence[Hashable], side: Literal['left', 'right']) -> int:
        '''
        For an ordered MultiIndex, compute slice bound
        that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if `side==\'right\') position
        of given label.

        Parameters
        ----------
        label : object or tuple of objects
        side : {\'left\', \'right\'}

        Returns
        -------
        int
            Index of label.

        Notes
        -----
        This method only works if level 0 index of the MultiIndex is lexsorted.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list(\'abbc\'), list(\'gefd\')])

        Get the locations from the leftmost \'b\' in the first level
        until the end of the multiindex:

        >>> mi.get_slice_bound(\'b\', side="left")
        1

        Like above, but if you get the locations from the rightmost
        \'b\' in the first level and \'f\' in the second level:

        >>> mi.get_slice_bound((\'b\',\'f\'), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        '''
    def slice_locs(self, start: Incomplete | None = None, end: Incomplete | None = None, step: Incomplete | None = None) -> tuple[int, int]:
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbd'), list('deff')],
        ...                                names=['A', 'B'])

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start='b')
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start='b', end=('b', 'f'))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
    def _partial_tup_index(self, tup: tuple, side: Literal['left', 'right'] = 'left'): ...
    def _get_loc_single_level_index(self, level_index: Index, key: Hashable) -> int:
        """
        If key is NA value, location of index unify as -1.

        Parameters
        ----------
        level_index: Index
        key : label

        Returns
        -------
        loc : int
            If key is NA value, loc is -1
            Else, location of key in index.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        """
    def get_loc(self, key):
        """
        Get location for a label or a tuple of labels.

        The location is returned as an integer/slice or boolean
        mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)

        Returns
        -------
        int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Notes
        -----
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_loc('b')
        slice(1, 3, None)

        >>> mi.get_loc(('b', 'e'))
        1
        """
    def get_loc_level(self, key, level: IndexLabel = 0, drop_level: bool = True):
        """
        Get location and sliced index for requested label(s)/level(s).

        Parameters
        ----------
        key : label or sequence of labels
        level : int/level name or list thereof, optional
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        tuple
            A 2-tuple where the elements :

            Element 0: int, slice object or boolean array.

            Element 1: The resulting sliced multiindex/index. If the key
            contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')],
        ...                                names=['A', 'B'])

        >>> mi.get_loc_level('b')
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level('e', level='B')
        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(['b', 'e'])
        (1, None)
        """
    def _get_loc_level(self, key, level: int | list[int] = 0):
        """
        get_loc_level but with `level` known to be positional, not name-based.
        """
    def _get_level_indexer(self, key, level: int = 0, indexer: npt.NDArray[np.bool_] | None = None): ...
    def get_locs(self, seq) -> npt.NDArray[np.intp]:
        """
        Get location for a sequence of labels.

        Parameters
        ----------
        seq : label, slice, list, mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        numpy.ndarray
            NumPy array of integers suitable for passing to iloc.

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_locs('b')  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ['e', 'f']])  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice('e', 'f')])  # doctest: +SKIP
        array([2], dtype=int64)
        """
    def _reorder_indexer(self, seq: tuple[Scalar | Iterable | AnyArrayLike, ...], indexer: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """
        Reorder an indexer of a MultiIndex (self) so that the labels are in the
        same order as given in seq

        Parameters
        ----------
        seq : label/slice/list/mask or a sequence of such
        indexer: a position indexer of self

        Returns
        -------
        indexer : a sorted position indexer of self ordered as seq
        """
    def truncate(self, before: Incomplete | None = None, after: Incomplete | None = None) -> MultiIndex:
        """
        Slice index between two labels / tuples, return new MultiIndex.

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start.
        after : label or tuple, can be partial. Default None
            None defaults to end.

        Returns
        -------
        MultiIndex
            The truncated MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b', 'c'], ['x', 'y', 'z']])
        >>> mi
        MultiIndex([('a', 'x'), ('b', 'y'), ('c', 'z')],
                   )
        >>> mi.truncate(before='a', after='b')
        MultiIndex([('a', 'x'), ('b', 'y')],
                   )
        """
    def equals(self, other: object) -> bool:
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
    def equal_levels(self, other: MultiIndex) -> bool:
        """
        Return True if the levels of both MultiIndex objects are the same

        """
    def _union(self, other, sort) -> MultiIndex: ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool: ...
    def _get_reconciled_name_object(self, other) -> MultiIndex:
        """
        If the result of a set operation will be self,
        return self, unless the names change, in which
        case make a shallow copy of self.
        """
    def _maybe_match_names(self, other):
        """
        Try to find common names to attach to the result of an operation between
        a and b. Return a consensus list of names if they match at least partly
        or list of None if they have completely different names.
        """
    def _wrap_intersection_result(self, other, result) -> MultiIndex: ...
    def _wrap_difference_result(self, other, result: MultiIndex) -> MultiIndex: ...
    def _convert_can_do_setop(self, other): ...
    def astype(self, dtype, copy: bool = True): ...
    def _validate_fill_value(self, item): ...
    def putmask(self, mask, value: MultiIndex) -> MultiIndex:
        """
        Return a new MultiIndex of the values set with the mask.

        Parameters
        ----------
        mask : array like
        value : MultiIndex
            Must either be the same length as self or length one

        Returns
        -------
        MultiIndex
        """
    def insert(self, loc: int, item) -> MultiIndex:
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
        item : tuple
            Must be same length as number of levels in the MultiIndex

        Returns
        -------
        new_index : Index
        """
    def delete(self, loc) -> MultiIndex:
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
    def isin(self, values, level: Incomplete | None = None) -> npt.NDArray[np.bool_]: ...
    rename: Incomplete
    __add__: Incomplete
    __radd__: Incomplete
    __iadd__: Incomplete
    __sub__: Incomplete
    __rsub__: Incomplete
    __isub__: Incomplete
    __pow__: Incomplete
    __rpow__: Incomplete
    __mul__: Incomplete
    __rmul__: Incomplete
    __floordiv__: Incomplete
    __rfloordiv__: Incomplete
    __truediv__: Incomplete
    __rtruediv__: Incomplete
    __mod__: Incomplete
    __rmod__: Incomplete
    __divmod__: Incomplete
    __rdivmod__: Incomplete
    __neg__: Incomplete
    __pos__: Incomplete
    __abs__: Incomplete
    __invert__: Incomplete

def _lexsort_depth(codes: list[np.ndarray], nlevels: int) -> int:
    """Count depth (up to a maximum of `nlevels`) with which codes are lexsorted."""
def sparsify_labels(label_list, start: int = 0, sentinel: object = ''): ...
def _get_na_rep(dtype: DtypeObj) -> str: ...
def maybe_droplevels(index: Index, key) -> Index:
    """
    Attempt to drop level or levels from the given index.

    Parameters
    ----------
    index: Index
    key : scalar or tuple

    Returns
    -------
    Index
    """
def _coerce_indexer_frozen(array_like, categories, copy: bool = False) -> np.ndarray:
    """
    Coerce the array-like indexer to the smallest integer dtype that can encode all
    of the given categories.

    Parameters
    ----------
    array_like : array-like
    categories : array-like
    copy : bool

    Returns
    -------
    np.ndarray
        Non-writeable.
    """
def _require_listlike(level, arr, arrname: str):
    """
    Ensure that level is either None or listlike, and arr is list-of-listlike.
    """

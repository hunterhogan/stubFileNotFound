import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable
from pandas import DataFrame as DataFrame, Series as Series
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs.indexing import NDFrameIndexerBase as NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim as item_from_zerodim
from pandas._typing import Axis as Axis, AxisInt as AxisInt, Self as Self, npt as npt
from pandas.compat import PYPY as PYPY
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.cast import can_hold_element as can_hold_element, maybe_promote as maybe_promote
from pandas.core.dtypes.common import is_array_like as is_array_like, is_bool_dtype as is_bool_dtype, is_hashable as is_hashable, is_integer as is_integer, is_iterator as is_iterator, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_sequence as is_sequence
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import construct_1d_array_from_inferred_fill_value as construct_1d_array_from_inferred_fill_value, infer_fill_value as infer_fill_value, is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.indexers import check_array_indexer as check_array_indexer, is_list_like_indexer as is_list_like_indexer, is_scalar_indexer as is_scalar_indexer, length_of_indexer as length_of_indexer
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex
from pandas.errors import AbstractMethodError as AbstractMethodError, ChainedAssignmentError as ChainedAssignmentError, IndexingError as IndexingError, InvalidIndexError as InvalidIndexError, LossySetitemError as LossySetitemError, _chained_assignment_msg as _chained_assignment_msg, _chained_assignment_warning_msg as _chained_assignment_warning_msg, _check_cacher as _check_cacher
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, TypeVar

T = TypeVar('T')
_NS: Incomplete
_one_ellipsis_message: str

class _IndexSlice:
    """
    Create an object to more easily perform multi-index slicing.

    See Also
    --------
    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.

    Notes
    -----
    See :ref:`Defined Levels <advanced.shown_levels>`
    for further info on slicing a MultiIndex.

    Examples
    --------
    >>> midx = pd.MultiIndex.from_product([['A0','A1'], ['B0','B1','B2','B3']])
    >>> columns = ['foo', 'bar']
    >>> dfmi = pd.DataFrame(np.arange(16).reshape((len(midx), len(columns))),
    ...                     index=midx, columns=columns)

    Using the default slice command:

    >>> dfmi.loc[(slice(None), slice('B0', 'B1')), :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11

    Using the IndexSlice class for a more intuitive command:

    >>> idx = pd.IndexSlice
    >>> dfmi.loc[idx[:, 'B0':'B1'], :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11
    """
    def __getitem__(self, arg): ...

IndexSlice: Incomplete

class IndexingMixin:
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """
    @property
    def iloc(self) -> _iLocIndexer:
        """
        Purely integer-location based indexing for selection by position.

        .. deprecated:: 2.2.0

           Returning a tuple from a callable is deprecated.

        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean
        array.

        Allowed inputs are:

        - An integer, e.g. ``5``.
        - A list or array of integers, e.g. ``[4, 3, 0]``.
        - A slice object with ints, e.g. ``1:7``.
        - A boolean array.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above).
          This is useful in method chains, when you don't have a reference to the
          calling object, but would like to base your selection on
          some value.
        - A tuple of row and column indexes. The tuple elements consist of one of the
          above inputs, e.g. ``(0, 1)``.

        ``.iloc`` will raise ``IndexError`` if a requested indexer is
        out-of-bounds, except *slice* indexers which allow out-of-bounds
        indexing (this conforms with python/numpy *slice* semantics).

        See more at :ref:`Selection by Position <indexing.integer>`.

        See Also
        --------
        DataFrame.iat : Fast integer location scalar accessor.
        DataFrame.loc : Purely label-location based indexer for selection by label.
        Series.iloc : Purely integer-location based indexing for
                       selection by position.

        Examples
        --------
        >>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
        ...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
        ...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]
        >>> df = pd.DataFrame(mydict)
        >>> df
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        **Indexing just the rows**

        With a scalar integer.

        >>> type(df.iloc[0])
        <class 'pandas.core.series.Series'>
        >>> df.iloc[0]
        a    1
        b    2
        c    3
        d    4
        Name: 0, dtype: int64

        With a list of integers.

        >>> df.iloc[[0]]
           a  b  c  d
        0  1  2  3  4
        >>> type(df.iloc[[0]])
        <class 'pandas.core.frame.DataFrame'>

        >>> df.iloc[[0, 1]]
             a    b    c    d
        0    1    2    3    4
        1  100  200  300  400

        With a `slice` object.

        >>> df.iloc[:3]
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        With a boolean mask the same length as the index.

        >>> df.iloc[[True, False, True]]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        With a callable, useful in method chains. The `x` passed
        to the ``lambda`` is the DataFrame being sliced. This selects
        the rows whose index label even.

        >>> df.iloc[lambda x: x.index % 2 == 0]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        **Indexing both axes**

        You can mix the indexer types for the index and columns. Use ``:`` to
        select the entire axis.

        With scalar integers.

        >>> df.iloc[0, 1]
        2

        With lists of integers.

        >>> df.iloc[[0, 2], [1, 3]]
              b     d
        0     2     4
        2  2000  4000

        With `slice` objects.

        >>> df.iloc[1:3, 0:3]
              a     b     c
        1   100   200   300
        2  1000  2000  3000

        With a boolean array whose length matches the columns.

        >>> df.iloc[:, [True, False, True, False]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000

        With a callable function that expects the Series or DataFrame.

        >>> df.iloc[:, lambda df: [0, 2]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000
        """
    @property
    def loc(self) -> _LocIndexer:
        '''
        Access a group of rows and columns by label(s) or a boolean array.

        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.

        Allowed inputs are:

        - A single label, e.g. ``5`` or ``\'a\'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``[\'a\', \'b\', \'c\']``.
        - A slice object with labels, e.g. ``\'a\':\'f\'``.

          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included

        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - An alignable boolean Series. The index of the key will be aligned before
          masking.
        - An alignable Index. The Index of the returned selection will be the input.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)

        See more at :ref:`Selection by Label <indexing.label>`.

        Raises
        ------
        KeyError
            If any items are not found.
        IndexingError
            If an indexed key is passed and its index is unalignable to the frame index.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
                       Series/DataFrame.
        Series.loc : Access group of values using labels.

        Examples
        --------
        **Getting values**

        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=[\'cobra\', \'viper\', \'sidewinder\'],
        ...                   columns=[\'max_speed\', \'shield\'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8

        Single label. Note this returns the row as a Series.

        >>> df.loc[\'viper\']
        max_speed    4
        shield       5
        Name: viper, dtype: int64

        List of labels. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[[\'viper\', \'sidewinder\']]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

        Single label for row and column

        >>> df.loc[\'cobra\', \'shield\']
        2

        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.

        >>> df.loc[\'cobra\':\'viper\', \'max_speed\']
        cobra    1
        viper    4
        Name: max_speed, dtype: int64

        Boolean list with the same length as the row axis

        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8

        Alignable boolean Series:

        >>> df.loc[pd.Series([False, True, False],
        ...                  index=[\'viper\', \'sidewinder\', \'cobra\'])]
                             max_speed  shield
        sidewinder          7       8

        Index (same behavior as ``df.reindex``)

        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]
               max_speed  shield
        foo
        cobra          1       2
        viper          4       5

        Conditional that returns a boolean Series

        >>> df.loc[df[\'shield\'] > 6]
                    max_speed  shield
        sidewinder          7       8

        Conditional that returns a boolean Series with column labels specified

        >>> df.loc[df[\'shield\'] > 6, [\'max_speed\']]
                    max_speed
        sidewinder          7

        Multiple conditional using ``&`` that returns a boolean Series

        >>> df.loc[(df[\'max_speed\'] > 1) & (df[\'shield\'] < 8)]
                    max_speed  shield
        viper          4       5

        Multiple conditional using ``|`` that returns a boolean Series

        >>> df.loc[(df[\'max_speed\'] > 4) | (df[\'shield\'] < 5)]
                    max_speed  shield
        cobra               1       2
        sidewinder          7       8

        Please ensure that each condition is wrapped in parentheses ``()``.
        See the :ref:`user guide<indexing.boolean>`
        for more details and explanations of Boolean indexing.

        .. note::
            If you find yourself using 3 or more conditionals in ``.loc[]``,
            consider using :ref:`advanced indexing<advanced.advanced_hierarchical>`.

            See below for using ``.loc[]`` on MultiIndex DataFrames.

        Callable that returns a boolean Series

        >>> df.loc[lambda df: df[\'shield\'] == 8]
                    max_speed  shield
        sidewinder          7       8

        **Setting values**

        Set value for all items matching the list of labels

        >>> df.loc[[\'viper\', \'sidewinder\'], [\'shield\']] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50

        Set value for an entire row

        >>> df.loc[\'cobra\'] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50

        Set value for an entire column

        >>> df.loc[:, \'max_speed\'] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50

        Set value for rows matching callable condition

        >>> df.loc[df[\'shield\'] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0

        Add value matching location

        >>> df.loc["viper", "shield"] += 5
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       5
        sidewinder          0       0

        Setting using a ``Series`` or a ``DataFrame`` sets the values matching the
        index labels, not the index positions.

        >>> shuffled_df = df.loc[["viper", "cobra", "sidewinder"]]
        >>> df.loc[:] += shuffled_df
        >>> df
                    max_speed  shield
        cobra              60      20
        viper               0      10
        sidewinder          0       0

        **Getting values on a DataFrame with an index that has integer labels**

        Another example using integers for the index

        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=[7, 8, 9], columns=[\'max_speed\', \'shield\'])
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.

        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        **Getting values with a MultiIndex**

        A number of examples using a DataFrame with a MultiIndex

        >>> tuples = [
        ...     (\'cobra\', \'mark i\'), (\'cobra\', \'mark ii\'),
        ...     (\'sidewinder\', \'mark i\'), (\'sidewinder\', \'mark ii\'),
        ...     (\'viper\', \'mark ii\'), (\'viper\', \'mark iii\')
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20],
        ...           [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=[\'max_speed\', \'shield\'], index=index)
        >>> df
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Single label. Note this returns a DataFrame with a single index.

        >>> df.loc[\'cobra\']
                 max_speed  shield
        mark i          12       2
        mark ii          0       4

        Single index tuple. Note this returns a Series.

        >>> df.loc[(\'cobra\', \'mark ii\')]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64

        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.

        >>> df.loc[\'cobra\', \'mark i\']
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64

        Single tuple. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[[(\'cobra\', \'mark ii\')]]
                       max_speed  shield
        cobra mark ii          0       4

        Single tuple for the index with a single label for the column

        >>> df.loc[(\'cobra\', \'mark i\'), \'shield\']
        2

        Slice from index tuple to single label

        >>> df.loc[(\'cobra\', \'mark i\'):\'viper\']
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Slice from index tuple to index tuple

        >>> df.loc[(\'cobra\', \'mark i\'):(\'viper\', \'mark ii\')]
                            max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1

        Please see the :ref:`user guide<advanced.advanced_hierarchical>`
        for more details and explanations of advanced indexing.
        '''
    @property
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.

        Similar to ``loc``, in that both provide label-based lookups. Use
        ``at`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        KeyError
            If getting a value and 'label' does not exist in a DataFrame or Series.

        ValueError
            If row/column label pair is not a tuple or if any label
            from the pair is not a scalar for DataFrame.
            If label is list-like (*excluding* NamedTuple) for Series.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column pair by label.
        DataFrame.iat : Access a single value for a row/column pair by integer
            position.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer
            position(s).
        Series.at : Access a single value by label.
        Series.iat : Access a single value by integer position.
        Series.loc : Access a group of rows by label(s).
        Series.iloc : Access a group of rows by integer position(s).

        Notes
        -----
        See :ref:`Fast scalar value getting and setting <indexing.basics.get_value>`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...                   index=[4, 5, 6], columns=['A', 'B', 'C'])
        >>> df
            A   B   C
        4   0   2   3
        5   0   4   1
        6  10  20  30

        Get value at specified row/column pair

        >>> df.at[4, 'B']
        2

        Set value at specified row/column pair

        >>> df.at[4, 'B'] = 10
        >>> df.at[4, 'B']
        10

        Get value within a Series

        >>> df.loc[5].at['B']
        4
        """
    @property
    def iat(self) -> _iAtIndexer:
        """
        Access a single value for a row/column pair by integer position.

        Similar to ``iloc``, in that both provide integer-based lookups. Use
        ``iat`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        IndexError
            When integer position is out of bounds.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer position(s).

        Examples
        --------
        >>> df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...                   columns=['A', 'B', 'C'])
        >>> df
            A   B   C
        0   0   2   3
        1   0   4   1
        2  10  20  30

        Get value at specified row/column pair

        >>> df.iat[1, 2]
        1

        Set value at specified row/column pair

        >>> df.iat[1, 2] = 10
        >>> df.iat[1, 2]
        10

        Get value within a series

        >>> df.loc[0].iat[1]
        2
        """

class _LocationIndexer(NDFrameIndexerBase):
    _valid_types: str
    axis: AxisInt | None
    _takeable: bool
    def __call__(self, axis: Axis | None = None) -> Self: ...
    def _get_setitem_indexer(self, key):
        """
        Convert a potentially-label-based key into a positional indexer.
        """
    def _maybe_mask_setitem_value(self, indexer, value):
        """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.

        Note this is only for loc, not iloc.
        """
    def _ensure_listlike_indexer(self, key, axis: Incomplete | None = None, value: Incomplete | None = None) -> None:
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
    def __setitem__(self, key, value) -> None: ...
    def _validate_key(self, key, axis: AxisInt):
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
    def _expand_ellipsis(self, tup: tuple) -> tuple:
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
    def _validate_tuple_indexer(self, key: tuple) -> tuple:
        """
        Check the key for valid keys across my indexer.
        """
    def _is_nested_tuple_indexer(self, tup: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
    def _convert_tuple(self, key: tuple) -> tuple: ...
    def _validate_key_length(self, key: tuple) -> tuple: ...
    def _getitem_tuple_same_dim(self, tup: tuple):
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.

        This is only called after a failed call to _getitem_lowerdim.
        """
    def _getitem_lowerdim(self, tup: tuple): ...
    def _getitem_nested_tuple(self, tup: tuple): ...
    def _convert_to_indexer(self, key, axis: AxisInt): ...
    def _check_deprecated_callable_usage(self, key: Any, maybe_callable: T) -> T: ...
    def __getitem__(self, key): ...
    def _is_scalar_access(self, key: tuple): ...
    def _getitem_tuple(self, tup: tuple): ...
    def _getitem_axis(self, key, axis: AxisInt): ...
    def _has_valid_setitem_indexer(self, indexer) -> bool: ...
    def _getbool_axis(self, key, axis: AxisInt): ...

class _LocIndexer(_LocationIndexer):
    _takeable: bool
    _valid_types: str
    def _validate_key(self, key, axis: Axis): ...
    def _has_valid_setitem_indexer(self, indexer) -> bool: ...
    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
    def _multi_take_opportunity(self, tup: tuple) -> bool:
        """
        Check whether there is the possibility to use ``_multi_take``.

        Currently the limit is that all axes being indexed, must be indexed with
        list-likes.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        bool
            Whether the current indexing,
            can be passed through `_multi_take`.
        """
    def _multi_take(self, tup: tuple):
        """
        Create the indexers for the passed tuple of keys, and
        executes the take operation. This allows the take operation to be
        executed all at once, rather than once for each dimension.
        Improving efficiency.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        values: same type as the object being indexed
        """
    def _getitem_iterable(self, key, axis: AxisInt):
        """
        Index current object with an iterable collection of keys.

        Parameters
        ----------
        key : iterable
            Targeted labels.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If no key was found. Will change in the future to raise if not all
            keys were found.

        Returns
        -------
        scalar, DataFrame, or Series: indexed value(s).
        """
    def _getitem_tuple(self, tup: tuple): ...
    def _get_label(self, label, axis: AxisInt): ...
    def _handle_lowerdim_multi_index_axis0(self, tup: tuple): ...
    def _getitem_axis(self, key, axis: AxisInt): ...
    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        """
        This is pretty simple as we just have to deal with labels.
        """
    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Convert indexing key into something we can use to do actual fancy
        indexing on a ndarray.

        Examples
        ix[:5] -> slice(0, 5)
        ix[[1,2,3]] -> [1,2,3]
        ix[['foo', 'bar', 'baz']] -> [i, j, k] (indices of foo, bar, baz)

        Going by Zen of Python?
        'In the face of ambiguity, refuse the temptation to guess.'
        raise AmbiguousIndexError with integer labels?
        - No, prefer label-based indexing
        """
    def _get_listlike_indexer(self, key, axis: AxisInt):
        """
        Transform a list-like of keys into a new index and an indexer.

        Parameters
        ----------
        key : list-like
            Targeted labels.
        axis:  int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.

        Returns
        -------
        keyarr: Index
            New index (coinciding with 'key' if the axis is unique).
        values : array-like
            Indexer for the return object, -1 denotes keys not found.
        """

class _iLocIndexer(_LocationIndexer):
    _valid_types: str
    _takeable: bool
    def _validate_key(self, key, axis: AxisInt): ...
    def _has_valid_setitem_indexer(self, indexer) -> bool:
        """
        Validate that a positional indexer cannot enlarge its target
        will raise if needed, does not modify the indexer externally.

        Returns
        -------
        bool
        """
    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
    def _validate_integer(self, key: int | np.integer, axis: AxisInt) -> None:
        """
        Check that 'key' is a valid position in the desired axis.

        Parameters
        ----------
        key : int
            Requested position.
        axis : int
            Desired axis.

        Raises
        ------
        IndexError
            If 'key' is not a valid position in axis 'axis'.
        """
    def _getitem_tuple(self, tup: tuple): ...
    def _get_list_axis(self, key, axis: AxisInt):
        """
        Return Series values by list or array of integers.

        Parameters
        ----------
        key : list-like positional indexer
        axis : int

        Returns
        -------
        Series object

        Notes
        -----
        `axis` can only be zero.
        """
    def _getitem_axis(self, key, axis: AxisInt): ...
    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt): ...
    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Much simpler as we only have to deal with our valid types.
        """
    def _get_setitem_indexer(self, key): ...
    def _setitem_with_indexer(self, indexer, value, name: str = 'iloc'):
        """
        _setitem_with_indexer is for setting values on a Series/DataFrame
        using positional indexers.

        If the relevant keys are not present, the Series/DataFrame may be
        expanded.

        This method is currently broken when dealing with non-unique Indexes,
        since it goes from positional indexers back to labels when calling
        BlockManager methods, see GH#12991, GH#22046, GH#15686.
        """
    def _setitem_with_indexer_split_path(self, indexer, value, name: str):
        """
        Setitem column-wise.
        """
    def _setitem_with_indexer_2d_value(self, indexer, value) -> None: ...
    def _setitem_with_indexer_frame_value(self, indexer, value: DataFrame, name: str): ...
    def _setitem_single_column(self, loc: int, value, plane_indexer) -> None:
        """

        Parameters
        ----------
        loc : int
            Indexer for column position
        plane_indexer : int, slice, listlike[int]
            The indexer we use for setitem along axis=0.
        """
    def _setitem_single_block(self, indexer, value, name: str) -> None:
        """
        _setitem_with_indexer for the case when we have a single Block.
        """
    def _setitem_with_indexer_missing(self, indexer, value):
        """
        Insert new row(s) or column(s) into the Series or DataFrame.
        """
    def _ensure_iterable_column_indexer(self, column_indexer):
        """
        Ensure that our column indexer is something that can be iterated over.
        """
    def _align_series(self, indexer, ser: Series, multiindex_indexer: bool = False, using_cow: bool = False):
        """
        Parameters
        ----------
        indexer : tuple, slice, scalar
            Indexer used to get the locations that will be set to `ser`.
        ser : pd.Series
            Values to assign to the locations specified by `indexer`.
        multiindex_indexer : bool, optional
            Defaults to False. Should be set to True if `indexer` was from
            a `pd.MultiIndex`, to avoid unnecessary broadcasting.

        Returns
        -------
        `np.array` of `ser` broadcast to the appropriate shape for assignment
        to the locations selected by `indexer`
        """
    def _align_frame(self, indexer, df: DataFrame) -> DataFrame: ...

class _ScalarAccessIndexer(NDFrameIndexerBase):
    """
    Access scalars quickly.
    """
    _takeable: bool
    def _convert_key(self, key) -> None: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...

class _AtIndexer(_ScalarAccessIndexer):
    _takeable: bool
    def _convert_key(self, key):
        """
        Require they keys to be the same type as the index. (so we don't
        fallback)
        """
    @property
    def _axes_are_unique(self) -> bool: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...

class _iAtIndexer(_ScalarAccessIndexer):
    _takeable: bool
    def _convert_key(self, key):
        """
        Require integer args. (and convert to label arguments)
        """

def _tuplify(ndim: int, loc: Hashable) -> tuple[Hashable | slice, ...]:
    """
    Given an indexer for the first dimension, create an equivalent tuple
    for indexing over all dimensions.

    Parameters
    ----------
    ndim : int
    loc : object

    Returns
    -------
    tuple
    """
def _tupleize_axis_indexer(ndim: int, axis: AxisInt, key) -> tuple:
    """
    If we have an axis, adapt the given key to be axis-independent.
    """
def check_bool_indexer(index: Index, key) -> np.ndarray:
    """
    Check if key is a valid boolean indexer for an object with such index and
    perform reindexing or conversion if needed.

    This function assumes that is_bool_indexer(key) == True.

    Parameters
    ----------
    index : Index
        Index of the object on which the indexing is done.
    key : list-like
        Boolean indexer to check.

    Returns
    -------
    np.array
        Resulting key.

    Raises
    ------
    IndexError
        If the key does not have the same length as index.
    IndexingError
        If the index of the key is unalignable to index.
    """
def convert_missing_indexer(indexer):
    """
    Reverse convert a missing indexer, which is a dict
    return the scalar indexer and a boolean indicating if we converted
    """
def convert_from_missing_indexer_tuple(indexer, axes):
    """
    Create a filtered indexer that doesn't have any missing indexers.
    """
def maybe_convert_ix(*args):
    """
    We likely want to take the cross-product.
    """
def is_nested_tuple(tup, labels) -> bool:
    """
    Returns
    -------
    bool
    """
def is_label_like(key) -> bool:
    """
    Returns
    -------
    bool
    """
def need_slice(obj: slice) -> bool:
    """
    Returns
    -------
    bool
    """
def check_dict_or_set_indexers(key) -> None:
    """
    Check if the indexer is or contains a dict or set, which is no longer allowed.
    """

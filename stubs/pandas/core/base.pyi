import np
import npt
import pandas._libs.lib as lib
import pandas.compat.numpy.function as nv
import pandas.core.accessor
import pandas.core.algorithms as algorithms
import pandas.core.arraylike
import pandas.core.nanops as nanops
import pandas.core.ops as ops
import typing
from _typeshed import Incomplete
from builtins import AxisInt
from pandas._config import using_copy_on_write as using_copy_on_write
from pandas._libs.lib import is_scalar as is_scalar
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._typing import NDFrameT as NDFrameT
from pandas.core.accessor import DirNamesMixin as DirNamesMixin
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import can_hold_element as can_hold_element
from pandas.core.dtypes.common import is_object_dtype as is_object_dtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna, remove_na_arraylike as remove_na_arraylike
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar, Literal

TYPE_CHECKING: bool
Self: None
npt: None
PYPY: bool
_shared_docs: dict
_indexops_doc_kwargs: dict

class PandasObject(pandas.core.accessor.DirNamesMixin):
    def _reset_cache(self, key: str | None) -> None:
        """
        Reset cached properties. If ``key`` is passed, only clears that key.
        """
    def __sizeof__(self) -> int:
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
    @property
    def _constructor(self): ...

class NoNewAttributesMixin:
    def _freeze(self) -> None:
        """
        Prevents setting additional attributes.
        """
    def __setattr__(self, key: str, value) -> None: ...

class SelectionMixin(typing.Generic):
    _selection: ClassVar[None] = ...
    _internal_names: ClassVar[list] = ...
    _internal_names_set: ClassVar[set] = ...
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    _selected_obj: Incomplete
    ndim: Incomplete
    _obj_with_exclusions: Incomplete
    def __getitem__(self, key): ...
    def _gotitem(self, key, ndim: int, subset):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
    def _infer_selection(self, key, subset: Series | DataFrame):
        """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
    def aggregate(self, func, *args, **kwargs): ...
    def agg(self, func, *args, **kwargs): ...
    @property
    def _selection_list(self): ...

class IndexOpsMixin(pandas.core.arraylike.OpsMixin):
    __array_priority__: ClassVar[int] = ...
    _hidden_attrs: ClassVar[frozenset] = ...
    hasnans: Incomplete
    def transpose(self, *args, **kwargs) -> Self:
        """
        Return the transpose, which is by definition self.

        Returns
        -------
        %(klass)s
        """
    def __len__(self) -> int: ...
    def item(self):
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series or Index.

        Raises
        ------
        ValueError
            If the data is not length = 1.

        Examples
        --------
        >>> s = pd.Series([1])
        >>> s.item()
        1

        For an index:

        >>> s = pd.Series([1], index=['a'])
        >>> s.index.item()
        'a'
        """
    def to_numpy(self, dtype: npt.DTypeLike | None, copy: bool = ..., na_value: object = ..., **kwargs) -> np.ndarray:
        '''
        A NumPy ndarray representing the values in this Series or Index.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.
        **kwargs
            Additional keywords passed through to the ``to_numpy`` method
            of the underlying array (for extension arrays).

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Series.array : Get the actual data stored within.
        Index.array : Get the actual data stored within.
        DataFrame.to_numpy : Similar method for DataFrame.

        Notes
        -----
        The returned array will be the same up to equality (values equal
        in `self` will be equal in the returned array; likewise for values
        that are not equal). When `self` contains an ExtensionArray, the
        dtype may be different. For example, for a category-dtype Series,
        ``to_numpy()`` will return a NumPy array and the categorical dtype
        will be lost.

        For NumPy dtypes, this will be a reference to the actual data stored
        in this Series or Index (assuming ``copy=False``). Modifying the result
        in place will modify the data stored in the Series or Index (not that
        we recommend doing that).

        For extension types, ``to_numpy()`` *may* require copying data and
        coercing the result to a NumPy type (possibly object), which may be
        expensive. When you need a no-copy reference to the underlying data,
        :attr:`Series.array` should be used instead.

        This table lays out the different dtypes and default return types of
        ``to_numpy()`` for various dtypes within pandas.

        ================== ================================
        dtype              array type
        ================== ================================
        category[T]        ndarray[T] (same dtype as input)
        period             ndarray[object] (Periods)
        interval           ndarray[object] (Intervals)
        IntegerNA          ndarray[object]
        datetime64[ns]     datetime64[ns]
        datetime64[ns, tz] ndarray[object] (Timestamps)
        ================== ================================

        Examples
        --------
        >>> ser = pd.Series(pd.Categorical([\'a\', \'b\', \'a\']))
        >>> ser.to_numpy()
        array([\'a\', \'b\', \'a\'], dtype=object)

        Specify the `dtype` to control how datetime-aware data is represented.
        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`
        objects, each with the correct ``tz``.

        >>> ser = pd.Series(pd.date_range(\'2000\', periods=2, tz="CET"))
        >>> ser.to_numpy(dtype=object)
        array([Timestamp(\'2000-01-01 00:00:00+0100\', tz=\'CET\'),
               Timestamp(\'2000-01-02 00:00:00+0100\', tz=\'CET\')],
              dtype=object)

        Or ``dtype=\'datetime64[ns]\'`` to return an ndarray of native
        datetime64 values. The values are converted to UTC and the timezone
        info is dropped.

        >>> ser.to_numpy(dtype="datetime64[ns]")
        ... # doctest: +ELLIPSIS
        array([\'1999-12-31T23:00:00.000000000\', \'2000-01-01T23:00:00...\'],
              dtype=\'datetime64[ns]\')
        '''
    def argmax(self, axis: AxisInt | None, skipna: bool = ..., *args, **kwargs) -> int:
        """
        Return int position of the largest value in the Series.

        If the maximum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {None}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the maximum value.

        See Also
        --------
        Series.argmax : Return position of the maximum value.
        Series.argmin : Return position of the minimum value.
        numpy.ndarray.argmax : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series({'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0})
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        2
        >>> s.argmin()
        0

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
    def argmin(self, axis: AxisInt | None, skipna: bool = ..., *args, **kwargs) -> int:
        """
        Return int position of the smallest value in the Series.

        If the minimum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {None}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the minimum value.

        See Also
        --------
        Series.argmin : Return position of the minimum value.
        Series.argmax : Return position of the maximum value.
        numpy.ndarray.argmin : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series({'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0})
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        2
        >>> s.argmin()
        0

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
    def tolist(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.

        Examples
        --------
        For Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        >>> idx.to_list()
        [1, 2, 3]
        """
    def to_list(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.

        Examples
        --------
        For Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        >>> idx.to_list()
        [1, 2, 3]
        """
    def __iter__(self) -> Iterator:
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> for x in s:
        ...     print(x)
        1
        2
        3
        """
    def _map_values(self, mapper, na_action, convert: bool = ...):
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            The input correspondence object
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping function
        convert : bool, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object. Note that the dtype is always
            preserved for some extension array dtypes, such as Categorical.

        Returns
        -------
        Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
    def value_counts(self, normalize: bool = ..., sort: bool = ..., ascending: bool = ..., bins, dropna: bool = ...) -> Series:
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : bool, default True
            Sort by frequencies when True. Preserve the order of the data when False.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.
        DataFrame.value_counts: Equivalent method on DataFrames.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        Name: count, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        1.0    0.2
        2.0    0.2
        4.0    0.2
        Name: proportion, dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (0.996, 2.0]    2
        (2.0, 3.0]      2
        (3.0, 4.0]      1
        Name: count, dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        NaN    1
        Name: count, dtype: int64
        """
    def unique(self): ...
    def nunique(self, dropna: bool = ...) -> int:
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the count.

        Returns
        -------
        int

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> s = pd.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64

        >>> s.nunique()
        4
        """
    def _memory_usage(self, deep: bool = ...) -> int:
        """
        Memory usage of the values.

        Parameters
        ----------
        deep : bool, default False
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption.

        Returns
        -------
        bytes used

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
    def factorize(self, sort: bool = ..., use_na_sentinel: bool = ...) -> tuple[npt.NDArray[np.intp], Index]:
        '''
        Encode the object as an enumerated type or categorical variable.

        This method is useful for obtaining a numeric representation of an
        array when all that matters is identifying distinct values. `factorize`
        is available as both a top-level function :func:`pandas.factorize`,
        and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

        Parameters
        ----------
        sort : bool, default False
            Sort `uniques` and shuffle `codes` to maintain the
            relationship.

        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

            .. versionadded:: 1.5.0

        Returns
        -------
        codes : ndarray
            An integer ndarray that\'s an indexer into `uniques`.
            ``uniques.take(codes)`` will have the same values as `values`.
        uniques : ndarray, Index, or Categorical
            The unique valid values. When `values` is Categorical, `uniques`
            is a Categorical. When `values` is some other pandas object, an
            `Index` is returned. Otherwise, a 1-D ndarray is returned.

            .. note::

               Even if there\'s a missing value in `values`, `uniques` will
               *not* contain an entry for it.

        See Also
        --------
        cut : Discretize continuous-valued array.
        unique : Find the unique value in an array.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.factorize>` for more examples.

        Examples
        --------
        These examples all show factorize as a top-level method like
        ``pd.factorize(values)``. The results are identical for methods like
        :meth:`Series.factorize`.

        >>> codes, uniques = pd.factorize(np.array([\'b\', \'b\', \'a\', \'c\', \'b\'], dtype="O"))
        >>> codes
        array([0, 0, 1, 2, 0])
        >>> uniques
        array([\'b\', \'a\', \'c\'], dtype=object)

        With ``sort=True``, the `uniques` will be sorted, and `codes` will be
        shuffled so that the relationship is the maintained.

        >>> codes, uniques = pd.factorize(np.array([\'b\', \'b\', \'a\', \'c\', \'b\'], dtype="O"),
        ...                               sort=True)
        >>> codes
        array([1, 1, 0, 2, 1])
        >>> uniques
        array([\'a\', \'b\', \'c\'], dtype=object)

        When ``use_na_sentinel=True`` (the default), missing values are indicated in
        the `codes` with the sentinel value ``-1`` and missing values are not
        included in `uniques`.

        >>> codes, uniques = pd.factorize(np.array([\'b\', None, \'a\', \'c\', \'b\'], dtype="O"))
        >>> codes
        array([ 0, -1,  1,  2,  0])
        >>> uniques
        array([\'b\', \'a\', \'c\'], dtype=object)

        Thus far, we\'ve only factorized lists (which are internally coerced to
        NumPy arrays). When factorizing pandas objects, the type of `uniques`
        will differ. For Categoricals, a `Categorical` is returned.

        >>> cat = pd.Categorical([\'a\', \'a\', \'c\'], categories=[\'a\', \'b\', \'c\'])
        >>> codes, uniques = pd.factorize(cat)
        >>> codes
        array([0, 0, 1])
        >>> uniques
        [\'a\', \'c\']
        Categories (3, object): [\'a\', \'b\', \'c\']

        Notice that ``\'b\'`` is in ``uniques.categories``, despite not being
        present in ``cat.values``.

        For all other pandas objects, an Index of the appropriate type is
        returned.

        >>> cat = pd.Series([\'a\', \'a\', \'c\'])
        >>> codes, uniques = pd.factorize(cat)
        >>> codes
        array([0, 0, 1])
        >>> uniques
        Index([\'a\', \'c\'], dtype=\'object\')

        If NaN is in the values, and we want to include NaN in the uniques of the
        values, it can be achieved by setting ``use_na_sentinel=False``.

        >>> values = np.array([1, 2, 1, np.nan])
        >>> codes, uniques = pd.factorize(values)  # default: use_na_sentinel=True
        >>> codes
        array([ 0,  1,  0, -1])
        >>> uniques
        array([1., 2.])

        >>> codes, uniques = pd.factorize(values, use_na_sentinel=False)
        >>> codes
        array([0, 1, 0, 2])
        >>> uniques
        array([ 1.,  2., nan])
        '''
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted Index `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        .. note::

            The Index *must* be monotonically sorted, otherwise
            wrong locations will likely be returned. Pandas does *not*
            check this for you.

        Parameters
        ----------
        value : array-like or scalar
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.

        Returns
        -------
        int or array of int
            A scalar or array of insertion points with the
            same shape as `value`.

        See Also
        --------
        sort_values : Sort by the values along either axis.
        numpy.searchsorted : Similar method from NumPy.

        Notes
        -----
        Binary search is used to find the required insertion points.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> ser
        0    1
        1    2
        2    3
        dtype: int64

        >>> ser.searchsorted(4)
        3

        >>> ser.searchsorted([0, 4])
        array([0, 3])

        >>> ser.searchsorted([1, 3], side='left')
        array([0, 2])

        >>> ser.searchsorted([1, 3], side='right')
        array([1, 3])

        >>> ser = pd.Series(pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000']))
        >>> ser
        0   2000-03-11
        1   2000-03-12
        2   2000-03-13
        dtype: datetime64[ns]

        >>> ser.searchsorted('3/14/2000')
        3

        >>> ser = pd.Categorical(
        ...     ['apple', 'bread', 'bread', 'cheese', 'milk'], ordered=True
        ... )
        >>> ser
        ['apple', 'bread', 'bread', 'cheese', 'milk']
        Categories (4, object): ['apple' < 'bread' < 'cheese' < 'milk']

        >>> ser.searchsorted('bread')
        1

        >>> ser.searchsorted(['bread'], side='right')
        array([3])

        If the values are not monotonically sorted, wrong locations
        may be returned:

        >>> ser = pd.Series([2, 1, 3])
        >>> ser
        0    2
        1    1
        2    3
        dtype: int64

        >>> ser.searchsorted(1)  # doctest: +SKIP
        0  # wrong result, correct would be 1
        """
    def drop_duplicates(self, *, keep: DropKeep = ...): ...
    def _duplicated(self, keep: DropKeep = ...) -> npt.NDArray[np.bool_]: ...
    def _arith_method(self, other, op): ...
    def _construct_result(self, result, name):
        """
        Construct an appropriately-wrapped result from the ArrayLike result
        of an arithmetic-like operation.
        """
    @property
    def dtype(self): ...
    @property
    def _values(self): ...
    @property
    def T(self): ...
    @property
    def shape(self): ...
    @property
    def ndim(self): ...
    @property
    def nbytes(self): ...
    @property
    def size(self): ...
    @property
    def array(self): ...
    @property
    def empty(self): ...
    @property
    def is_unique(self): ...
    @property
    def is_monotonic_increasing(self): ...
    @property
    def is_monotonic_decreasing(self): ...

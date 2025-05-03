import lib as lib
import np
import npt
import pandas._libs.algos as libalgos
import pandas._libs.index as libindex
import pandas._libs.join as libjoin
import pandas._libs.lib
import pandas._libs.writers as writers
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algos
import pandas.core.arraylike as arraylike
import pandas.core.base
import pandas.core.common as com
import pandas.core.nanops as nanops
import pandas.core.ops as ops
import pandas.core.strings.accessor
from _typeshed import Incomplete
from builtins import str_t
from collections.abc import Level
from pandas.core.indexes.frozen import FrozenList
from pandas.io.formats.printing import PrettyDict
from typing import Any, AnyAll, ArrayLike, Axis, Callable, ClassVar as _ClassVar, DropKeep, DtypeObj, IgnoreRaise, IndexLabel, JoinHow, Literal, NaPosition, NoReturn, ReindexMethod

__all__ = ['Index']

class Index(pandas.core.base.IndexOpsMixin, pandas.core.base.PandasObject):
    __pandas_priority__: _ClassVar[int] = ...
    _typ: _ClassVar[str] = ...
    _data_cls: _ClassVar[tuple] = ...
    _id: _ClassVar[None] = ...
    _name: _ClassVar[None] = ...
    _no_setting_name: _ClassVar[bool] = ...
    _comparables: _ClassVar[list] = ...
    _attributes: _ClassVar[list] = ...
    _engine_types: _ClassVar[dict] = ...
    _supports_partial_string_indexing: _ClassVar[bool] = ...
    _accessors: _ClassVar[set] = ...
    str: _ClassVar[type[pandas.core.strings.accessor.StringMethods]] = ...
    _references: _ClassVar[None] = ...
    _default_na_rep: _ClassVar[str] = ...
    _requires_unique_msg: _ClassVar[str] = ...
    _can_hold_strings: Incomplete
    _constructor: Incomplete
    _engine: Incomplete
    _dir_additions_for_owner: Incomplete
    dtype: Incomplete
    name: Incomplete
    names: Incomplete
    _can_hold_na: Incomplete
    is_unique: Incomplete
    inferred_type: Incomplete
    _is_all_dates: Incomplete
    _is_multi: Incomplete
    _na_value: Incomplete
    _isnan: Incomplete
    hasnans: Incomplete
    _can_use_libjoin: Incomplete
    array: Incomplete
    _should_fallback_to_positional: Incomplete
    def _left_indexer_unique(self, other: Self) -> npt.NDArray[np.intp]: ...
    def _left_indexer(self, other: Self) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    def _inner_indexer(self, other: Self) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    def _outer_indexer(self, other: Self) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    @classmethod
    def __init__(cls, data, dtype, copy: bool = ..., name, tupleize_cols: bool = ...) -> Self: ...
    @classmethod
    def _ensure_array(cls, data, dtype, copy: bool):
        """
        Ensure we have a valid array to pass to _simple_new.
        """
    @classmethod
    def _dtype_to_subclass(cls, dtype: DtypeObj): ...
    @classmethod
    def _simple_new(cls, values: ArrayLike, name: Hashable | None, refs) -> Self:
        """
        We require that we have a dtype compat for the values. If we are passed
        a non-dtype compat, then coerce using the constructor.

        Must be careful not to recurse.
        """
    @classmethod
    def _with_infer(cls, *args, **kwargs):
        """
        Constructor that uses the 1.0.x behavior inferring numeric dtypes
        for ndarray[object] inputs.
        """
    def _maybe_check_unique(self) -> None:
        """
        Check that an Index has no duplicates.

        This is typically only called via
        `NDFrame.flags.allows_duplicate_labels.setter` when it's set to
        True (duplicates aren't allowed).

        Raises
        ------
        DuplicateLabelError
            When the index is not unique.
        """
    def _format_duplicate_message(self) -> DataFrame:
        """
        Construct the DataFrame for a DuplicateLabelError.

        This returns a DataFrame indicating the labels and positions
        of duplicates in an index. This should only be called when it's
        already known that duplicates are present.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'a'])
        >>> idx._format_duplicate_message()
            positions
        label
        a        [0, 2]
        """
    def _shallow_copy(self, values, name: Hashable = ...) -> Self:
        """
        Create a new Index with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        name : Label, defaults to self.name
        """
    def _view(self) -> Self:
        """
        fastpath to make a shallow copy, i.e. new object with same data.
        """
    def _rename(self, name: Hashable) -> Self:
        """
        fastpath for rename if new name is already validated.
        """
    def is_(self, other) -> bool:
        """
        More flexible, faster check like ``is`` but that works through views.

        Note: this is *not* the same as ``Index.identical()``, which checks
        that metadata is also the same.

        Parameters
        ----------
        other : object
            Other object to compare against.

        Returns
        -------
        bool
            True if both have same underlying data, False otherwise.

        See Also
        --------
        Index.identical : Works like ``Index.is_`` but also checks metadata.

        Examples
        --------
        >>> idx1 = pd.Index(['1', '2', '3'])
        >>> idx1.is_(idx1.view())
        True

        >>> idx1.is_(idx1.copy())
        False
        """
    def _reset_identity(self) -> None:
        """
        Initializes or resets ``_id`` attribute with new object.
        """
    def _cleanup(self) -> None: ...
    def __len__(self) -> int:
        """
        Return the length of the Index.
        """
    def __array__(self, dtype, copy) -> np.ndarray:
        """
        The array interface, return my values.
        """
    def __array_ufunc__(self, ufunc: np.ufunc, method: str_t, *inputs, **kwargs): ...
    def __array_wrap__(self, result, context, return_scalar: bool = ...):
        """
        Gets called after a ufunc and other functions e.g. np.split.
        """
    def ravel(self, order: str_t = ...) -> Self:
        """
        Return a view on self.

        Returns
        -------
        Index

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        >>> s.index.ravel()
        Index(['a', 'b', 'c'], dtype='object')
        """
    def view(self, cls): ...
    def astype(self, dtype, copy: bool = ...):
        """
        Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
            Index with values cast to specified dtype.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.astype('float')
        Index([1.0, 2.0, 3.0], dtype='float64')
        """
    def take(self, indices, axis: Axis = ..., allow_fill: bool = ..., fill_value, **kwargs) -> Self:
        """
        Return a new Index of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : int, optional
            The axis over which to select values, always 0.
        allow_fill : bool, default True
        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.take([2, 2, 1, 2])
        Index(['c', 'c', 'b', 'c'], dtype='object')
        """
    def _maybe_disallow_fill(self, allow_fill: bool, fill_value, indices) -> bool:
        """
        We only use pandas-style take when allow_fill is True _and_
        fill_value is not None.
        """
    def repeat(self, repeats, axis: None) -> Self:
        """
        Repeat elements of a Index.

        Returns a new Index where each element of the current Index
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Index.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        Index
            Newly created Index with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.repeat(2)
        Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')
        >>> idx.repeat([1, 2, 3])
        Index(['a', 'b', 'b', 'c', 'c', 'c'], dtype='object')
        """
    def copy(self, name: Hashable | None, deep: bool = ...) -> Self:
        """
        Make a copy of this object.

        Name is set on the new object.

        Parameters
        ----------
        name : Label, optional
            Set name for new object.
        deep : bool, default False

        Returns
        -------
        Index
            Index refer to new object which is a copy of this object.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> new_idx = idx.copy()
        >>> idx is new_idx
        False
        """
    def __copy__(self, **kwargs) -> Self: ...
    def __deepcopy__(self, memo) -> Self:
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
    def _format_data(self, name) -> str_t:
        """
        Return the formatted data as a unicode string.
        """
    def _format_attrs(self) -> list[tuple[str_t, str_t | int | bool | None]]:
        """
        Return a list of tuples of the (attr,formatted_value).
        """
    def _get_level_names(self) -> Hashable | Sequence[Hashable]:
        """
        Return a name or list of names with None replaced by the level number.
        """
    def _mpl_repr(self) -> np.ndarray: ...
    def format(self, name: bool = ..., formatter: Callable | None, na_rep: str_t = ...) -> list[str_t]:
        """
        Render a string representation of the Index.
        """
    def _format_flat(self, *, include_name: bool, formatter: Callable | None) -> list[str_t]:
        """
        Render a string representation of the Index.
        """
    def _format_with_header(self, *, header: list[str_t], na_rep: str_t) -> list[str_t]: ...
    def _get_values_for_csv(self, *, na_rep: str_t = ..., decimal: str_t = ..., float_format, date_format, quoting) -> npt.NDArray[np.object_]: ...
    def _summary(self, name) -> str_t:
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
    def to_flat_index(self) -> Self:
        """
        Identity method.

        This is implemented for compatibility with subclass implementations
        when chaining.

        Returns
        -------
        pd.Index
            Caller.

        See Also
        --------
        MultiIndex.to_flat_index : Subclass implementation.
        """
    def to_series(self, index, name: Hashable | None) -> Series:
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.

        See Also
        --------
        Index.to_frame : Convert an Index to a DataFrame.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')

        By default, the original index and original name is reused.

        >>> idx.to_series()
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: animal, dtype: object

        To enforce a new index, specify new labels to ``index``:

        >>> idx.to_series(index=[0, 1, 2])
        0     Ant
        1    Bear
        2     Cow
        Name: animal, dtype: object

        To override the name of the resulting column, specify ``name``:

        >>> idx.to_series(name='zoo')
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: zoo, dtype: object
        """
    def to_frame(self, index: bool = ..., name: Hashable = ...) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original Index.

        name : object, defaults to index.name
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
    def _validate_names(self, name, names, deep: bool = ...) -> list[Hashable]:
        """
        Handles the quirks of having a singular 'name' parameter for general
        Index and plural 'names' parameter for MultiIndex.
        """
    def _get_default_index_names(self, names: Hashable | Sequence[Hashable] | None, default) -> list[Hashable]:
        """
        Get names of index.

        Parameters
        ----------
        names : int, str or 1-dimensional list, default None
            Index names to set.
        default : str
            Default name of index.

        Raises
        ------
        TypeError
            if names not str or list-like
        """
    def _get_names(self) -> FrozenList: ...
    def _set_names(self, values, *, level) -> None:
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None

        Raises
        ------
        TypeError if each name is not hashable.
        """
    def set_names(self, names, *, level, inplace: bool = ...) -> Self | None:
        """
        Set Index or MultiIndex name.

        Able to set new names partially and by level.

        Parameters
        ----------

        names : label or list of label or dict-like for MultiIndex
            Name(s) to set.

            .. versionchanged:: 1.3.0

        level : int, label or list of int or label, optional
            If the index is a MultiIndex and names is not dict-like, level(s) to set
            (None for all levels). Otherwise level must be None.

            .. versionchanged:: 1.3.0

        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Index([1, 2, 3, 4], dtype='int64', name='quarter')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx = idx.set_names(['kind', 'year'])
        >>> idx.set_names('species', level=0)
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])

        When renaming levels with a dict, levels can not be passed.

        >>> idx.set_names({'kind': 'snake'})
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['snake', 'year'])
        """
    def rename(self, name, *, inplace: bool = ...) -> Self | None:
        """
        Alter Index or MultiIndex name.

        Able to set new names without level. Defaults to returning new index.
        Length of names must match number of levels in MultiIndex.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.set_names : Able to set new names partially and by level.

        Examples
        --------
        >>> idx = pd.Index(['A', 'C', 'A', 'B'], name='score')
        >>> idx.rename('grade')
        Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]],
        ...                                   names=['kind', 'year'])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['kind', 'year'])
        >>> idx.rename(['species', 'year'])
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])
        >>> idx.rename('species')
        Traceback (most recent call last):
        TypeError: Must pass list-like as `names`.
        """
    def _sort_levels_monotonic(self) -> Self:
        """
        Compat with MultiIndex.
        """
    def _validate_index_level(self, level) -> None:
        """
        Validate index level.

        For single-level Index getting level number is a no-op, but some
        verification must be done like in MultiIndex.

        """
    def _get_level_number(self, level) -> int: ...
    def sortlevel(self, level, ascending: bool | list[bool] = ..., sort_remaining, na_position: NaPosition = ...):
        """
        For internal compatibility with the Index API.

        Sort the Index. This is for compat with MultiIndex

        Parameters
        ----------
        ascending : bool, default True
            False to sort in descending order
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        level, sort_remaining are compat parameters

        Returns
        -------
        Index
        """
    def _get_level_values(self, level) -> Index:
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        MultiIndex.get_level_values : Get values for a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> idx = pd.Index(list('abc'))
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        Get level values by supplying `level` as integer:

        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object')
        """
    def get_level_values(self, level) -> Index:
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        MultiIndex.get_level_values : Get values for a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> idx = pd.Index(list('abc'))
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        Get level values by supplying `level` as integer:

        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object')
        """
    def droplevel(self, level: IndexLabel = ...):
        """
        Return index with requested level(s) removed.

        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex. The original index is not modified inplace.

        Parameters
        ----------
        level : int, str, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])

        >>> mi.droplevel()
        MultiIndex([(3, 5),
                    (4, 6)],
                   names=['y', 'z'])

        >>> mi.droplevel(2)
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel('z')
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel(['x', 'y'])
        Index([5, 6], dtype='int64', name='z')
        """
    def _drop_level_numbers(self, levnums: list[int]):
        """
        Drop MultiIndex levels by level _number_, not name.
        """
    def is_boolean(self) -> bool:
        '''
        Check if the Index only consists of booleans.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_bool_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of booleans.

        See Also
        --------
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype (deprecated).
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([True, False, True])
        >>> idx.is_boolean()  # doctest: +SKIP
        True

        >>> idx = pd.Index(["True", "False", "True"])
        >>> idx.is_boolean()  # doctest: +SKIP
        False

        >>> idx = pd.Index([True, False, "True"])
        >>> idx.is_boolean()  # doctest: +SKIP
        False
        '''
    def is_integer(self) -> bool:
        '''
        Check if the Index only consists of integers.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_integer_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of integers.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_integer()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_integer()  # doctest: +SKIP
        False

        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_integer()  # doctest: +SKIP
        False
        '''
    def is_floating(self) -> bool:
        """
        Check if the Index is a floating type.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_float_dtype` instead

        The Index may consist of only floats, NaNs, or a mix of floats,
        integers, or NaNs.

        Returns
        -------
        bool
            Whether or not the Index only consists of only consists of floats, NaNs, or
            a mix of floats, integers, or NaNs.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_floating()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1.0, 2.0, np.nan, 4.0])
        >>> idx.is_floating()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4, np.nan])
        >>> idx.is_floating()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_floating()  # doctest: +SKIP
        False
        """
    def is_numeric(self) -> bool:
        '''
        Check if the Index only consists of numeric data.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_numeric_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of numeric data.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4.0])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan, "Apple"])
        >>> idx.is_numeric()  # doctest: +SKIP
        False
        '''
    def is_object(self) -> bool:
        '''
        Check if the Index is of the object dtype.

        .. deprecated:: 2.0.0
           Use `pandas.api.types.is_object_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index is of the object dtype.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_object()  # doctest: +SKIP
        True

        >>> idx = pd.Index(["Apple", "Mango", 2.0])
        >>> idx.is_object()  # doctest: +SKIP
        True

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_object()  # doctest: +SKIP
        False

        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_object()  # doctest: +SKIP
        False
        '''
    def is_categorical(self) -> bool:
        '''
        Check if the Index holds categorical data.

        .. deprecated:: 2.0.0
              Use `isinstance(index.dtype, pd.CategoricalDtype)` instead.

        Returns
        -------
        bool
            True if the Index is categorical.

        See Also
        --------
        CategoricalIndex : Index for categorical data.
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_categorical()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_categorical()  # doctest: +SKIP
        False

        >>> s = pd.Series(["Peter", "Victor", "Elisabeth", "Mar"])
        >>> s
        0        Peter
        1       Victor
        2    Elisabeth
        3          Mar
        dtype: object
        >>> s.index.is_categorical()  # doctest: +SKIP
        False
        '''
    def is_interval(self) -> bool:
        """
        Check if the Index holds Interval objects.

        .. deprecated:: 2.0.0
            Use `isinstance(index.dtype, pd.IntervalDtype)` instead.

        Returns
        -------
        bool
            Whether or not the Index holds Interval objects.

        See Also
        --------
        IntervalIndex : Index for Interval objects.
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).

        Examples
        --------
        >>> idx = pd.Index([pd.Interval(left=0, right=5),
        ...                 pd.Interval(left=5, right=10)])
        >>> idx.is_interval()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_interval()  # doctest: +SKIP
        False
        """
    def _holds_integer(self) -> bool:
        """
        Whether the type is an integer type.
        """
    def holds_integer(self) -> bool:
        """
        Whether the type is an integer type.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.infer_dtype` instead
        """
    def __reduce__(self): ...
    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values. Characters such as
        empty strings `''` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array of whether my values are NA.

        See Also
        --------
        Index.notna : Boolean inverse of isna.
        Index.dropna : Omit entries with missing values.
        isna : Top-level isna.
        Series.isna : Detect missing values in Series object.

        Examples
        --------
        Show which entries in a pandas.Index are NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.isna()
        array([False, False,  True])

        Empty strings are not considered NA values. None is considered an NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.isna()
        array([False, False, False,  True])

        For datetimes, `NaT` (Not a Time) is considered as an NA value.

        >>> idx = pd.DatetimeIndex([pd.Timestamp('1940-04-25'),
        ...                         pd.Timestamp(''), None, pd.NaT])
        >>> idx
        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.isna()
        array([False,  True,  True,  True])
        """
    def isnull(self) -> npt.NDArray[np.bool_]:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values. Characters such as
        empty strings `''` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array of whether my values are NA.

        See Also
        --------
        Index.notna : Boolean inverse of isna.
        Index.dropna : Omit entries with missing values.
        isna : Top-level isna.
        Series.isna : Detect missing values in Series object.

        Examples
        --------
        Show which entries in a pandas.Index are NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.isna()
        array([False, False,  True])

        Empty strings are not considered NA values. None is considered an NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.isna()
        array([False, False, False,  True])

        For datetimes, `NaT` (Not a Time) is considered as an NA value.

        >>> idx = pd.DatetimeIndex([pd.Timestamp('1940-04-25'),
        ...                         pd.Timestamp(''), None, pd.NaT])
        >>> idx
        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.isna()
        array([False,  True,  True,  True])
        """
    def notna(self) -> npt.NDArray[np.bool_]:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``
        values.

        Returns
        -------
        numpy.ndarray[bool]
            Boolean array to indicate which entries are not NA.

        See Also
        --------
        Index.notnull : Alias of notna.
        Index.isna: Inverse of notna.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in an Index are not NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False])

        Empty strings are not considered NA values. None is considered a NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.notna()
        array([ True,  True,  True, False])
        """
    def notnull(self) -> npt.NDArray[np.bool_]:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``
        values.

        Returns
        -------
        numpy.ndarray[bool]
            Boolean array to indicate which entries are not NA.

        See Also
        --------
        Index.notnull : Alias of notna.
        Index.isna: Inverse of notna.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in an Index are not NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False])

        Empty strings are not considered NA values. None is considered a NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.notna()
        array([ True,  True,  True, False])
        """
    def fillna(self, value, downcast: pandas._libs.lib._NoDefault = ...):
        """
        Fill NA/NaN values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

            .. deprecated:: 2.1.0

        Returns
        -------
        Index

        See Also
        --------
        DataFrame.fillna : Fill NaN values of a DataFrame.
        Series.fillna : Fill NaN Values of a Series.

        Examples
        --------
        >>> idx = pd.Index([np.nan, np.nan, 3])
        >>> idx.fillna(0)
        Index([0.0, 0.0, 3.0], dtype='float64')
        """
    def dropna(self, how: AnyAll = ...) -> Self:
        """
        Return Index without NA/NaN values.

        Parameters
        ----------
        how : {'any', 'all'}, default 'any'
            If the Index is a MultiIndex, drop the value when any or all levels
            are NaN.

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index([1, np.nan, 3])
        >>> idx.dropna()
        Index([1.0, 3.0], dtype='float64')
        """
    def unique(self, level: Hashable | None) -> Self:
        """
        Return unique values in the index.

        Unique values are returned in order of appearance, this does NOT sort.

        Parameters
        ----------
        level : int or hashable, optional
            Only return values from specified level (for MultiIndex).
            If int, gets the level by integer position, else by level name.

        Returns
        -------
        Index

        See Also
        --------
        unique : Numpy array of unique values in that column.
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> idx = pd.Index([1, 1, 2, 3, 3])
        >>> idx.unique()
        Index([1, 2, 3], dtype='int64')
        """
    def drop_duplicates(self, *, keep: DropKeep = ...) -> Self:
        """
        Return Index with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        Returns
        -------
        Index

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Index.duplicated : Related method on Index, indicating duplicate
            Index values.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

        The `keep` parameter controls  which duplicate values are removed.
        The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> idx.drop_duplicates(keep='first')
        Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')

        The value 'last' keeps the last occurrence for each set of duplicated
        entries.

        >>> idx.drop_duplicates(keep='last')
        Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')

        The value ``False`` discards all sets of duplicated entries.

        >>> idx.drop_duplicates(keep=False)
        Index(['cow', 'beetle', 'hippo'], dtype='object')
        """
    def duplicated(self, keep: DropKeep = ...) -> npt.NDArray[np.bool_]:
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        np.ndarray[bool]

        See Also
        --------
        Series.duplicated : Equivalent method on pandas.Series.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep='first')
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> idx.duplicated(keep='last')
        array([ True, False,  True, False, False])

        By setting keep on ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
    def __iadd__(self, other): ...
    def __nonzero__(self) -> NoReturn: ...
    def __bool__(self) -> NoReturn: ...
    def _get_reconciled_name_object(self, other):
        """
        If the result of a set operation will be self,
        return self, unless the name changes, in which
        case make a shallow copy of self.
        """
    def _validate_sort_keyword(self, sort): ...
    def _dti_setop_align_tzs(self, other: Index, setop: str_t) -> tuple[Index, Index]:
        """
        With mismatched timezones, cast both to UTC.
        """
    def union(self, other, sort):
        '''
        Form the union of two Index objects.

        If the Index objects are incompatible, both Index objects will be
        cast to dtype(\'object\') first.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.
              3. Some values in `self` or `other` cannot be compared.
                 A RuntimeWarning is issued in this case.

            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Examples
        --------
        Union matching dtypes

        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Index([1, 2, 3, 4, 5, 6], dtype=\'int64\')

        Union mismatched dtypes

        >>> idx1 = pd.Index([\'a\', \'b\', \'c\', \'d\'])
        >>> idx2 = pd.Index([1, 2, 3, 4])
        >>> idx1.union(idx2)
        Index([\'a\', \'b\', \'c\', \'d\', 1, 2, 3, 4], dtype=\'object\')

        MultiIndex case

        >>> idx1 = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ... )
        >>> idx1
        MultiIndex([(1,  \'Red\'),
            (1, \'Blue\'),
            (2,  \'Red\'),
            (2, \'Blue\')],
           )
        >>> idx2 = pd.MultiIndex.from_arrays(
        ...     [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
        ... )
        >>> idx2
        MultiIndex([(3,   \'Red\'),
            (3, \'Green\'),
            (2,   \'Red\'),
            (2, \'Green\')],
           )
        >>> idx1.union(idx2)
        MultiIndex([(1,  \'Blue\'),
            (1,   \'Red\'),
            (2,  \'Blue\'),
            (2, \'Green\'),
            (2,   \'Red\'),
            (3, \'Green\'),
            (3,   \'Red\')],
           )
        >>> idx1.union(idx2, sort=False)
        MultiIndex([(1,   \'Red\'),
            (1,  \'Blue\'),
            (2,   \'Red\'),
            (2,  \'Blue\'),
            (3,   \'Red\'),
            (3, \'Green\'),
            (2, \'Green\')],
           )
        '''
    def _union(self, other: Index, sort: bool | None):
        """
        Specific union logic should go here. In subclasses, union behavior
        should be overwritten here rather than in `self.union`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * True : sort the result
            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.

        Returns
        -------
        Index
        """
    def _wrap_setop_result(self, other: Index, result) -> Index: ...
    def intersection(self, other, sort: bool = ...):
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
        sort : True, False or None, default False
            Whether to sort the resulting index.

            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.
            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Index([3, 4], dtype='int64')
        """
    def _intersection(self, other: Index, sort: bool = ...):
        """
        intersection specialized to the case with matching dtypes.
        """
    def _wrap_intersection_result(self, other, result): ...
    def _intersection_via_get_indexer(self, other: Index | MultiIndex, sort) -> ArrayLike | MultiIndex:
        """
        Find the intersection of two Indexes using get_indexer.

        Returns
        -------
        np.ndarray or ExtensionArray or MultiIndex
            The returned array will be unique.
        """
    def difference(self, other, sort):
        """
        Return a new Index with elements of index not in `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Examples
        --------
        >>> idx1 = pd.Index([2, 1, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2)
        Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Index([2, 1], dtype='int64')
        """
    def _difference(self, other, sort): ...
    def _wrap_difference_result(self, other, result): ...
    def symmetric_difference(self, other, result_name, sort):
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : str
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([2, 3, 4, 5])
        >>> idx1.symmetric_difference(idx2)
        Index([1, 5], dtype='int64')
        """
    def _assert_can_do_setop(self, other) -> bool: ...
    def _convert_can_do_setop(self, other) -> tuple[Index, Hashable]: ...
    def get_loc(self, key):
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> unique_index = pd.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1

        >>> monotonic_index = pd.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)

        >>> non_monotonic_index = pd.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True])
        """
    def get_indexer(self, target, method: ReindexMethod | None, limit: int | None, tolerance) -> npt.NDArray[np.intp]:
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : Index
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.

        Notes
        -----
        Returns -1 for unmatched values, for further explanation see the
        example below.

        Examples
        --------
        >>> index = pd.Index(['c', 'a', 'b'])
        >>> index.get_indexer(['a', 'b', 'x'])
        array([ 1,  2, -1])

        Notice that the return value is an array of locations in ``index``
        and ``x`` is marked by -1, as it is not in ``index``.
        """
    def _get_indexer(self, target: Index, method: str_t | None, limit: int | None, tolerance) -> npt.NDArray[np.intp]: ...
    def _should_partial_index(self, target: Index) -> bool:
        """
        Should we attempt partial-matching indexing?
        """
    def _check_indexing_method(self, method: str_t | None, limit: int | None, tolerance) -> None:
        """
        Raise if we have a get_indexer `method` that is not supported or valid.
        """
    def _convert_tolerance(self, tolerance, target: np.ndarray | Index) -> np.ndarray: ...
    def _get_fill_indexer(self, target: Index, method: str_t, limit: int | None, tolerance) -> npt.NDArray[np.intp]: ...
    def _get_fill_indexer_searchsorted(self, target: Index, method: str_t, limit: int | None) -> npt.NDArray[np.intp]:
        """
        Fallback pad/backfill get_indexer that works for monotonic decreasing
        indexes and non-monotonic targets.
        """
    def _get_nearest_indexer(self, target: Index, limit: int | None, tolerance) -> npt.NDArray[np.intp]:
        """
        Get the indexer for the nearest index labels; requires an index with
        values that can be subtracted from each other (e.g., not strings or
        tuples).
        """
    def _filter_indexer_tolerance(self, target: Index, indexer: npt.NDArray[np.intp], tolerance) -> npt.NDArray[np.intp]: ...
    def _difference_compat(self, target: Index, indexer: npt.NDArray[np.intp]) -> ArrayLike: ...
    def _validate_positional_slice(self, key: slice) -> None:
        """
        For positional indexing, a slice must have either int or None
        for each of start, stop, and step.
        """
    def _convert_slice_indexer(self, key: slice, kind: Literal['loc', 'getitem']):
        """
        Convert a slice indexer.

        By definition, these are labels unless 'iloc' is passed in.
        Floats are not allowed as the start, step, or stop of the slice.

        Parameters
        ----------
        key : label of the slice bound
        kind : {'loc', 'getitem'}
        """
    def _raise_invalid_indexer(self, form: Literal['slice', 'positional'], key, reraise: lib.NoDefault | None | Exception = ...) -> None:
        """
        Raise consistent invalid indexer message.
        """
    def _validate_can_reindex(self, indexer: np.ndarray) -> None:
        """
        Check if we are allowing reindexing with this particular indexer.

        Parameters
        ----------
        indexer : an integer ndarray

        Raises
        ------
        ValueError if its a duplicate axis
        """
    def reindex(self, target, method: ReindexMethod | None, level, limit: int | None, tolerance: float | None) -> tuple[Index, npt.NDArray[np.intp] | None]:
        """
        Create index with target's values.

        Parameters
        ----------
        target : an iterable
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        level : int, optional
            Level of multiindex.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : int or float, optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index.

        Raises
        ------
        TypeError
            If ``method`` passed along with ``level``.
        ValueError
            If non-unique multi-index
        ValueError
            If non-unique index and ``method`` or ``limit`` passed.

        See Also
        --------
        Series.reindex : Conform Series to new index with optional filling logic.
        DataFrame.reindex : Conform DataFrame to new index with optional filling logic.

        Examples
        --------
        >>> idx = pd.Index(['car', 'bike', 'train', 'tractor'])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.reindex(['car', 'bike'])
        (Index(['car', 'bike'], dtype='object'), array([0, 1]))
        """
    def _wrap_reindex_result(self, target, indexer, preserve_names: bool): ...
    def _maybe_preserve_names(self, target: Index, preserve_names: bool): ...
    def _reindex_non_unique(self, target: Index) -> tuple[Index, npt.NDArray[np.intp], npt.NDArray[np.intp] | None]:
        """
        Create a new index with target's values (move/add/delete values as
        necessary) use with non-unique Index and a possibly non-unique target.

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.
        new_indexer : np.ndarray[np.intp] or None

        """
    def join(self, other: Index, *, how: JoinHow = ..., level: Level | None, return_indexers: bool = ..., sort: bool = ...) -> Index | tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """
        Compute join_index and indexers to conform data structures to the new index.

        Parameters
        ----------
        other : Index
        how : {'left', 'right', 'inner', 'outer'}
        level : int or level name, default None
        return_indexers : bool, default False
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns
        -------
        join_index, (left_indexer, right_indexer)

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([4, 5, 6])
        >>> idx1.join(idx2, how='outer')
        Index([1, 2, 3, 4, 5, 6], dtype='int64')
        """
    def _join_empty(self, other: Index, how: JoinHow, sort: bool) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]: ...
    def _join_via_get_indexer(self, other: Index, how: JoinHow, sort: bool) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]: ...
    def _join_multi(self, other: Index, how: JoinHow): ...
    def _join_non_unique(self, other: Index, how: JoinHow = ..., sort: bool = ...) -> tuple[Index, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    def _join_level(self, other: Index, level, how: JoinHow = ..., keep_order: bool = ...) -> tuple[MultiIndex, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """
        The join method *only* affects the level of the resulting
        MultiIndex. Otherwise it just exactly aligns the Index data to the
        labels of the level in the MultiIndex.

        If ```keep_order == True```, the order of the data indexed by the
        MultiIndex will not be changed; otherwise, it will tie out
        with `other`.
        """
    def _join_monotonic(self, other: Index, how: JoinHow = ...) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]: ...
    def _wrap_joined_index(self, joined: ArrayLike, other: Self, lidx: npt.NDArray[np.intp], ridx: npt.NDArray[np.intp]) -> Self: ...
    def _get_engine_target(self) -> ArrayLike:
        """
        Get the ndarray or ExtensionArray that we can pass to the IndexEngine
        constructor.
        """
    def _get_join_target(self) -> np.ndarray:
        """
        Get the ndarray or ExtensionArray that we can pass to the join
        functions.
        """
    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        """
        Cast the ndarray returned from one of the libjoin.foo_indexer functions
        back to type(self._data).
        """
    def memory_usage(self, deep: bool = ...) -> int:
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
    def where(self, cond, other) -> Index:
        """
        Replace values where the condition is False.

        The replacement is taken from other.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Condition to select the values on.
        other : scalar, or array-like, default None
            Replacement if the condition is False.

        Returns
        -------
        pandas.Index
            A copy of self with values replaced from other
            where the condition is False.

        See Also
        --------
        Series.where : Same method for Series.
        DataFrame.where : Same method for DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['car', 'bike', 'train', 'tractor'])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.where(idx.isin(['car', 'train']), 'other')
        Index(['car', 'other', 'train', 'other'], dtype='object')
        """
    @classmethod
    def _raise_scalar_data_error(cls, data): ...
    def _validate_fill_value(self, value):
        """
        Check if the value can be inserted into our array without casting,
        and convert it to an appropriate native type if necessary.

        Raises
        ------
        TypeError
            If the value cannot be inserted into an array of this dtype.
        """
    def _is_memory_usage_qualified(self) -> bool:
        """
        Return a boolean if we need a qualified .info display.
        """
    def __contains__(self, key: Any) -> bool:
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
    def __setitem__(self, key, value) -> None: ...
    def __getitem__(self, key):
        """
        Override numpy.ndarray's __getitem__ method to work as desired.

        This function adds lists and Series as valid boolean indexers
        (ndarrays only supports ndarray with dtype=bool).

        If resulting ndim != 1, plain ndarray is returned instead of
        corresponding `Index` subclass.

        """
    def _getitem_slice(self, slobj: slice) -> Self:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
    def _can_hold_identifiers_and_holds_name(self, name) -> bool:
        """
        Faster check for ``name in self`` when we know `name` is a Python
        identifier (e.g. in NDFrame.__getattr__, which hits this to support
        . key lookup). For indexes that can't hold identifiers (everything
        but object & categorical) we just return False.

        https://github.com/pandas-dev/pandas/issues/19764
        """
    def append(self, other: Index | Sequence[Index]) -> Index:
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.append(pd.Index([4]))
        Index([1, 2, 3, 4], dtype='int64')
        """
    def _concat(self, to_concat: list[Index], name: Hashable) -> Index:
        """
        Concatenate multiple Index objects.
        """
    def putmask(self, mask, value) -> Index:
        """
        Return a new Index of the values set with the mask.

        Returns
        -------
        Index

        See Also
        --------
        numpy.ndarray.putmask : Changes elements of an array
            based on conditional and input values.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([5, 6, 7])
        >>> idx1.putmask([True, False, False], idx2)
        Index([5, 2, 3], dtype='int64')
        """
    def equals(self, other: Any) -> bool:
        '''
        Determine if two Index object are equal.

        The things that are being compared are:

        * The elements inside the Index object.
        * The order of the elements inside the Index object.

        Parameters
        ----------
        other : Any
            The other object to compare against.

        Returns
        -------
        bool
            True if "other" is an Index and it has the same elements and order
            as the calling index; False otherwise.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx1
        Index([1, 2, 3], dtype=\'int64\')
        >>> idx1.equals(pd.Index([1, 2, 3]))
        True

        The elements inside are compared

        >>> idx2 = pd.Index(["1", "2", "3"])
        >>> idx2
        Index([\'1\', \'2\', \'3\'], dtype=\'object\')

        >>> idx1.equals(idx2)
        False

        The order is compared

        >>> ascending_idx = pd.Index([1, 2, 3])
        >>> ascending_idx
        Index([1, 2, 3], dtype=\'int64\')
        >>> descending_idx = pd.Index([3, 2, 1])
        >>> descending_idx
        Index([3, 2, 1], dtype=\'int64\')
        >>> ascending_idx.equals(descending_idx)
        False

        The dtype is *not* compared

        >>> int64_idx = pd.Index([1, 2, 3], dtype=\'int64\')
        >>> int64_idx
        Index([1, 2, 3], dtype=\'int64\')
        >>> uint64_idx = pd.Index([1, 2, 3], dtype=\'uint64\')
        >>> uint64_idx
        Index([1, 2, 3], dtype=\'uint64\')
        >>> int64_idx.equals(uint64_idx)
        True
        '''
    def identical(self, other) -> bool:
        '''
        Similar to equals, but checks that object attributes and types are also equal.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.

        Examples
        --------
        >>> idx1 = pd.Index([\'1\', \'2\', \'3\'])
        >>> idx2 = pd.Index([\'1\', \'2\', \'3\'])
        >>> idx2.identical(idx1)
        True

        >>> idx1 = pd.Index([\'1\', \'2\', \'3\'], name="A")
        >>> idx2 = pd.Index([\'1\', \'2\', \'3\'], name="B")
        >>> idx2.identical(idx1)
        False
        '''
    def asof(self, label):
        """
        Return the label from the index, or, if not present, the previous one.

        Assuming that the index is sorted, return the passed index label if it
        is in the index, or return the previous index label if the passed one
        is not in the index.

        Parameters
        ----------
        label : object
            The label up to which the method returns the latest index label.

        Returns
        -------
        object
            The passed label if it is in the index. The previous label if the
            passed label is not in the sorted index or `NaN` if there is no
            such label.

        See Also
        --------
        Series.asof : Return the latest value in a Series up to the
            passed index.
        merge_asof : Perform an asof merge (similar to left join but it
            matches on nearest key rather than equal key).
        Index.get_loc : An `asof` is a thin wrapper around `get_loc`
            with method='pad'.

        Examples
        --------
        `Index.asof` returns the latest index label up to the passed label.

        >>> idx = pd.Index(['2013-12-31', '2014-01-02', '2014-01-03'])
        >>> idx.asof('2014-01-01')
        '2013-12-31'

        If the label is in the index, the method returns the passed label.

        >>> idx.asof('2014-01-02')
        '2014-01-02'

        If all of the labels in the index are later than the passed label,
        NaN is returned.

        >>> idx.asof('1999-01-02')
        nan

        If the index is not sorted, an error is raised.

        >>> idx_not_sorted = pd.Index(['2013-12-31', '2015-01-02',
        ...                            '2014-01-03'])
        >>> idx_not_sorted.asof('2013-12-31')
        Traceback (most recent call last):
        ValueError: index must be monotonic increasing or decreasing
        """
    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
        """
        Return the locations (indices) of labels in the index.

        As in the :meth:`pandas.Index.asof`, if the label (a particular entry in
        ``where``) is not in the index, the latest index label up to the
        passed label is chosen and its index returned.

        If all of the labels in the index are later than a label in ``where``,
        -1 is returned.

        ``mask`` is used to ignore ``NA`` values in the index during calculation.

        Parameters
        ----------
        where : Index
            An Index consisting of an array of timestamps.
        mask : np.ndarray[bool]
            Array of booleans denoting where values in the original
            data are not ``NA``.

        Returns
        -------
        np.ndarray[np.intp]
            An array of locations (indices) of the labels from the index
            which correspond to the return values of :meth:`pandas.Index.asof`
            for every element in ``where``.

        See Also
        --------
        Index.asof : Return the label from the index, or, if not present, the
            previous one.

        Examples
        --------
        >>> idx = pd.date_range('2023-06-01', periods=3, freq='D')
        >>> where = pd.DatetimeIndex(['2023-05-30 00:12:00', '2023-06-01 00:00:00',
        ...                           '2023-06-02 23:59:59'])
        >>> mask = np.ones(3, dtype=bool)
        >>> idx.asof_locs(where, mask)
        array([-1,  0,  1])

        We can use ``mask`` to ignore certain values in the index during calculation.

        >>> mask[1] = False
        >>> idx.asof_locs(where, mask)
        array([-1,  0,  0])
        """
    def sort_values(self, *, return_indexer: bool = ..., ascending: bool = ..., na_position: NaPosition = ..., key: Callable | None) -> Self | tuple[Self, np.ndarray]:
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        sorted_index : pandas.Index
            Sorted copy of the index.
        indexer : numpy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([10, 100, 1, 1000])
        >>> idx
        Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))
        """
    def sort(self, *args, **kwargs):
        """
        Use sort_values instead.
        """
    def shift(self, periods: int = ..., freq):
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or str, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.Index
            Shifted index.

        See Also
        --------
        Series.shift : Shift values of Series.

        Notes
        -----
        This method is only implemented for datetime-like index classes,
        i.e., DatetimeIndex, PeriodIndex and TimedeltaIndex.

        Examples
        --------
        Put the first 5 month starts of 2011 into an index.

        >>> month_starts = pd.date_range('1/1/2011', periods=5, freq='MS')
        >>> month_starts
        DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',
                       '2011-05-01'],
                      dtype='datetime64[ns]', freq='MS')

        Shift the index by 10 days.

        >>> month_starts.shift(10, freq='D')
        DatetimeIndex(['2011-01-11', '2011-02-11', '2011-03-11', '2011-04-11',
                       '2011-05-11'],
                      dtype='datetime64[ns]', freq=None)

        The default value of `freq` is the `freq` attribute of the index,
        which is 'MS' (month start) in this example.

        >>> month_starts.shift(10)
        DatetimeIndex(['2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01',
                       '2012-03-01'],
                      dtype='datetime64[ns]', freq='MS')
        """
    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        *args
            Passed to `numpy.ndarray.argsort`.
        **kwargs
            Passed to `numpy.ndarray.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Integer indices that would sort the index if used as
            an indexer.

        See Also
        --------
        numpy.argsort : Similar method for NumPy arrays.
        Index.sort_values : Return sorted copy of Index.

        Examples
        --------
        >>> idx = pd.Index(['b', 'a', 'd', 'c'])
        >>> idx
        Index(['b', 'a', 'd', 'c'], dtype='object')

        >>> order = idx.argsort()
        >>> order
        array([1, 0, 3, 2])

        >>> idx[order]
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
    def _check_indexing_error(self, key): ...
    def get_indexer_non_unique(self, target) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : Index

        Returns
        -------
        indexer : np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.
        missing : np.ndarray[np.intp]
            An indexer into the target of the values not found.
            These correspond to the -1 in the indexer array.

        Examples
        --------
        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['b', 'b'])
        (array([1, 3, 4, 1, 3, 4]), array([], dtype=int64))

        In the example below there are no matched values.

        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['q', 'r', 't'])
        (array([-1, -1, -1]), array([0, 1, 2]))

        For this reason, the returned ``indexer`` contains only integers equal to -1.
        It demonstrates that there's no match between the index and the ``target``
        values at these positions. The mask [0, 1, 2] in the return value shows that
        the first, second, and third elements are missing.

        Notice that the return value is a tuple contains two items. In the example
        below the first item is an array of locations in ``index``. The second
        item is a mask shows that the first and third elements are missing.

        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['f', 'b', 's'])
        (array([-1,  1,  3,  4, -1]), array([0, 2]))
        """
    def get_indexer_for(self, target) -> npt.NDArray[np.intp]:
        """
        Guaranteed return of an indexer even when non-unique.

        This dispatches to get_indexer or get_indexer_non_unique
        as appropriate.

        Returns
        -------
        np.ndarray[np.intp]
            List of indices.

        Examples
        --------
        >>> idx = pd.Index([np.nan, 'var1', np.nan])
        >>> idx.get_indexer_for([np.nan])
        array([0, 2])
        """
    def _get_indexer_strict(self, key, axis_name: str_t) -> tuple[Index, np.ndarray]:
        """
        Analogue to get_indexer that raises if any elements are missing.
        """
    def _raise_if_missing(self, key, indexer, axis_name: str_t) -> None:
        """
        Check that indexer can be used to return a result.

        e.g. at least one element was found,
        unless the list of keys was actually empty.

        Parameters
        ----------
        key : list-like
            Targeted labels (only used to show correct error message).
        indexer: array-like of booleans
            Indices corresponding to the key,
            (with -1 indicating not found).
        axis_name : str

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.
        """
    def _get_indexer_non_comparable(self, target: Index, method, unique: bool = ...) -> npt.NDArray[np.intp] | tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Called from get_indexer or get_indexer_non_unique when the target
        is of a non-comparable dtype.

        For get_indexer lookups with method=None, get_indexer is an _equality_
        check, so non-comparable dtypes mean we will always have no matches.

        For get_indexer lookups with a method, get_indexer is an _inequality_
        check, so non-comparable dtypes mean we will always raise TypeError.

        Parameters
        ----------
        target : Index
        method : str or None
        unique : bool, default True
            * True if called from get_indexer.
            * False if called from get_indexer_non_unique.

        Raises
        ------
        TypeError
            If doing an inequality check, i.e. method is not None.
        """
    def _maybe_downcast_for_indexing(self, other: Index) -> tuple[Index, Index]:
        """
        When dealing with an object-dtype Index and a non-object Index, see
        if we can upcast the object-dtype one to improve performance.
        """
    def _find_common_type_compat(self, target) -> DtypeObj:
        """
        Implementation of find_common_type that adjusts for Index-specific
        special cases.
        """
    def _should_compare(self, other: Index) -> bool:
        """
        Check if `self == other` can ever have non-False entries.
        """
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
    def groupby(self, values) -> PrettyDict[Hashable, np.ndarray]:
        """
        Group the index labels by a given array of values.

        Parameters
        ----------
        values : array
            Values used to determine the groups.

        Returns
        -------
        dict
            {group name -> group labels}
        """
    def map(self, mapper, na_action: Literal['ignore'] | None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Union[Index, MultiIndex]
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map({1: 'a', 2: 'b', 3: 'c'})
        Index(['a', 'b', 'c'], dtype='object')

        Using `map` with a function:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map('I am a {}'.format)
        Index(['I am a 1', 'I am a 2', 'I am a 3'], dtype='object')

        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.map(lambda x: x.upper())
        Index(['A', 'B', 'C'], dtype='object')
        """
    def _transform_index(self, func, *, level) -> Index:
        """
        Apply function to all values found in index.

        This includes transforming multiindex entries separately.
        Only apply function to one level of the MultiIndex if level is specified.
        """
    def isin(self, values, level) -> npt.NDArray[np.bool_]:
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        np.ndarray[bool]
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1,2,3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> midx = pd.MultiIndex.from_arrays([[1,2,3],
        ...                                  ['red', 'blue', 'green']],
        ...                                  names=('number', 'color'))
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(['red', 'orange', 'yellow'], level='color')
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, 'red'), (3, 'red')])
        array([ True, False, False])
        """
    def _get_string_slice(self, key: str_t): ...
    def slice_indexer(self, start: Hashable | None, end: Hashable | None, step: int | None) -> slice:
        """
        Compute the slice indexer for input labels and step.

        Index needs to be ordered and unique.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, default None

        Returns
        -------
        slice

        Raises
        ------
        KeyError : If key does not exist, or key is not unique and index is
            not ordered.

        Notes
        -----
        This function assumes that the data is sorted, so use at your own peril

        Examples
        --------
        This is a method on all index types. For example you can do:

        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_indexer(start='b', end='c')
        slice(1, 3, None)

        >>> idx = pd.MultiIndex.from_arrays([list('abcd'), list('efgh')])
        >>> idx.slice_indexer(start='b', end=('c', 'g'))
        slice(1, 3, None)
        """
    def _maybe_cast_indexer(self, key):
        """
        If we have a float key and are not a floating index, then try to cast
        to an int if equivalent.
        """
    def _maybe_cast_listlike_indexer(self, target) -> Index:
        """
        Analogue to maybe_cast_indexer for get_indexer instead of get_loc.
        """
    def _validate_indexer(self, form: Literal['positional', 'slice'], key, kind: Literal['getitem', 'iloc']) -> None:
        """
        If we are positional indexer, validate that we have appropriate
        typed bounds must be an integer.
        """
    def _maybe_cast_slice_bound(self, label, side: str_t):
        """
        This function should be overloaded in subclasses that allow non-trivial
        casting on label-slice bounds, e.g. datetime-like indices allowing
        strings containing formatted datetimes.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
    def _searchsorted_monotonic(self, label, side: Literal['left', 'right'] = ...): ...
    def get_slice_bound(self, label, side: Literal['left', 'right']) -> int:
        """
        Calculate slice bound that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.

        See Also
        --------
        Index.get_loc : Get integer location, slice or boolean mask for requested
            label.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.get_slice_bound(3, 'left')
        3

        >>> idx.get_slice_bound(3, 'right')
        4

        If ``label`` is non-unique in the index, an error will be raised.

        >>> idx_duplicate = pd.Index(['a', 'b', 'a', 'c', 'd'])
        >>> idx_duplicate.get_slice_bound('a', 'left')
        Traceback (most recent call last):
        KeyError: Cannot get left slice bound for non-unique label: 'a'
        """
    def slice_locs(self, start, end, step) -> tuple[int, int]:
        """
        Compute slice locations for input labels.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, defaults None
            If None, defaults to 1.

        Returns
        -------
        tuple[int, int]

        See Also
        --------
        Index.get_loc : Get location for a single label.

        Notes
        -----
        This method only works if the index is monotonic or unique.

        Examples
        --------
        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_locs(start='b', end='c')
        (1, 3)
        """
    def delete(self, loc) -> Self:
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='object')

        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='object')
        """
    def insert(self, loc: int, item) -> Index:
        """
        Make new Index inserting new item at location.

        Follows Python numpy.insert semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.insert(1, 'x')
        Index(['a', 'x', 'b', 'c'], dtype='object')
        """
    def drop(self, labels: Index | np.ndarray | Iterable[Hashable], errors: IgnoreRaise = ...) -> Index:
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like or scalar
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        Raises
        ------
        KeyError
            If not all of the labels are found in the selected axis

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.drop(['a'])
        Index(['b', 'c'], dtype='object')
        """
    def infer_objects(self, copy: bool = ...) -> Index:
        """
        If we have an object dtype, try to infer a non-object dtype.

        Parameters
        ----------
        copy : bool, default True
            Whether to make a copy in cases where no inference occurs.
        """
    def diff(self, periods: int = ...) -> Index:
        """
        Computes the difference between consecutive values in the Index object.

        If periods is greater than 1, computes the difference between values that
        are `periods` number of positions apart.

        Parameters
        ----------
        periods : int, optional
            The number of positions between the current and previous
            value to compute the difference with. Default is 1.

        Returns
        -------
        Index
            A new Index object with the computed differences.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([10, 20, 30, 40, 50])
        >>> idx.diff()
        Index([nan, 10.0, 10.0, 10.0, 10.0], dtype='float64')

        """
    def round(self, decimals: int = ...) -> Self:
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.

        Returns
        -------
        Index
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([10.1234, 20.5678, 30.9123, 40.4567, 50.7890])
        >>> idx.round(decimals=2)
        Index([10.12, 20.57, 30.91, 40.46, 50.79], dtype='float64')

        """
    def _cmp_method(self, other, op):
        """
        Wrapper used to dispatch comparison operations.
        """
    def _logical_method(self, other, op): ...
    def _construct_result(self, result, name): ...
    def _arith_method(self, other, op): ...
    def _unary_method(self, op): ...
    def __abs__(self) -> Index: ...
    def __neg__(self) -> Index: ...
    def __pos__(self) -> Index: ...
    def __invert__(self) -> Index: ...
    def any(self, *args, **kwargs):
        """
        Return whether any element is Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.all : Return whether all elements are True.
        Series.all : Return whether all elements are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        >>> index = pd.Index([0, 1, 2])
        >>> index.any()
        True

        >>> index = pd.Index([0, 0, 0])
        >>> index.any()
        False
        """
    def all(self, *args, **kwargs):
        """
        Return whether all elements are Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.any : Return whether any element in an Index is True.
        Series.any : Return whether any element in a Series is True.
        Series.all : Return whether all elements in a Series are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        True, because nonzero integers are considered True.

        >>> pd.Index([1, 2, 3]).all()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 1, 2]).all()
        False
        """
    def _maybe_disable_logical_methods(self, opname: str_t) -> None:
        """
        raise if this Index subclass does not support any or all.
        """
    def argmin(self, axis, skipna: bool = ..., *args, **kwargs) -> int:
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
    def argmax(self, axis, skipna: bool = ..., *args, **kwargs) -> int:
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
    def min(self, axis, skipna: bool = ..., *args, **kwargs):
        """
        Return the minimum value of the Index.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.min()
        'a'

        For a MultiIndex, the minimum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.min()
        ('a', 1)
        """
    def max(self, axis, skipna: bool = ..., *args, **kwargs):
        """
        Return the maximum value of the Index.

        Parameters
        ----------
        axis : int, optional
            For compatibility with NumPy. Only 0 or None are allowed.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.max()
        ('b', 2)
        """
    @property
    def _engine_type(self): ...
    @property
    def _formatter_func(self): ...
    @property
    def nlevels(self): ...
    @property
    def is_monotonic_increasing(self): ...
    @property
    def is_monotonic_decreasing(self): ...
    @property
    def _is_strictly_monotonic_increasing(self): ...
    @property
    def _is_strictly_monotonic_decreasing(self): ...
    @property
    def has_duplicates(self): ...
    @property
    def values(self): ...
    @property
    def _values(self): ...
    @property
    def _index_as_unique(self): ...
    @property
    def shape(self): ...

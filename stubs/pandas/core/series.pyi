import lib as lib
import np
import npt
import pandas as pandas
import pandas._libs.lib
import pandas._libs.properties as properties
import pandas._libs.reshape as reshape
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algorithms
import pandas.core.arrays.arrow.accessors
import pandas.core.arrays.categorical
import pandas.core.arrays.sparse.accessor
import pandas.core.base
import pandas.core.base as base
import pandas.core.common as com
import pandas.core.generic
import pandas.core.indexes.accessors
import pandas.core.indexes.base as ibase
import pandas.core.methods.selectn as selectn
import pandas.core.missing as missing
import pandas.core.nanops as nanops
import pandas.core.ops as ops
import pandas.core.roperator as roperator
import pandas.core.strings.accessor
import pandas.io.formats.format as fmt
import pandas.plotting._core
from _typeshed import Incomplete
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexes.base import Index
from typing import Any, Callable, ClassVar, IO, Literal

__all__ = ['Series']

class Series(pandas.core.base.IndexOpsMixin, pandas.core.generic.NDFrame):
    _typ: ClassVar[str] = ...
    _HANDLED_TYPES: ClassVar[tuple] = ...
    _metadata: ClassVar[list] = ...
    _internal_names_set: ClassVar[set] = ...
    _accessors: ClassVar[set] = ...
    _hidden_attrs: ClassVar[frozenset] = ...
    __pandas_priority__: ClassVar[int] = ...
    _agg_see_also_doc: ClassVar[str] = ...
    _agg_examples_doc: ClassVar[str] = ...
    _AXIS_ORDERS: ClassVar[list] = ...
    _AXIS_LEN: ClassVar[int] = ...
    _info_axis_number: ClassVar[int] = ...
    _info_axis_name: ClassVar[str] = ...
    str: ClassVar[type[pandas.core.strings.accessor.StringMethods]] = ...
    dt: ClassVar[type[pandas.core.indexes.accessors.CombinedDatetimelikeProperties]] = ...
    cat: ClassVar[type[pandas.core.arrays.categorical.CategoricalAccessor]] = ...
    plot: ClassVar[type[pandas.plotting._core.PlotAccessor]] = ...
    sparse: ClassVar[type[pandas.core.arrays.sparse.accessor.SparseAccessor]] = ...
    struct: ClassVar[type[pandas.core.arrays.arrow.accessors.StructAccessor]] = ...
    list: ClassVar[type[pandas.core.arrays.arrow.accessors.ListAccessor]] = ...
    name: Incomplete
    index: Incomplete
    def __init__(self, data, index, dtype: Dtype | None, name, copy: bool | None, fastpath: bool | lib.NoDefault = ...) -> None: ...
    def _init_dict(self, data: Mapping, index: Index | None, dtype: DtypeObj | None):
        '''
        Derive the "_mgr" and "index" attributes of a new Series from a
        dictionary input.

        Parameters
        ----------
        data : dict or dict-like
            Data used to populate the new Series.
        index : Index or None, default None
            Index for the new Series: if None, use dict keys.
        dtype : np.dtype, ExtensionDtype, or None, default None
            The dtype for the new Series: if None, infer from data.

        Returns
        -------
        _data : BlockManager for the new Series
        index : index for the new Series
        '''
    def _constructor_from_mgr(self, mgr, axes): ...
    def _constructor_expanddim_from_mgr(self, mgr, axes): ...
    def ravel(self, order: str = ...) -> ArrayLike:
        """
        Return the flattened underlying data as an ndarray or ExtensionArray.

        .. deprecated:: 2.2.0
            Series.ravel is deprecated. The underlying array is already 1D, so
            ravel is not necessary.  Use :meth:`to_numpy` for conversion to a numpy
            array instead.

        Returns
        -------
        numpy.ndarray or ExtensionArray
            Flattened data of the Series.

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.ravel()  # doctest: +SKIP
        array([1, 2, 3])
        """
    def __len__(self) -> int:
        """
        Return the length of the Series.
        """
    def view(self, dtype: Dtype | None) -> Series:
        """
        Create a new view of the Series.

        .. deprecated:: 2.2.0
            ``Series.view`` is deprecated and will be removed in a future version.
            Use :meth:`Series.astype` as an alternative to change the dtype.

        This function will return a new Series with a view of the same
        underlying values in memory, optionally reinterpreted with a new data
        type. The new data type must preserve the same size in bytes as to not
        cause index misalignment.

        Parameters
        ----------
        dtype : data type
            Data type object or one of their string representations.

        Returns
        -------
        Series
            A new Series object as a view of the same data in memory.

        See Also
        --------
        numpy.ndarray.view : Equivalent numpy function to create a new view of
            the same data in memory.

        Notes
        -----
        Series are instantiated with ``dtype=float64`` by default. While
        ``numpy.ndarray.view()`` will return a view with the same data type as
        the original array, ``Series.view()`` (without specified dtype)
        will try using ``float64`` and may fail if the original data type size
        in bytes is not the same.

        Examples
        --------
        Use ``astype`` to change the dtype instead.
        """
    def __array__(self, dtype: npt.DTypeLike | None, copy: bool | None) -> np.ndarray:
        '''
        Return the values as a NumPy array.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool or None, optional
            Unused.

        Returns
        -------
        numpy.ndarray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        See Also
        --------
        array : Create a new array from data.
        Series.array : Zero-copy view to the array backing the Series.
        Series.to_numpy : Series method for similar behavior.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> np.asarray(ser)
        array([1, 2, 3])

        For timezone-aware data, the timezones may be retained with
        ``dtype=\'object\'``

        >>> tzser = pd.Series(pd.date_range(\'2000\', periods=2, tz="CET"))
        >>> np.asarray(tzser, dtype="object")
        array([Timestamp(\'2000-01-01 00:00:00+0100\', tz=\'CET\'),
               Timestamp(\'2000-01-02 00:00:00+0100\', tz=\'CET\')],
              dtype=object)

        Or the values may be localized to UTC and the tzinfo discarded with
        ``dtype=\'datetime64[ns]\'``

        >>> np.asarray(tzser, dtype="datetime64[ns]")  # doctest: +ELLIPSIS
        array([\'1999-12-31T23:00:00.000000000\', ...],
              dtype=\'datetime64[ns]\')
        '''
    def __column_consortium_standard__(self, *, api_version: str | None) -> Any:
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of pandas.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def _ixs(self, i: int, axis: AxisInt = ...) -> Any:
        """
        Return the i-th value or values in the Series by location.

        Parameters
        ----------
        i : int

        Returns
        -------
        scalar
        """
    def _slice(self, slobj: slice, axis: AxisInt = ...) -> Series: ...
    def __getitem__(self, key): ...
    def _get_with(self, key): ...
    def _get_values_tuple(self, key: tuple): ...
    def _get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Series: ...
    def _get_value(self, label, takeable: bool = ...):
        """
        Quickly retrieve single value at passed index label.

        Parameters
        ----------
        label : object
        takeable : interpret the index as indexers, default False

        Returns
        -------
        scalar value
        """
    def __setitem__(self, key, value) -> None: ...
    def _set_with_engine(self, key, value, warn: bool = ...) -> None: ...
    def _set_with(self, key, value, warn: bool = ...) -> None: ...
    def _set_labels(self, key, value, warn: bool = ...) -> None: ...
    def _set_values(self, key, value, warn: bool = ...) -> None: ...
    def _set_value(self, label, value, takeable: bool = ...) -> None:
        """
        Quickly set single value at passed label.

        If label is not contained, a new object is created with the label
        placed at the end of the result index.

        Parameters
        ----------
        label : object
            Partial indexing with MultiIndex not allowed.
        value : object
            Scalar value.
        takeable : interpret the index as indexers, default False
        """
    def _get_cacher(self):
        """return my cacher or None"""
    def _reset_cacher(self) -> None:
        """
        Reset the cacher.
        """
    def _set_as_cached(self, item, cacher) -> None:
        """
        Set the _cacher attribute on the calling object with a weakref to
        cacher.
        """
    def _clear_item_cache(self) -> None: ...
    def _check_is_chained_assignment_possible(self) -> bool:
        """
        See NDFrame._check_is_chained_assignment_possible.__doc__
        """
    def _maybe_update_cacher(self, clear: bool = ..., verify_is_copy: bool = ..., inplace: bool = ...) -> None:
        """
        See NDFrame._maybe_update_cacher.__doc__
        """
    def repeat(self, repeats: int | Sequence[int], axis: None) -> Series:
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.
        axis : None
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.repeat(2)
        0    a
        0    a
        1    b
        1    b
        2    c
        2    c
        dtype: object
        >>> s.repeat([1, 2, 3])
        0    a
        1    b
        1    b
        2    c
        2    c
        2    c
        dtype: object
        """
    def reset_index(self, level: IndexLabel | None, *, drop: bool = ..., name: Level = ..., inplace: bool = ..., allow_duplicates: bool = ...) -> DataFrame | Series | None:
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column, or
        when the index is meaningless and needs to be reset to the default
        before another operation.

        Parameters
        ----------
        level : int, str, tuple, or list, default optional
            For a Series with a MultiIndex, only remove the specified levels
            from the index. Removes all levels by default.
        drop : bool, default False
            Just reset the index, without inserting it as a column in
            the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series
            values. Uses ``self.name`` by default. This argument is ignored
            when `drop` is True.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).
        allow_duplicates : bool, default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame or None
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        See Also
        --------
        DataFrame.reset_index: Analogous function for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4], name='foo',
        ...               index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        Generate a DataFrame with default index.

        >>> s.reset_index()
          idx  foo
        0   a    1
        1   b    2
        2   c    3
        3   d    4

        To specify the name of the new column use `name`.

        >>> s.reset_index(name='values')
          idx  values
        0   a       1
        1   b       2
        2   c       3
        3   d       4

        To generate a new Series with the default set `drop` to True.

        >>> s.reset_index(drop=True)
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64

        The `level` parameter is interesting for Series with a multi-level
        index.

        >>> arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
        ...           np.array(['one', 'two', 'one', 'two'])]
        >>> s2 = pd.Series(
        ...     range(4), name='foo',
        ...     index=pd.MultiIndex.from_arrays(arrays,
        ...                                     names=['a', 'b']))

        To remove a specific level from the Index, use `level`.

        >>> s2.reset_index(level='a')
               a  foo
        b
        one  bar    0
        two  bar    1
        one  baz    2
        two  baz    3

        If `level` is not set, all levels are removed from the Index.

        >>> s2.reset_index()
             a    b  foo
        0  bar  one    0
        1  bar  two    1
        2  baz  one    2
        3  baz  two    3
        """
    def to_string(self, buf: FilePath | WriteBuffer[str] | None, na_rep: str = ..., float_format: str | None, header: bool = ..., index: bool = ..., length: bool = ..., dtype: bool = ..., name: bool = ..., max_rows: int | None, min_rows: int | None) -> str | None:
        """
        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, optional
            String representation of NaN to use, default 'NaN'.
        float_format : one-parameter function, optional
            Formatter function to apply to columns' elements if they are
            floats, default None.
        header : bool, default True
            Add the Series header (index name).
        index : bool, optional
            Add index (row) labels, default True.
        length : bool, default False
            Add the Series length.
        dtype : bool, default False
            Add the Series dtype.
        name : bool, default False
            Add the Series name if not None.
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.
        min_rows : int, optional
            The number of rows to display in a truncated repr (when number
            of rows is above `max_rows`).

        Returns
        -------
        str or None
            String representation of Series if ``buf=None``, otherwise None.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3]).to_string()
        >>> ser
        '0    1\\n1    2\\n2    3'
        """
    def to_markdown(self, buf: IO[str] | None, mode: str = ..., index: bool = ..., storage_options: StorageOptions | None, **kwargs) -> str | None:
        '''
        Print Series in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        **kwargs
            These parameters will be passed to `tabulate                 <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        str
            Series in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        Examples
                    --------
                    >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")
                    >>> print(s.to_markdown())
                    |    | animal   |
                    |---:|:---------|
                    |  0 | elk      |
                    |  1 | pig      |
                    |  2 | dog      |
                    |  3 | quetzal  |

                    Output markdown with a tabulate option.

                    >>> print(s.to_markdown(tablefmt="grid"))
                    +----+----------+
                    |    | animal   |
                    +====+==========+
                    |  0 | elk      |
                    +----+----------+
                    |  1 | pig      |
                    +----+----------+
                    |  2 | dog      |
                    +----+----------+
                    |  3 | quetzal  |
                    +----+----------+
        '''
    def items(self) -> Iterable[tuple[Hashable, Any]]:
        '''
        Lazily iterate over (index, value) tuples.

        This method returns an iterable tuple (index, value). This is
        convenient if you want to create a lazy iterator.

        Returns
        -------
        iterable
            Iterable of tuples containing the (index, value) pairs from a
            Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> s = pd.Series([\'A\', \'B\', \'C\'])
        >>> for index, value in s.items():
        ...     print(f"Index : {index}, Value : {value}")
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        '''
    def keys(self) -> Index:
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s.keys()
        Index([0, 1, 2], dtype='int64')
        """
    def to_dict(self, *, into: type[MutableMappingT] | MutableMappingT = ...) -> MutableMappingT:
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.MutableMapping subclass to use as the return
            object. Can be the actual class or an empty instance of the mapping
            type you want.  If you want a collections.defaultdict, you must
            pass it initialized.

        Returns
        -------
        collections.abc.MutableMapping
            Key-value representation of Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(into=OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(into=dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
    def to_frame(self, name: Hashable = ...) -> DataFrame:
        '''
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, optional
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"],
        ...               name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        '''
    def _set_name(self, name, inplace: bool = ..., deep: bool | None) -> Series:
        """
        Set the Series name.

        Parameters
        ----------
        name : str
        inplace : bool
            Whether to modify `self` directly or return a copy.
        deep : bool|None, default None
            Whether to do a deep copy, a shallow copy, or Copy on Write(None)
        """
    def groupby(self, by, axis: Axis = ..., level: IndexLabel | None, as_index: bool = ..., sort: bool = ..., group_keys: bool = ..., observed: bool | lib.NoDefault = ..., dropna: bool = ...) -> SeriesGroupBy:
        '''
        Group Series using a mapper or by a Series of columns.

        A groupby operation involves some combination of splitting the
        object, applying a function, and combining the results. This can be
        used to group large amounts of data and compute operations on these
        groups.

        Parameters
        ----------
        by : mapping, function, label, pd.Grouper or list of such
            Used to determine the groups for the groupby.
            If ``by`` is a function, it\'s called on each value of the object\'s
            index. If a dict or Series is passed, the Series or dict VALUES
            will be used to determine the groups (the Series\' values are first
            aligned; see ``.align()`` method). If a list or ndarray of length
            equal to the selected axis is passed (see the `groupby user guide
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#splitting-an-object-into-groups>`_),
            the values are used as-is to determine the groups. A label or list
            of labels may be passed to group by the columns in ``self``.
            Notice that a tuple is interpreted as a (single) key.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            Split along rows (0) or columns (1). For `Series` this parameter
            is unused and defaults to 0.

            .. deprecated:: 2.1.0

                Will be removed and behave like axis=0 in a future version.
                For ``axis=1``, do ``frame.T.groupby(...)`` instead.

        level : int, level name, or sequence of such, default None
            If the axis is a MultiIndex (hierarchical), group by a particular
            level or levels. Do not specify both ``by`` and ``level``.
        as_index : bool, default True
            Return object with group labels as the
            index. Only relevant for DataFrame input. as_index=False is
            effectively "SQL-style" grouped output. This argument has no effect
            on filtrations (see the `filtrations in the user guide
            <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#filtration>`_),
            such as ``head()``, ``tail()``, ``nth()`` and in transformations
            (see the `transformations in the user guide
            <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#transformation>`_).
        sort : bool, default True
            Sort group keys. Get better performance by turning this off.
            Note this does not influence the order of observations within each
            group. Groupby preserves the order of rows within each group. If False,
            the groups will appear in the same order as they did in the original DataFrame.
            This argument has no effect on filtrations (see the `filtrations in the user guide
            <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#filtration>`_),
            such as ``head()``, ``tail()``, ``nth()`` and in transformations
            (see the `transformations in the user guide
            <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#transformation>`_).

            .. versionchanged:: 2.0.0

                Specifying ``sort=False`` with an ordered categorical grouper will no
                longer sort the values.

        group_keys : bool, default True
            When calling apply and the ``by`` argument produces a like-indexed
            (i.e. :ref:`a transform <groupby.transform>`) result, add group keys to
            index to identify pieces. By default group keys are not included
            when the result\'s index (and column) labels match the inputs, and
            are included otherwise.

            .. versionchanged:: 1.5.0

               Warns that ``group_keys`` will no longer be ignored when the
               result from ``apply`` is a like-indexed Series or DataFrame.
               Specify ``group_keys`` explicitly to include the group keys or
               not.

            .. versionchanged:: 2.0.0

               ``group_keys`` now defaults to ``True``.

        observed : bool, default False
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.

            .. deprecated:: 2.1.0

                The default value will change to True in a future version of pandas.

        dropna : bool, default True
            If True, and if group keys contain NA values, NA values together
            with row/column will be dropped.
            If False, NA values will also be treated as the key in groups.

        Returns
        -------
        pandas.api.typing.SeriesGroupBy
            Returns a groupby object that contains information about the groups.

        See Also
        --------
        resample : Convenience method for frequency conversion and resampling
            of time series.

        Notes
        -----
        See the `user guide
        <https://pandas.pydata.org/pandas-docs/stable/groupby.html>`__ for more
        detailed usage and examples, including splitting an object into groups,
        iterating through groups, selecting a group, aggregation, and more.

        Examples
        --------
        >>> ser = pd.Series([390., 350., 30., 20.],
        ...                 index=[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],
        ...                 name="Max Speed")
        >>> ser
        Falcon    390.0
        Falcon    350.0
        Parrot     30.0
        Parrot     20.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(["a", "b", "a", "b"]).mean()
        a    210.0
        b    185.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).mean()
        Falcon    370.0
        Parrot     25.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(ser > 100).mean()
        Max Speed
        False     25.0
        True     370.0
        Name: Max Speed, dtype: float64

        **Grouping by Indexes**

        We can groupby different levels of a hierarchical index
        using the `level` parameter:

        >>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],
        ...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))
        >>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")
        >>> ser
        Animal  Type
        Falcon  Captive    390.0
                Wild       350.0
        Parrot  Captive     30.0
                Wild        20.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).mean()
        Animal
        Falcon    370.0
        Parrot     25.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level="Type").mean()
        Type
        Captive    210.0
        Wild       185.0
        Name: Max Speed, dtype: float64

        We can also choose to include `NA` in group keys or not by defining
        `dropna` parameter, the default setting is `True`.

        >>> ser = pd.Series([1, 2, 3, 3], index=["a", \'a\', \'b\', np.nan])
        >>> ser.groupby(level=0).sum()
        a    3
        b    3
        dtype: int64

        >>> ser.groupby(level=0, dropna=False).sum()
        a    3
        b    3
        NaN  3
        dtype: int64

        >>> arrays = [\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\']
        >>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")
        >>> ser.groupby(["a", "b", "a", np.nan]).mean()
        a    210.0
        b    350.0
        Name: Max Speed, dtype: float64

        >>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()
        a    210.0
        b    350.0
        NaN   20.0
        Name: Max Speed, dtype: float64
        '''
    def count(self) -> int:
        """
        Return number of non-NA/null observations in the Series.

        Returns
        -------
        int
            Number of non-null values in the Series.

        See Also
        --------
        DataFrame.count : Count non-NA cells for each column or row.

        Examples
        --------
        >>> s = pd.Series([0.0, 1.0, np.nan])
        >>> s.count()
        2
        """
    def mode(self, dropna: bool = ...) -> Series:
        """
        Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        Examples
        --------
        >>> s = pd.Series([2, 4, 2, 2, 4, None])
        >>> s.mode()
        0    2.0
        dtype: float64

        More than one mode:

        >>> s = pd.Series([2, 4, 8, 2, 4, None])
        >>> s.mode()
        0    2.0
        1    4.0
        dtype: float64

        With and without considering null value:

        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode(dropna=False)
        0   NaN
        dtype: float64
        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode()
        0    4.0
        dtype: float64
        """
    def unique(self) -> ArrayLike:
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array. See Notes.

        See Also
        --------
        Series.drop_duplicates : Return Series with duplicate values removed.
        unique : Top-level unique method for any 1-d array-like object.
        Index.unique : Return Index with unique values from an Index object.

        Notes
        -----
        Returns the unique values as a NumPy array. In case of an
        extension-array backed Series, a new
        :class:`~api.extensions.ExtensionArray` of that type with just
        the unique values is returned. This includes

            * Categorical
            * Period
            * Datetime with Timezone
            * Datetime without Timezone
            * Timedelta
            * Interval
            * Sparse
            * IntegerNA

        See Examples section.

        Examples
        --------
        >>> pd.Series([2, 1, 3, 3], name='A').unique()
        array([2, 1, 3])

        >>> pd.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00']
        Length: 1, dtype: datetime64[ns]

        >>> pd.Series([pd.Timestamp('2016-01-01', tz='US/Eastern')
        ...            for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00-05:00']
        Length: 1, dtype: datetime64[ns, US/Eastern]

        An Categorical will return categories in the order of
        appearance and with the same dtype.

        >>> pd.Series(pd.Categorical(list('baabc'))).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Series(pd.Categorical(list('baabc'), categories=list('abc'),
        ...                          ordered=True)).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        """
    def drop_duplicates(self, *, keep: DropKeep = ..., inplace: bool = ..., ignore_index: bool = ...) -> Series | None:
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with duplicates dropped or None if ``inplace=True``.

        See Also
        --------
        Index.drop_duplicates : Equivalent method on Index.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Series.duplicated : Related method on Series, indicating duplicate
            Series values.
        Series.unique : Return unique values as an array.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama', 'hippo'],
        ...               name='animal')
        >>> s
        0     llama
        1       cow
        2     llama
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behaviour of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates()
        0     llama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep='last')
        1       cow
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries.

        >>> s.drop_duplicates(keep=False)
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
    def duplicated(self, keep: DropKeep = ...) -> Series:
        """
        Indicate duplicate Series values.

        Duplicated values are indicated as ``True`` values in the resulting
        Series. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series[bool]
            Series indicating whether each value has occurred in the
            preceding values.

        See Also
        --------
        Index.duplicated : Equivalent method on pandas.Index.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Series.drop_duplicates : Remove duplicate values from Series.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set on False and all others on True:

        >>> animals = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama'])
        >>> animals.duplicated()
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        which is equivalent to

        >>> animals.duplicated(keep='first')
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> animals.duplicated(keep='last')
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        By setting keep on ``False``, all duplicates are True:

        >>> animals.duplicated(keep=False)
        0     True
        1    False
        2     True
        3    False
        4     True
        dtype: bool
        """
    def idxmin(self, axis: Axis = ..., skipna: bool = ..., *args, **kwargs) -> Hashable:
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmin : Return indices of the minimum values
            along the given axis.
        DataFrame.idxmin : Return index of first occurrence of minimum
            over requested axis.
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 1],
        ...               index=['A', 'B', 'C', 'D'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    1.0
        dtype: float64

        >>> s.idxmin()
        'A'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmin(skipna=False)
        nan
        """
    def idxmax(self, axis: Axis = ..., skipna: bool = ..., *args, **kwargs) -> Hashable:
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmax : Return indices of the maximum values
            along the given axis.
        DataFrame.idxmax : Return index of first occurrence of maximum
            over requested axis.
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmax``. This method
        returns the label of the maximum, while ``ndarray.argmax`` returns
        the position. To get the position, use ``series.values.argmax()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 3, 4],
        ...               index=['A', 'B', 'C', 'D', 'E'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    4.0
        dtype: float64

        >>> s.idxmax()
        'C'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmax(skipna=False)
        nan
        """
    def round(self, decimals: int = ..., *args, **kwargs) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Series
            Rounded values of the Series.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0.1, 1.3, 2.7])
        >>> s.round()
        0    0.0
        1    1.0
        2    3.0
        dtype: float64
        """
    def quantile(self, q: float | Sequence[float] | AnyArrayLike = ..., interpolation: QuantileInterpolation = ...) -> float | Series:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * (x-i)/(j-i)`, where `(x-i)/(j-i)` is
                  the fractional part of the index surrounded by `i > j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        float or Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        See Also
        --------
        core.window.Rolling.quantile : Calculate the rolling quantile.
        numpy.percentile : Returns the q-th percentile(s) of the array elements.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.quantile(.5)
        2.5
        >>> s.quantile([.25, .5, .75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """
    def corr(self, other: Series, method: CorrelationMethod = ..., min_periods: int | None) -> float:
        """
        Compute correlation with `other` Series, excluding missing values.

        The two `Series` objects are not required to be the same length and will be
        aligned internally before the correlation function is applied.

        Parameters
        ----------
        other : Series
            Series with which to compute the correlation.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - kendall : Kendall Tau correlation coefficient
            - spearman : Spearman rank correlation
            - callable: Callable with input two 1d ndarrays and returning a float.

            .. warning::
                Note that the returned matrix from corr will have 1 along the
                diagonals and will be symmetric regardless of the callable's
                behavior.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.

        Returns
        -------
        float
            Correlation with other.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation between columns.
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.

        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.

        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

        Automatic data alignment: as with all pandas operations, automatic data alignment is performed for this method.
        ``corr()`` automatically considers values with matching indices.

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> s1 = pd.Series([.2, .0, .6, .2])
        >>> s2 = pd.Series([.3, .6, .0, .1])
        >>> s1.corr(s2, method=histogram_intersection)
        0.3

        Pandas auto-aligns the values with matching indices

        >>> s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s2 = pd.Series([1, 2, 3], index=[2, 1, 0])
        >>> s1.corr(s2)
        -1.0
        """
    def cov(self, other: Series, min_periods: int | None, ddof: int | None = ...) -> float:
        """
        Compute covariance with Series, excluding missing values.

        The two `Series` objects are not required to be the same length and
        will be aligned internally before the covariance is calculated.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.
        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        See Also
        --------
        DataFrame.cov : Compute pairwise covariance of columns.

        Examples
        --------
        >>> s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
        >>> s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
        >>> s1.cov(s2)
        -0.01685762652715874
        """
    def diff(self, periods: int = ...) -> Series:
        """
        First discrete difference of element.

        Calculates the difference of a Series element compared with another
        element in the Series (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative
            values.

        Returns
        -------
        Series
            First differences of the Series.

        See Also
        --------
        Series.pct_change: Percent change over given number of periods.
        Series.shift: Shift index by desired number of periods with an
            optional time freq.
        DataFrame.diff: First discrete difference of object.

        Notes
        -----
        For boolean dtypes, this uses :meth:`operator.xor` rather than
        :meth:`operator.sub`.
        The result is calculated according to current dtype in Series,
        however dtype of the result is always float64.

        Examples
        --------

        Difference with previous row

        >>> s = pd.Series([1, 1, 2, 3, 5, 8])
        >>> s.diff()
        0    NaN
        1    0.0
        2    1.0
        3    1.0
        4    2.0
        5    3.0
        dtype: float64

        Difference with 3rd previous row

        >>> s.diff(periods=3)
        0    NaN
        1    NaN
        2    NaN
        3    2.0
        4    4.0
        5    6.0
        dtype: float64

        Difference with following row

        >>> s.diff(periods=-1)
        0    0.0
        1   -1.0
        2   -1.0
        3   -2.0
        4   -3.0
        5    NaN
        dtype: float64

        Overflow in input dtype

        >>> s = pd.Series([1, 0], dtype=np.uint8)
        >>> s.diff()
        0      NaN
        1    255.0
        dtype: float64
        """
    def autocorr(self, lag: int = ...) -> float:
        """
        Compute the lag-N autocorrelation.

        This method computes the Pearson correlation between
        the Series and its shifted self.

        Parameters
        ----------
        lag : int, default 1
            Number of lags to apply before performing autocorrelation.

        Returns
        -------
        float
            The Pearson correlation between self and self.shift(lag).

        See Also
        --------
        Series.corr : Compute the correlation between two Series.
        Series.shift : Shift index by desired number of periods.
        DataFrame.corr : Compute pairwise correlation of columns.
        DataFrame.corrwith : Compute pairwise correlation between rows or
            columns of two DataFrame objects.

        Notes
        -----
        If the Pearson correlation is not well defined return 'NaN'.

        Examples
        --------
        >>> s = pd.Series([0.25, 0.5, 0.2, -0.05])
        >>> s.autocorr()  # doctest: +ELLIPSIS
        0.10355...
        >>> s.autocorr(lag=2)  # doctest: +ELLIPSIS
        -0.99999...

        If the Pearson correlation is not well defined, then 'NaN' is returned.

        >>> s = pd.Series([1, 0, 0, 0])
        >>> s.autocorr()
        nan
        """
    def dot(self, other: AnyArrayLike) -> Series | np.ndarray:
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame, or the Series and
        each columns of an array.

        It can also be called using `self @ other`.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series or numpy.ndarray
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame or a numpy.ndarray between the Series
            and each columns of the numpy array.

        See Also
        --------
        DataFrame.dot: Compute the matrix product with the DataFrame.
        Series.mul: Multiplication of series and other, element-wise.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> other = pd.Series([-1, 2, -3, 4])
        >>> s.dot(other)
        8
        >>> s @ other
        8
        >>> df = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(df)
        0    24
        1    14
        dtype: int64
        >>> arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(arr)
        array([24, 14])
        """
    def __matmul__(self, other):
        """
        Matrix multiplication using binary `@` operator.
        """
    def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator.
        """
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted Series `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        .. note::

            The Series *must* be monotonically sorted, otherwise
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
    def _append(self, to_append, ignore_index: bool = ..., verify_integrity: bool = ...): ...
    def compare(self, other: Series, align_axis: Axis = ..., keep_shape: bool = ..., keep_equal: bool = ..., result_names: Suffixes = ...) -> DataFrame | Series:
        '''
        Compare to another Series and show the differences.

        Parameters
        ----------
        other : Series
            Object to compare with.

        align_axis : {0 or \'index\', 1 or \'columns\'}, default 1
            Determine which axis to align the comparison on.

            * 0, or \'index\' : Resulting differences are stacked vertically
                with rows drawn alternately from self and other.
            * 1, or \'columns\' : Resulting differences are aligned horizontally
                with columns drawn alternately from self and other.

        keep_shape : bool, default False
            If true, all rows and columns are kept.
            Otherwise, only the ones with different values are kept.

        keep_equal : bool, default False
            If true, the result keeps values that are equal.
            Otherwise, equal values are shown as NaNs.

        result_names : tuple, default (\'self\', \'other\')
            Set the dataframes names in the comparison.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            If axis is 0 or \'index\' the result will be a Series.
            The resulting index will be a MultiIndex with \'self\' and \'other\'
            stacked alternately at the inner level.

            If axis is 1 or \'columns\' the result will be a DataFrame.
            It will have two columns namely \'self\' and \'other\'.

        See Also
        --------
        DataFrame.compare : Compare with another DataFrame and show differences.

        Notes
        -----
        Matching NaNs will not appear as a difference.

        Examples
        --------
        >>> s1 = pd.Series(["a", "b", "c", "d", "e"])
        >>> s2 = pd.Series(["a", "a", "c", "b", "e"])

        Align the differences on columns

        >>> s1.compare(s2)
          self other
        1    b     a
        3    d     b

        Stack the differences on indices

        >>> s1.compare(s2, align_axis=0)
        1  self     b
           other    a
        3  self     d
           other    b
        dtype: object

        Keep all original rows

        >>> s1.compare(s2, keep_shape=True)
          self other
        0  NaN   NaN
        1    b     a
        2  NaN   NaN
        3    d     b
        4  NaN   NaN

        Keep all original rows and also all original values

        >>> s1.compare(s2, keep_shape=True, keep_equal=True)
          self other
        0    a     a
        1    b     a
        2    c     c
        3    d     b
        4    e     e
        '''
    def combine(self, other: Series | Hashable, func: Callable[[Hashable, Hashable], Hashable], fill_value: Hashable | None) -> Series:
        """
        Combine the Series with a Series or scalar according to `func`.

        Combine the Series and `other` using `func` to perform elementwise
        selection for combined Series.
        `fill_value` is assumed when value is missing at some index
        from one of the two objects being combined.

        Parameters
        ----------
        other : Series or scalar
            The value(s) to be combined with the `Series`.
        func : function
            Function that takes two scalars as inputs and returns an element.
        fill_value : scalar, optional
            The value to assume when an index is missing from
            one Series or the other. The default specifies to use the
            appropriate NaN value for the underlying dtype of the Series.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine_first : Combine Series values, choosing the calling
            Series' values first.

        Examples
        --------
        Consider 2 Datasets ``s1`` and ``s2`` containing
        highest clocked speeds of different birds.

        >>> s1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
        >>> s1
        falcon    330.0
        eagle     160.0
        dtype: float64
        >>> s2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        >>> s2
        falcon    345.0
        eagle     200.0
        duck       30.0
        dtype: float64

        Now, to combine the two datasets and view the highest speeds
        of the birds across the two datasets

        >>> s1.combine(s2, max)
        duck        NaN
        eagle     200.0
        falcon    345.0
        dtype: float64

        In the previous example, the resulting value for duck is missing,
        because the maximum of a NaN and a float is a NaN.
        So, in the example, we set ``fill_value=0``,
        so the maximum value returned will be the value from some dataset.

        >>> s1.combine(s2, max, fill_value=0)
        duck       30.0
        eagle     200.0
        falcon    345.0
        dtype: float64
        """
    def combine_first(self, other) -> Series:
        """
        Update null elements with value in the same location in 'other'.

        Combine two Series objects by filling null values in one Series with
        non-null values from the other Series. Result index will be the union
        of the two indexes.

        Parameters
        ----------
        other : Series
            The value(s) to be used for filling null values.

        Returns
        -------
        Series
            The result of combining the provided Series with the other object.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.

        Examples
        --------
        >>> s1 = pd.Series([1, np.nan])
        >>> s2 = pd.Series([3, 4, 5])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        2    5.0
        dtype: float64

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> s1 = pd.Series({'falcon': np.nan, 'eagle': 160.0})
        >>> s2 = pd.Series({'eagle': 200.0, 'duck': 30.0})
        >>> s1.combine_first(s2)
        duck       30.0
        eagle     160.0
        falcon      NaN
        dtype: float64
        """
    def update(self, other: Series | Sequence | Mapping) -> None:
        """
        Modify Series in place using values from passed Series.

        Uses non-NA values from passed Series to make updates. Aligns
        on index.

        Parameters
        ----------
        other : Series, or object coercible into Series

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s.update(pd.Series(['d', 'e'], index=[0, 2]))
        >>> s
        0    d
        1    b
        2    e
        dtype: object

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6, 7, 8]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, np.nan, 6]))
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        ``other`` can also be a non-Series object type
        that is coercible into a Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.update([4, np.nan, 6])
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        >>> s = pd.Series([1, 2, 3])
        >>> s.update({1: 9})
        >>> s
        0    1
        1    9
        2    3
        dtype: int64
        """
    def sort_values(self, *, axis: Axis = ..., ascending: bool | Sequence[bool] = ..., inplace: bool = ..., kind: SortKind = ..., na_position: NaPosition = ..., ignore_index: bool = ..., key: ValueKeyFunc | None) -> Series | None:
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some
        criterion.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        ascending : bool or list of bools, default True
            If True, sort values in ascending order, otherwise descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable  algorithms.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            If not None, apply the key function to the series values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return an array-like.

        Returns
        -------
        Series or None
            Series ordered by values or None if ``inplace=True``.

        See Also
        --------
        Series.sort_index : Sort by the Series indices.
        DataFrame.sort_values : Sort DataFrame by the values along either axis.
        DataFrame.sort_index : Sort DataFrame by indices.

        Examples
        --------
        >>> s = pd.Series([np.nan, 1, 3, 10, 5])
        >>> s
        0     NaN
        1     1.0
        2     3.0
        3     10.0
        4     5.0
        dtype: float64

        Sort values ascending order (default behaviour)

        >>> s.sort_values(ascending=True)
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        0     NaN
        dtype: float64

        Sort values descending order

        >>> s.sort_values(ascending=False)
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        dtype: float64

        Sort values putting NAs first

        >>> s.sort_values(na_position='first')
        0     NaN
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        dtype: float64

        Sort a series of strings

        >>> s = pd.Series(['z', 'b', 'd', 'a', 'c'])
        >>> s
        0    z
        1    b
        2    d
        3    a
        4    c
        dtype: object

        >>> s.sort_values()
        3    a
        1    b
        4    c
        2    d
        0    z
        dtype: object

        Sort using a key function. Your `key` function will be
        given the ``Series`` of values and should return an array-like.

        >>> s = pd.Series(['a', 'B', 'c', 'D', 'e'])
        >>> s.sort_values()
        1    B
        3    D
        0    a
        2    c
        4    e
        dtype: object
        >>> s.sort_values(key=lambda x: x.str.lower())
        0    a
        1    B
        2    c
        3    D
        4    e
        dtype: object

        NumPy ufuncs work well here. For example, we can
        sort by the ``sin`` of the value

        >>> s = pd.Series([-4, -2, 0, 2, 4])
        >>> s.sort_values(key=np.sin)
        1   -2
        4    4
        2    0
        0   -4
        3    2
        dtype: int64

        More complicated user-defined functions can be used,
        as long as they expect a Series and return an array-like

        >>> s.sort_values(key=lambda x: (np.tan(x.cumsum())))
        0   -4
        3    2
        4    4
        1   -2
        2    0
        dtype: int64
        """
    def sort_index(self, *, axis: Axis = ..., level: IndexLabel | None, ascending: bool | Sequence[bool] = ..., inplace: bool = ..., kind: SortKind = ..., na_position: NaPosition = ..., sort_remaining: bool = ..., ignore_index: bool = ..., key: IndexKeyFunc | None) -> Series | None:
        """
        Sort Series by index labels.

        Returns a new Series sorted by label if `inplace` argument is
        ``False``, otherwise updates the original series and returns None.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        level : int, optional
            If not None, sort on values in specified index level(s).
        ascending : bool or list-like of bools, default True
            Sort ascending vs. descending. When the index is a MultiIndex the
            sort direction can be controlled for each level individually.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms. For
            DataFrames, this option is only applied when sorting on a single
            column or label.
        na_position : {'first', 'last'}, default 'last'
            If 'first' puts NaNs at the beginning, 'last' puts NaNs at the end.
            Not implemented for MultiIndex.
        sort_remaining : bool, default True
            If True and sorting by level and index is multilevel, sort by other
            levels too (in order) after sorting by specified level.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        Series or None
            The original Series sorted by the labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index: Sort DataFrame by the index.
        DataFrame.sort_values: Sort DataFrame by the value.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
        >>> s.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        Sort Descending

        >>> s.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        By default NaNs are put at the end, but use `na_position` to place
        them at the beginning

        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, np.nan])
        >>> s.sort_index(na_position='first')
        NaN     d
         1.0    c
         2.0    b
         3.0    a
        dtype: object

        Specify index level to sort

        >>> arrays = [np.array(['qux', 'qux', 'foo', 'foo',
        ...                     'baz', 'baz', 'bar', 'bar']),
        ...           np.array(['two', 'one', 'two', 'one',
        ...                     'two', 'one', 'two', 'one'])]
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
        >>> s.sort_index(level=1)
        bar  one    8
        baz  one    6
        foo  one    4
        qux  one    2
        bar  two    7
        baz  two    5
        foo  two    3
        qux  two    1
        dtype: int64

        Does not sort by remaining levels when sorting by levels

        >>> s.sort_index(level=1, sort_remaining=False)
        qux  one    2
        foo  one    4
        baz  one    6
        bar  one    8
        qux  two    1
        foo  two    3
        baz  two    5
        bar  two    7
        dtype: int64

        Apply a key function before sorting

        >>> s = pd.Series([1, 2, 3, 4], index=['A', 'b', 'C', 'd'])
        >>> s.sort_index(key=lambda x : x.str.lower())
        A    1
        b    2
        C    3
        d    4
        dtype: int64
        """
    def argsort(self, axis: Axis = ..., kind: SortKind = ..., order: None, stable: None) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Override ndarray.argsort. Argsorts the value, omitting NA/null values,
        and places the result in the same locations as the non-NA values.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        stable : None
            Has no effect but is accepted for compatibility with numpy.

        Returns
        -------
        Series[np.intp]
            Positions of values within the sort order with -1 indicating
            nan values.

        See Also
        --------
        numpy.ndarray.argsort : Returns the indices that would sort this array.

        Examples
        --------
        >>> s = pd.Series([3, 2, 1])
        >>> s.argsort()
        0    2
        1    1
        2    0
        dtype: int64
        """
    def nlargest(self, n: int = ..., keep: Literal['first', 'last', 'all'] = ...) -> Series:
        '''
        Return the largest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many descending sorted values.
        keep : {\'first\', \'last\', \'all\'}, default \'first\'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
              of appearance.
            - ``last`` : return the last `n` occurrences in reverse
              order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
              size larger than `n`.

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        See Also
        --------
        Series.nsmallest: Get the `n` smallest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
        relative to the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Malta": 434000, "Maldives": 434000,
        ...                         "Brunei": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nlargest()
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3``. Default `keep` value is \'first\'
        so Malta will be kept.

        >>> s.nlargest(3)
        France    65000000
        Italy     59000000
        Malta       434000
        dtype: int64

        The `n` largest elements where ``n=3`` and keeping the last duplicates.
        Brunei will be kept since it is the last with value 434000 based on
        the index order.

        >>> s.nlargest(3, keep=\'last\')
        France      65000000
        Italy       59000000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has five elements due to the three duplicates.

        >>> s.nlargest(3, keep=\'all\')
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64
        '''
    def nsmallest(self, n: int = ..., keep: Literal['first', 'last', 'all'] = ...) -> Series:
        '''
        Return the smallest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.
        keep : {\'first\', \'last\', \'all\'}, default \'first\'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
              of appearance.
            - ``last`` : return the last `n` occurrences in reverse
              order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
              size larger than `n`.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        See Also
        --------
        Series.nlargest: Get the `n` largest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values().head(n)`` for small `n` relative to
        the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Brunei": 434000, "Malta": 434000,
        ...                         "Maldives": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Brunei        434000
        Malta         434000
        Maldives      434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` smallest elements where ``n=5`` by default.

        >>> s.nsmallest()
        Montserrat    5200
        Nauru        11300
        Tuvalu       11300
        Anguilla     11300
        Iceland     337000
        dtype: int64

        The `n` smallest elements where ``n=3``. Default `keep` value is
        \'first\' so Nauru and Tuvalu will be kept.

        >>> s.nsmallest(3)
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` and keeping the last
        duplicates. Anguilla and Tuvalu will be kept since they are the last
        with value 11300 based on the index order.

        >>> s.nsmallest(3, keep=\'last\')
        Montserrat   5200
        Anguilla    11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has four elements due to the three duplicates.

        >>> s.nsmallest(3, keep=\'all\')
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        Anguilla    11300
        dtype: int64
        '''
    def swaplevel(self, i: Level = ..., j: Level = ..., copy: bool | None) -> Series:
        '''
        Swap levels i and j in a :class:`MultiIndex`.

        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i, j : int or str
            Levels of the indices to be swapped. Can pass level name as string.
        copy : bool, default True
                    Whether to copy underlying data.

                    .. note::
                        The `copy` keyword will change behavior in pandas 3.0.
                        `Copy-on-Write
                        <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                        will be enabled by default, which means that all methods with a
                        `copy` keyword will use a lazy copy mechanism to defer the copy and
                        ignore the `copy` keyword. The `copy` keyword will be removed in a
                        future version of pandas.

                        You can already get the future behavior and improvements through
                        enabling copy on write ``pd.options.mode.copy_on_write = True``

        Returns
        -------
        Series
            Series with levels swapped in MultiIndex.

        Examples
        --------
        >>> s = pd.Series(
        ...     ["A", "B", "A", "C"],
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> s
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C
        dtype: object

        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behaviour.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.

        >>> s.swaplevel()
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C
        dtype: object

        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.

        >>> s.swaplevel(0)
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C
        dtype: object

        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.

        >>> s.swaplevel(0, 1)
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C
        dtype: object
        '''
    def reorder_levels(self, order: Sequence[Level]) -> Series:
        '''
        Rearrange index levels using input order.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int representing new level order
            Reference level by number or key.

        Returns
        -------
        type of caller (new object)

        Examples
        --------
        >>> arrays = [np.array(["dog", "dog", "cat", "cat", "bird", "bird"]),
        ...           np.array(["white", "black", "white", "black", "white", "black"])]
        >>> s = pd.Series([1, 2, 3, 3, 5, 2], index=arrays)
        >>> s
        dog   white    1
              black    2
        cat   white    3
              black    3
        bird  white    5
              black    2
        dtype: int64
        >>> s.reorder_levels([1, 0])
        white  dog     1
        black  dog     2
        white  cat     3
        black  cat     3
        white  bird    5
        black  bird    2
        dtype: int64
        '''
    def explode(self, ignore_index: bool = ...) -> Series:
        """
        Transform each element of a list-like to a row.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of elements in
        the output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> s = pd.Series([[1, 2, 3], 'foo', [], [3, 4]])
        >>> s
        0    [1, 2, 3]
        1          foo
        2           []
        3       [3, 4]
        dtype: object

        >>> s.explode()
        0      1
        0      2
        0      3
        1    foo
        2    NaN
        3      3
        3      4
        dtype: object
        """
    def unstack(self, level: IndexLabel = ..., fill_value: Hashable | None, sort: bool = ...) -> DataFrame:
        """
        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.

        Parameters
        ----------
        level : int, str, or list of these, default last level
            Level(s) to unstack, can pass level name.
        fill_value : scalar value, default None
            Value to use when replacing NaN values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        DataFrame
            Unstacked Series.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4],
        ...               index=pd.MultiIndex.from_product([['one', 'two'],
        ...                                                 ['a', 'b']]))
        >>> s
        one  a    1
             b    2
        two  a    3
             b    4
        dtype: int64

        >>> s.unstack(level=-1)
             a  b
        one  1  2
        two  3  4

        >>> s.unstack(level=0)
           one  two
        a    1    3
        b    2    4
        """
    def map(self, arg: Callable | Mapping | Series, na_action: Literal['ignore'] | None) -> Series:
        """
        Map values of Series according to an input mapping or function.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict`` or
        a :class:`Series`.

        Parameters
        ----------
        arg : function, collections.abc.Mapping subclass or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        Series.replace: Replace values given in `to_replace` with `value`.
        DataFrame.apply : Apply a function row-/column-wise.
        DataFrame.map : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``NaN``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``NaN``.

        Examples
        --------
        >>> s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
        >>> s
        0      cat
        1      dog
        2      NaN
        3   rabbit
        dtype: object

        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
        in the ``dict`` are converted to ``NaN``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})
        0   kitten
        1    puppy
        2      NaN
        3      NaN
        dtype: object

        It also accepts a function:

        >>> s.map('I am a {}'.format)
        0       I am a cat
        1       I am a dog
        2       I am a nan
        3    I am a rabbit
        dtype: object

        To avoid applying the function to missing values (and keep them as
        ``NaN``) ``na_action='ignore'`` can be used:

        >>> s.map('I am a {}'.format, na_action='ignore')
        0     I am a cat
        1     I am a dog
        2            NaN
        3  I am a rabbit
        dtype: object
        """
    def _gotitem(self, key, ndim, subset) -> Self:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            Requested ndim of result.
        subset : object, default None
            Subset to act on.
        """
    def aggregate(self, func, axis: Axis = ..., *args, **kwargs):
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series or when passed to Series.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.
        axis : {0 or 'index'}
                Unused. Parameter needed for compatibility with DataFrame.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        Series.apply : Invoke function on a Series.
        Series.transform : Transform function producing a Series with like indexes.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.agg('min')
        1

        >>> s.agg(['min', 'max'])
        min   1
        max   4
        dtype: int64
        """
    def agg(self, func, axis: Axis = ..., *args, **kwargs):
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series or when passed to Series.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.
        axis : {0 or 'index'}
                Unused. Parameter needed for compatibility with DataFrame.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        Series.apply : Invoke function on a Series.
        Series.transform : Transform function producing a Series with like indexes.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.agg('min')
        1

        >>> s.agg(['min', 'max'])
        min   1
        max   4
        dtype: int64
        """
    def transform(self, func: AggFuncType, axis: Axis = ..., *args, **kwargs) -> DataFrame | Series:
        '''
        Call ``func`` on self producing a Series with the same axis shape as self.

        Parameters
        ----------
        func : function, str, list-like or dict-like
            Function to use for transforming the data. If a function, must either
            work when passed a Series or when passed to Series.apply. If func
            is both list-like and dict-like, dict-like behavior takes precedence.

            Accepted combinations are:

            - function
            - string function name
            - list-like of functions and/or function names, e.g. ``[np.exp, \'sqrt\']``
            - dict-like of axis labels -> functions, function names or list-like of such.
        axis : {0 or \'index\'}
                Unused. Parameter needed for compatibility with DataFrame.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        Series
            A Series that must have the same length as self.

        Raises
        ------
        ValueError : If the returned Series has a different length than self.

        See Also
        --------
        Series.agg : Only perform aggregating type operations.
        Series.apply : Invoke function on a Series.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({\'A\': range(3), \'B\': range(1, 4)})
        >>> df
           A  B
        0  0  1
        1  1  2
        2  2  3
        >>> df.transform(lambda x: x + 1)
           A  B
        0  1  2
        1  2  3
        2  3  4

        Even though the resulting Series must have the same length as the
        input Series, it is possible to provide several input functions:

        >>> s = pd.Series(range(3))
        >>> s
        0    0
        1    1
        2    2
        dtype: int64
        >>> s.transform([np.sqrt, np.exp])
               sqrt        exp
        0  0.000000   1.000000
        1  1.000000   2.718282
        2  1.414214   7.389056

        You can call transform on a GroupBy object:

        >>> df = pd.DataFrame({
        ...     "Date": [
        ...         "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05",
        ...         "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05"],
        ...     "Data": [5, 8, 6, 1, 50, 100, 60, 120],
        ... })
        >>> df
                 Date  Data
        0  2015-05-08     5
        1  2015-05-07     8
        2  2015-05-06     6
        3  2015-05-05     1
        4  2015-05-08    50
        5  2015-05-07   100
        6  2015-05-06    60
        7  2015-05-05   120
        >>> df.groupby(\'Date\')[\'Data\'].transform(\'sum\')
        0     55
        1    108
        2     66
        3    121
        4     55
        5    108
        6     66
        7    121
        Name: Data, dtype: int64

        >>> df = pd.DataFrame({
        ...     "c": [1, 1, 1, 2, 2, 2, 2],
        ...     "type": ["m", "n", "o", "m", "m", "n", "n"]
        ... })
        >>> df
           c type
        0  1    m
        1  1    n
        2  1    o
        3  2    m
        4  2    m
        5  2    n
        6  2    n
        >>> df[\'size\'] = df.groupby(\'c\')[\'type\'].transform(len)
        >>> df
           c type size
        0  1    m    3
        1  1    n    3
        2  1    o    3
        3  2    m    4
        4  2    m    4
        5  2    n    4
        6  2    n    4
        '''
    def apply(self, func: AggFuncType, convert_dtype: bool | lib.NoDefault = ..., args: tuple[Any, ...] = ..., *, by_row: Literal[False, 'compat'] = ..., **kwargs) -> DataFrame | Series:
        '''
        Invoke function on values of Series.

        Can be ufunc (a NumPy function that applies to the entire Series)
        or a Python function that only works on single values.

        Parameters
        ----------
        func : function
            Python function or NumPy ufunc to apply.
        convert_dtype : bool, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object. Note that the dtype is always
            preserved for some extension array dtypes, such as Categorical.

            .. deprecated:: 2.1.0
                ``convert_dtype`` has been deprecated. Do ``ser.astype(object).apply()``
                instead if you want ``convert_dtype=False``.
        args : tuple
            Positional arguments passed to func after the series value.
        by_row : False or "compat", default "compat"
            If ``"compat"`` and func is a callable, func will be passed each element of
            the Series, like ``Series.map``. If func is a list or dict of
            callables, will first try to translate each func into pandas methods. If
            that doesn\'t work, will try call to apply again with ``by_row="compat"``
            and if that fails, will call apply again with ``by_row=False``
            (backward compatible).
            If False, the func will be passed the whole Series at once.

            ``by_row`` has no effect when ``func`` is a string.

            .. versionadded:: 2.1.0
        **kwargs
            Additional keyword arguments passed to func.

        Returns
        -------
        Series or DataFrame
            If func returns a Series object the result will be a DataFrame.

        See Also
        --------
        Series.map: For element-wise operations.
        Series.agg: Only perform aggregating type operations.
        Series.transform: Only perform transforming type operations.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        Create a series with typical summer temperatures for each city.

        >>> s = pd.Series([20, 21, 12],
        ...               index=[\'London\', \'New York\', \'Helsinki\'])
        >>> s
        London      20
        New York    21
        Helsinki    12
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x):
        ...     return x ** 2
        >>> s.apply(square)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Square the values by passing an anonymous function as an
        argument to ``apply()``.

        >>> s.apply(lambda x: x ** 2)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword.

        >>> def subtract_custom_value(x, custom_value):
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``.

        >>> def add_custom_values(x, **kwargs):
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        dtype: int64

        Use a function from the Numpy library.

        >>> s.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        dtype: float64
        '''
    def _reindex_indexer(self, new_index: Index | None, indexer: npt.NDArray[np.intp] | None, copy: bool | None) -> Series: ...
    def _needs_reindex_multi(self, axes, method, level) -> bool:
        """
        Check if we do need a multi reindex; this is for compat with
        higher dims.
        """
    def rename(self, index: Renamer | Hashable | None, *, axis: Axis | None, copy: bool | None, inplace: bool = ..., level: Level | None, errors: IgnoreRaise = ...) -> Series | None:
        '''
        Alter Series index labels or name.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don\'t throw an
        error.

        Alternatively, change ``Series.name`` with a scalar value.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        index : scalar, hashable sequence, dict-like or function optional
            Functions or dict-like are transformations to apply to
            the index.
            Scalar or hashable sequence-like will alter the ``Series.name``
            attribute.
        axis : {0 or \'index\'}
            Unused. Parameter needed for compatibility with DataFrame.
        copy : bool, default True
            Also copy underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``
        inplace : bool, default False
            Whether to return a new Series. If True the value of copy is ignored.
        level : int or level name, default None
            In case of MultiIndex, only rename labels in the specified level.
        errors : {\'ignore\', \'raise\'}, default \'ignore\'
            If \'raise\', raise `KeyError` when a `dict-like mapper` or
            `index` contains labels that are not present in the index being transformed.
            If \'ignore\', existing keys will be renamed and extra keys will be ignored.

        Returns
        -------
        Series or None
            Series with index labels or name altered or None if ``inplace=True``.

        See Also
        --------
        DataFrame.rename : Corresponding DataFrame method.
        Series.rename_axis : Set the name of the axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        >>> s.rename(lambda x: x ** 2)  # function, changes labels
        0    1
        1    2
        4    3
        dtype: int64
        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels
        0    1
        3    2
        5    3
        dtype: int64
        '''
    def set_axis(self, labels, *, axis: Axis = ..., copy: bool | None) -> Series:
        """
        Assign desired index to given axis.

        Indexes for row labels can be changed by assigning
        a list-like or Index.

        Parameters
        ----------
        labels : list-like, Index
            The values for the new index.

        axis : {0 or 'index'}, default 0
            The axis to update. The value 0 identifies the rows. For `Series`
            this parameter is unused and defaults to 0.

        copy : bool, default True
            Whether to make a copy of the underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

        Returns
        -------
        Series
            An object of type Series.

        See Also
        --------
        Series.rename_axis : Alter the name of the index.

                Examples
                --------
                >>> s = pd.Series([1, 2, 3])
                >>> s
                0    1
                1    2
                2    3
                dtype: int64

                >>> s.set_axis(['a', 'b', 'c'], axis=0)
                a    1
                b    2
                c    3
                dtype: int64
        """
    def reindex(self, index, *, axis: Axis | None, method: ReindexMethod | None, copy: bool | None, level: Level | None, fill_value: Scalar | None, limit: int | None, tolerance) -> Series:
        '''
        Conform Series to new index with optional filling logic.

        Places NA/NaN in locations having no value in the previous index. A new object
        is produced unless the new index is equivalent to the current one and
        ``copy=False``.

        Parameters
        ----------

        index : array-like, optional
            New labels for the index. Preferably an Index object to avoid
            duplicating data.
        axis : int or str, optional
            Unused.
        method : {None, \'backfill\'/\'bfill\', \'pad\'/\'ffill\', \'nearest\'}
            Method to use for filling holes in reindexed DataFrame.
            Please note: this is only applicable to DataFrames/Series with a
            monotonically increasing/decreasing index.

            * None (default): don\'t fill gaps
            * pad / ffill: Propagate last valid observation forward to next
              valid.
            * backfill / bfill: Use next valid observation to fill gap.
            * nearest: Use nearest valid observations to fill gap.

        copy : bool, default True
            Return a new object, even if the passed indexes are the same.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : scalar, default np.nan
            Value to use for missing values. Defaults to NaN, but can be any
            "compatible" value.
        limit : int, default None
            Maximum number of consecutive elements to forward or backward fill.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index\'s type.

        Returns
        -------
        Series with changed index.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        ``DataFrame.reindex`` supports two calling conventions

        * ``(index=index_labels, columns=column_labels, ...)``
        * ``(labels, axis={\'index\', \'columns\'}, ...)``

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Create a dataframe with some fictional data.

        >>> index = [\'Firefox\', \'Chrome\', \'Safari\', \'IE10\', \'Konqueror\']
        >>> df = pd.DataFrame({\'http_status\': [200, 200, 404, 404, 301],
        ...                   \'response_time\': [0.04, 0.02, 0.07, 0.08, 1.0]},
        ...                   index=index)
        >>> df
                   http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00

        Create a new index and reindex the dataframe. By default
        values in the new index that do not have corresponding
        records in the dataframe are assigned ``NaN``.

        >>> new_index = [\'Safari\', \'Iceweasel\', \'Comodo Dragon\', \'IE10\',
        ...              \'Chrome\']
        >>> df.reindex(new_index)
                       http_status  response_time
        Safari               404.0           0.07
        Iceweasel              NaN            NaN
        Comodo Dragon          NaN            NaN
        IE10                 404.0           0.08
        Chrome               200.0           0.02

        We can fill in the missing values by passing a value to
        the keyword ``fill_value``. Because the index is not monotonically
        increasing or decreasing, we cannot use arguments to the keyword
        ``method`` to fill the ``NaN`` values.

        >>> df.reindex(new_index, fill_value=0)
                       http_status  response_time
        Safari                 404           0.07
        Iceweasel                0           0.00
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Chrome                 200           0.02

        >>> df.reindex(new_index, fill_value=\'missing\')
                      http_status response_time
        Safari                404          0.07
        Iceweasel         missing       missing
        Comodo Dragon     missing       missing
        IE10                  404          0.08
        Chrome                200          0.02

        We can also reindex the columns.

        >>> df.reindex(columns=[\'http_status\', \'user_agent\'])
                   http_status  user_agent
        Firefox            200         NaN
        Chrome             200         NaN
        Safari             404         NaN
        IE10               404         NaN
        Konqueror          301         NaN

        Or we can use "axis-style" keyword arguments

        >>> df.reindex([\'http_status\', \'user_agent\'], axis="columns")
                   http_status  user_agent
        Firefox            200         NaN
        Chrome             200         NaN
        Safari             404         NaN
        IE10               404         NaN
        Konqueror          301         NaN

        To further illustrate the filling functionality in
        ``reindex``, we will create a dataframe with a
        monotonically increasing index (for example, a sequence
        of dates).

        >>> date_index = pd.date_range(\'1/1/2010\', periods=6, freq=\'D\')
        >>> df2 = pd.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
        ...                    index=date_index)
        >>> df2
                    prices
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0

        Suppose we decide to expand the dataframe to cover a wider
        date range.

        >>> date_index2 = pd.date_range(\'12/29/2009\', periods=10, freq=\'D\')
        >>> df2.reindex(date_index2)
                    prices
        2009-12-29     NaN
        2009-12-30     NaN
        2009-12-31     NaN
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN

        The index entries that did not have a value in the original data frame
        (for example, \'2009-12-29\') are by default filled with ``NaN``.
        If desired, we can fill in the missing values using one of several
        options.

        For example, to back-propagate the last valid value to fill the ``NaN``
        values, pass ``bfill`` as an argument to the ``method`` keyword.

        >>> df2.reindex(date_index2, method=\'bfill\')
                    prices
        2009-12-29   100.0
        2009-12-30   100.0
        2009-12-31   100.0
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN

        Please note that the ``NaN`` value present in the original dataframe
        (at index value 2010-01-03) will not be filled by any of the
        value propagation schemes. This is because filling while reindexing
        does not look at dataframe values, but only compares the original and
        desired indexes. If you do want to fill in the ``NaN`` values present
        in the original dataframe, use the ``fillna()`` method.

        See the :ref:`user guide <basics.reindexing>` for more.
        '''
    def rename_axis(self, mapper: IndexLabel | lib.NoDefault = ..., *, index: pandas._libs.lib._NoDefault = ..., axis: Axis = ..., copy: bool = ..., inplace: bool = ...) -> Self | None:
        '''
        Set the name of the axis for the index or columns.

        Parameters
        ----------
        mapper : scalar, list-like, optional
            Value to set the axis name attribute.
        index, columns : scalar, list-like, dict-like or function, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to that axis\' values.
            Note that the ``columns`` parameter is not allowed if the
            object is a Series. This parameter only apply for DataFrame
            type objects.

            Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index``
            and/or ``columns``.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            The axis to rename. For `Series` this parameter is unused and defaults to 0.
        copy : bool, default None
            Also copy underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Series
            or DataFrame.

        Returns
        -------
        Series, DataFrame, or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Series.rename : Alter Series index labels or name.
        DataFrame.rename : Alter DataFrame index labels or name.
        Index.rename : Set new names on index.

        Notes
        -----
        ``DataFrame.rename_axis`` supports two calling conventions

        * ``(index=index_mapper, columns=columns_mapper, ...)``
        * ``(mapper, axis={\'index\', \'columns\'}, ...)``

        The first calling convention will only modify the names of
        the index and/or the names of the Index object that is the columns.
        In this case, the parameter ``copy`` is ignored.

        The second calling convention will modify the names of the
        corresponding index if mapper is a list or a scalar.
        However, if mapper is dict-like or a function, it will use the
        deprecated behavior of modifying the axis *labels*.

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Examples
        --------
        **Series**

        >>> s = pd.Series(["dog", "cat", "monkey"])
        >>> s
        0       dog
        1       cat
        2    monkey
        dtype: object
        >>> s.rename_axis("animal")
        animal
        0    dog
        1    cat
        2    monkey
        dtype: object

        **DataFrame**

        >>> df = pd.DataFrame({"num_legs": [4, 4, 2],
        ...                    "num_arms": [0, 0, 2]},
        ...                   ["dog", "cat", "monkey"])
        >>> df
                num_legs  num_arms
        dog            4         0
        cat            4         0
        monkey         2         2
        >>> df = df.rename_axis("animal")
        >>> df
                num_legs  num_arms
        animal
        dog            4         0
        cat            4         0
        monkey         2         2
        >>> df = df.rename_axis("limbs", axis="columns")
        >>> df
        limbs   num_legs  num_arms
        animal
        dog            4         0
        cat            4         0
        monkey         2         2

        **MultiIndex**

        >>> df.index = pd.MultiIndex.from_product([[\'mammal\'],
        ...                                        [\'dog\', \'cat\', \'monkey\']],
        ...                                       names=[\'type\', \'name\'])
        >>> df
        limbs          num_legs  num_arms
        type   name
        mammal dog            4         0
               cat            4         0
               monkey         2         2

        >>> df.rename_axis(index={\'type\': \'class\'})
        limbs          num_legs  num_arms
        class  name
        mammal dog            4         0
               cat            4         0
               monkey         2         2

        >>> df.rename_axis(columns=str.upper)
        LIMBS          num_legs  num_arms
        type   name
        mammal dog            4         0
               cat            4         0
               monkey         2         2
        '''
    def drop(self, labels: IndexLabel | None, *, axis: Axis = ..., index: IndexLabel | None, columns: IndexLabel | None, level: Level | None, inplace: bool = ..., errors: IgnoreRaise = ...) -> Series | None:
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed
        by specifying the level.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        index : single label or list-like
            Redundant for application on Series, but 'index' can be used instead
            of 'labels'.
        columns : single label or list-like
            No change is made to the Series; use 'index' or 'labels' instead.
        level : int or level name, optional
            For MultiIndex, level for which the labels will be removed.
        inplace : bool, default False
            If True, do operation inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are dropped.

        Returns
        -------
        Series or None
            Series with specified index labels removed or None if ``inplace=True``.

        Raises
        ------
        KeyError
            If none of the labels are found in the index.

        See Also
        --------
        Series.reindex : Return only specified index labels of Series.
        Series.dropna : Return series without null values.
        Series.drop_duplicates : Return Series with duplicate values removed.
        DataFrame.drop : Drop specified labels from rows or columns.

        Examples
        --------
        >>> s = pd.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A  0
        B  1
        C  2
        dtype: int64

        Drop labels B en C

        >>> s.drop(labels=['B', 'C'])
        A  0
        dtype: int64

        Drop 2nd level label in MultiIndex Series

        >>> midx = pd.MultiIndex(levels=[['llama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        llama   speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.drop(labels='weight', level=1)
        llama   speed      45.0
                length      1.2
        cow     speed      30.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        dtype: float64
        """
    def pop(self, item: Hashable) -> Any:
        """
        Return item and drops from series. Raise KeyError if not found.

        Parameters
        ----------
        item : label
            Index of the element that needs to be removed.

        Returns
        -------
        Value that is popped from series.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])

        >>> ser.pop(0)
        1

        >>> ser
        1    2
        2    3
        dtype: int64
        """
    def info(self, verbose: bool | None, buf: IO[str] | None, max_cols: int | None, memory_usage: bool | str | None, show_counts: bool = ...) -> None:
        '''
        Print a concise summary of a Series.

        This method prints information about a Series including
        the index dtype, non-null values and memory usage.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary. By default, the setting in
            ``pandas.options.display.max_info_columns`` is followed.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.

        memory_usage : bool, str, optional
            Specifies whether total memory usage of the Series
            elements (including the index) should be displayed. By default,
            this follows the ``pandas.options.display.memory_usage`` setting.

            True always show memory usage. False never shows memory usage.
            A value of \'deep\' is equivalent to "True with deep introspection".
            Memory usage is shown in human-readable units (base-2
            representation). Without deep introspection a memory estimation is
            made based in column dtype and number of rows assuming values
            consume the same memory amount for corresponding dtypes. With deep
            memory introspection, a real memory usage calculation is performed
            at the cost of computational resources. See the
            :ref:`Frequently Asked Questions <df-memory-usage>` for more
            details.
        show_counts : bool, optional
            Whether to show the non-null counts. By default, this is shown
            only if the DataFrame is smaller than
            ``pandas.options.display.max_info_rows`` and
            ``pandas.options.display.max_info_columns``. A value of True always
            shows the counts, and False never shows the counts.

        Returns
        -------
        None
            This method prints a summary of a Series and returns None.

        See Also
        --------
        Series.describe: Generate descriptive statistics of Series.
        Series.memory_usage: Memory usage of Series.

        Examples
        --------
        >>> int_values = [1, 2, 3, 4, 5]
        >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']
        >>> s = pd.Series(text_values, index=int_values)
        >>> s.info()
        <class \'pandas.core.series.Series\'>
        Index: 5 entries, 1 to 5
        Series name: None
        Non-Null Count  Dtype
        --------------  -----
        5 non-null      object
        dtypes: object(1)
        memory usage: 80.0+ bytes

        Prints a summary excluding information about its values:

        >>> s.info(verbose=False)
        <class \'pandas.core.series.Series\'>
        Index: 5 entries, 1 to 5
        dtypes: object(1)
        memory usage: 80.0+ bytes

        Pipe output of Series.info to buffer instead of sys.stdout, get
        buffer content and writes to a text file:

        >>> import io
        >>> buffer = io.StringIO()
        >>> s.info(buf=buffer)
        >>> s = buffer.getvalue()
        >>> with open("df_info.txt", "w",
        ...           encoding="utf-8") as f:  # doctest: +SKIP
        ...     f.write(s)
        260

        The `memory_usage` parameter allows deep introspection mode, specially
        useful for big Series and fine-tune memory optimization:

        >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)
        >>> s = pd.Series(np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6))
        >>> s.info()
        <class \'pandas.core.series.Series\'>
        RangeIndex: 1000000 entries, 0 to 999999
        Series name: None
        Non-Null Count    Dtype
        --------------    -----
        1000000 non-null  object
        dtypes: object(1)
        memory usage: 7.6+ MB

        >>> s.info(memory_usage=\'deep\')
        <class \'pandas.core.series.Series\'>
        RangeIndex: 1000000 entries, 0 to 999999
        Series name: None
        Non-Null Count    Dtype
        --------------    -----
        1000000 non-null  object
        dtypes: object(1)
        memory usage: 55.3 MB
        '''
    def _replace_single(self, to_replace, method: str, inplace: bool, limit):
        """
        Replaces values in a Series using the fill method specified when no
        replacement value is given in the replace method
        """
    def memory_usage(self, index: bool = ..., deep: bool = ...) -> int:
        '''
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.
        DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = pd.Series(range(3))
        >>> s.memory_usage()
        152

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        The memory footprint of `object` values is ignored by default:

        >>> s = pd.Series(["a", "b"])
        >>> s.values
        array([\'a\', \'b\'], dtype=object)
        >>> s.memory_usage()
        144
        >>> s.memory_usage(deep=True)
        244
        '''
    def isin(self, values) -> Series:
        """
        Whether elements in Series are contained in `values`.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        Series
            Series of booleans indicating if each element is in values.

        Raises
        ------
        TypeError
          * If `values` is a string

        See Also
        --------
        DataFrame.isin : Equivalent method on DataFrame.

        Examples
        --------
        >>> s = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama',
        ...                'hippo'], name='animal')
        >>> s.isin(['cow', 'llama'])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        To invert the boolean values, use the ``~`` operator:

        >>> ~s.isin(['cow', 'llama'])
        0    False
        1    False
        2    False
        3     True
        4    False
        5     True
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('llama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['llama'])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Strings and integers are distinct and are therefore not comparable:

        >>> pd.Series([1]).isin(['1'])
        0    False
        dtype: bool
        >>> pd.Series([1.1]).isin(['1.1'])
        0    False
        dtype: bool
        """
    def between(self, left, right, inclusive: Literal['both', 'neither', 'left', 'right'] = ...) -> Series:
        '''
        Return boolean Series equivalent to left <= series <= right.

        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : {"both", "neither", "left", "right"}
            Include boundaries. Whether to set each bound as closed or open.

            .. versionchanged:: 1.3.0

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = pd.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        With `inclusive` set to ``"neither"`` boundary values are excluded:

        >>> s.between(1, 4, inclusive="neither")
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = pd.Series([\'Alice\', \'Bob\', \'Carol\', \'Eve\'])
        >>> s.between(\'Anna\', \'Daniel\')
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        '''
    def case_when(self, caselist: list[tuple[ArrayLike | Callable[[Series], Series | np.ndarray | Sequence[bool]], ArrayLike | Scalar | Callable[[Series], Series | np.ndarray]]]) -> Series:
        """
        Replace values where the conditions are True.

        Parameters
        ----------
        caselist : A list of tuples of conditions and expected replacements
            Takes the form:  ``(condition0, replacement0)``,
            ``(condition1, replacement1)``, ... .
            ``condition`` should be a 1-D boolean array-like object
            or a callable. If ``condition`` is a callable,
            it is computed on the Series
            and should return a boolean Series or array.
            The callable must not change the input Series
            (though pandas doesn`t check it). ``replacement`` should be a
            1-D array-like object, a scalar or a callable.
            If ``replacement`` is a callable, it is computed on the Series
            and should return a scalar or Series. The callable
            must not change the input Series
            (though pandas doesn`t check it).

            .. versionadded:: 2.2.0

        Returns
        -------
        Series

        See Also
        --------
        Series.mask : Replace values where the condition is True.

        Examples
        --------
        >>> c = pd.Series([6, 7, 8, 9], name='c')
        >>> a = pd.Series([0, 0, 1, 2])
        >>> b = pd.Series([0, 3, 4, 5])

        >>> c.case_when(caselist=[(a.gt(0), a),  # condition, replacement
        ...                       (b.gt(0), b)])
        0    6
        1    3
        2    1
        3    2
        Name: c, dtype: int64
        """
    def isna(self) -> Series:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
        values.
        Everything else gets mapped to False values. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).

        Returns
        -------
        Series
            Mask of bool values for each element in Series that
            indicates whether an element is an NA value.

        See Also
        --------
        Series.isnull : Alias of isna.
        Series.notna : Boolean inverse of isna.
        Series.dropna : Omit axes labels with missing values.
        isna : Top-level isna.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> df = pd.DataFrame(dict(age=[5, 6, np.nan],
        ...                        born=[pd.NaT, pd.Timestamp('1939-05-27'),
        ...                              pd.Timestamp('1940-04-25')],
        ...                        name=['Alfred', 'Batman', ''],
        ...                        toy=[None, 'Batmobile', 'Joker']))
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.isna()
        0    False
        1    False
        2     True
        dtype: bool
        """
    def isnull(self) -> Series:
        """
        Series.isnull is an alias for Series.isna.

        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
        values.
        Everything else gets mapped to False values. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).

        Returns
        -------
        Series
            Mask of bool values for each element in Series that
            indicates whether an element is an NA value.

        See Also
        --------
        Series.isnull : Alias of isna.
        Series.notna : Boolean inverse of isna.
        Series.dropna : Omit axes labels with missing values.
        isna : Top-level isna.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> df = pd.DataFrame(dict(age=[5, 6, np.nan],
        ...                        born=[pd.NaT, pd.Timestamp('1939-05-27'),
        ...                              pd.Timestamp('1940-04-25')],
        ...                        name=['Alfred', 'Batman', ''],
        ...                        toy=[None, 'Batmobile', 'Joker']))
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.isna()
        0    False
        1    False
        2     True
        dtype: bool
        """
    def notna(self) -> Series:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        Series
            Mask of bool values for each element in Series that
            indicates whether an element is not an NA value.

        See Also
        --------
        Series.notnull : Alias of notna.
        Series.isna : Boolean inverse of notna.
        Series.dropna : Omit axes labels with missing values.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in a DataFrame are not NA.

        >>> df = pd.DataFrame(dict(age=[5, 6, np.nan],
        ...                        born=[pd.NaT, pd.Timestamp('1939-05-27'),
        ...                              pd.Timestamp('1940-04-25')],
        ...                        name=['Alfred', 'Batman', ''],
        ...                        toy=[None, 'Batmobile', 'Joker']))
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are not NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool
        """
    def notnull(self) -> Series:
        """
        Series.notnull is an alias for Series.notna.

        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        Series
            Mask of bool values for each element in Series that
            indicates whether an element is not an NA value.

        See Also
        --------
        Series.notnull : Alias of notna.
        Series.isna : Boolean inverse of notna.
        Series.dropna : Omit axes labels with missing values.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in a DataFrame are not NA.

        >>> df = pd.DataFrame(dict(age=[5, 6, np.nan],
        ...                        born=[pd.NaT, pd.Timestamp('1939-05-27'),
        ...                              pd.Timestamp('1940-04-25')],
        ...                        name=['Alfred', 'Batman', ''],
        ...                        toy=[None, 'Batmobile', 'Joker']))
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are not NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool
        """
    def dropna(self, *, axis: Axis = ..., inplace: bool = ..., how: AnyAll | None, ignore_index: bool = ...) -> Series | None:
        """
        Return a new Series with missing values removed.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        Series.isna: Indicate missing values.
        Series.notna : Indicate existing (non-missing) values.
        Series.fillna : Replace missing values.
        DataFrame.dropna : Drop rows or columns which contain NA values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> ser = pd.Series([1., 2., np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Empty strings are not considered NA values. ``None`` is considered an
        NA value.

        >>> ser = pd.Series([np.nan, 2, pd.NaT, '', None, 'I stay'])
        >>> ser
        0       NaN
        1         2
        2       NaT
        3
        4      None
        5    I stay
        dtype: object
        >>> ser.dropna()
        1         2
        3
        5    I stay
        dtype: object
        """
    def to_timestamp(self, freq: Frequency | None, how: Literal['s', 'e', 'start', 'end'] = ..., copy: bool | None) -> Series:
        """
        Cast to DatetimeIndex of Timestamps, at *beginning* of period.

        Parameters
        ----------
        freq : str, default frequency of PeriodIndex
            Desired frequency.
        how : {'s', 'e', 'start', 'end'}
            Convention for converting period to timestamp; start of period
            vs. end.
        copy : bool, default True
            Whether or not to return a copy.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

        Returns
        -------
        Series with DatetimeIndex

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2023', '2024', '2025'], freq='Y')
        >>> s1 = pd.Series([1, 2, 3], index=idx)
        >>> s1
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        The resulting frequency of the Timestamps is `YearBegin`

        >>> s1 = s1.to_timestamp()
        >>> s1
        2023-01-01    1
        2024-01-01    2
        2025-01-01    3
        Freq: YS-JAN, dtype: int64

        Using `freq` which is the offset that the Timestamps will have

        >>> s2 = pd.Series([1, 2, 3], index=idx)
        >>> s2 = s2.to_timestamp(freq='M')
        >>> s2
        2023-01-31    1
        2024-01-31    2
        2025-01-31    3
        Freq: YE-JAN, dtype: int64
        """
    def to_period(self, freq: str | None, copy: bool | None) -> Series:
        """
        Convert Series from DatetimeIndex to PeriodIndex.

        Parameters
        ----------
        freq : str, default None
            Frequency associated with the PeriodIndex.
        copy : bool, default True
            Whether or not to return a copy.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

        Returns
        -------
        Series
            Series with index converted to PeriodIndex.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023', '2024', '2025'])
        >>> s = pd.Series([1, 2, 3], index=idx)
        >>> s = s.to_period()
        >>> s
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        Viewing the index

        >>> s.index
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        """
    def hist(self: Series, by, ax, grid: bool = ..., xlabelsize: int | None, xrot: float | None, ylabelsize: int | None, yrot: float | None, figsize: tuple[int, int] | None, bins: int | Sequence[int] = ..., backend: str | None, legend: bool = ..., **kwargs):
        """
        Draw histogram of the input series using matplotlib.

        Parameters
        ----------
        by : object, optional
            If passed, then used to form histograms for separate groups.
        ax : matplotlib axis object
            If not passed, uses gca().
        grid : bool, default True
            Whether to show axis grid lines.
        xlabelsize : int, default None
            If specified changes the x-axis label size.
        xrot : float, default None
            Rotation of x axis labels.
        ylabelsize : int, default None
            If specified changes the y-axis label size.
        yrot : float, default None
            Rotation of y axis labels.
        figsize : tuple, default None
            Figure size in inches by default.
        bins : int or sequence, default 10
            Number of histogram bins to be used. If an integer is given, bins + 1
            bin edges are calculated and returned. If bins is a sequence, gives
            bin edges, including left edge of first bin and right edge of last
            bin. In this case, bins is returned unmodified.
        backend : str, default None
            Backend to use instead of the backend specified in the option
            ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
            specify the ``plotting.backend`` for the whole session, set
            ``pd.options.plotting.backend``.
        legend : bool, default False
            Whether to show the legend.

        **kwargs
            To be passed to the actual plotting function.

        Returns
        -------
        matplotlib.AxesSubplot
            A histogram plot.

        See Also
        --------
        matplotlib.axes.Axes.hist : Plot a histogram using matplotlib.

        Examples
        --------
        For Series:

        .. plot::
            :context: close-figs

            >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
            >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
            >>> hist = ser.hist()

        For Groupby:

        .. plot::
            :context: close-figs

            >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
            >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
            >>> hist = ser.groupby(level=0).hist()
        """
    def _cmp_method(self, other, op): ...
    def _logical_method(self, other, op): ...
    def _arith_method(self, other, op): ...
    def _align_for_op(self, right, align_asobject: bool = ...):
        """align lhs and rhs Series"""
    def _binop(self, other: Series, func, level, fill_value) -> Series:
        """
        Perform generic binary operation with optional fill value.

        Parameters
        ----------
        other : Series
        func : binary operator
        fill_value : float or object
            Value to substitute for NA/null values. If both Series are NA in a
            location, the result will be NA regardless of the passed fill value.
        level : int or level name, default None
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.

        Returns
        -------
        Series
        """
    def _construct_result(self, result: ArrayLike | tuple[ArrayLike, ArrayLike], name: Hashable) -> Series | tuple[Series, Series]:
        """
        Construct an appropriately-labelled Series from the result of an op.

        Parameters
        ----------
        result : ndarray or ExtensionArray
        name : Label

        Returns
        -------
        Series
            In the case of __divmod__ or __rdivmod__, a 2-tuple of Series.
        """
    def _flex_method(self, other, op, *, level, fill_value, axis: Axis = ...): ...
    def eq(self, other, level: Level | None, fill_value: float | None, axis: Axis = ...) -> Series:
        """
        Return Equal to of series and other, element-wise (binary operator `eq`).

        Equivalent to ``series == other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.eq(b, fill_value=0)
        a     True
        b    False
        c    False
        d    False
        e    False
        dtype: bool
        """
    def ne(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Not equal to of series and other, element-wise (binary operator `ne`).

        Equivalent to ``series != other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.ne(b, fill_value=0)
        a    False
        b     True
        c     True
        d     True
        e     True
        dtype: bool
        """
    def le(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Less than or equal to of series and other, element-wise (binary operator `le`).

        Equivalent to ``series <= other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.le(b, fill_value=0)
        a    False
        b     True
        c     True
        d    False
        e    False
        f     True
        dtype: bool
        """
    def lt(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Less than of series and other, element-wise (binary operator `lt`).

        Equivalent to ``series < other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.lt(b, fill_value=0)
        a    False
        b    False
        c     True
        d    False
        e    False
        f     True
        dtype: bool
        """
    def ge(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Greater than or equal to of series and other, element-wise (binary operator `ge`).

        Equivalent to ``series >= other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.ge(b, fill_value=0)
        a     True
        b     True
        c    False
        d    False
        e     True
        f    False
        dtype: bool
        """
    def gt(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Greater than of series and other, element-wise (binary operator `gt`).

        Equivalent to ``series > other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.gt(b, fill_value=0)
        a     True
        b    False
        c    False
        d    False
        e     True
        f    False
        dtype: bool
        """
    def add(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Addition of series and other, element-wise (binary operator `add`).

        Equivalent to ``series + other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.radd : Reverse of the Addition operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.add(b, fill_value=0)
        a    2.0
        b    1.0
        c    1.0
        d    1.0
        e    NaN
        dtype: float64
        """
    def radd(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Addition of series and other, element-wise (binary operator `radd`).

        Equivalent to ``other + series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.add : Element-wise Addition, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.add(b, fill_value=0)
        a    2.0
        b    1.0
        c    1.0
        d    1.0
        e    NaN
        dtype: float64
        """
    def sub(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Subtraction of series and other, element-wise (binary operator `sub`).

        Equivalent to ``series - other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rsub : Reverse of the Subtraction operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.subtract(b, fill_value=0)
        a    0.0
        b    1.0
        c    1.0
        d   -1.0
        e    NaN
        dtype: float64
        """
    def subtract(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Subtraction of series and other, element-wise (binary operator `sub`).

        Equivalent to ``series - other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rsub : Reverse of the Subtraction operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.subtract(b, fill_value=0)
        a    0.0
        b    1.0
        c    1.0
        d   -1.0
        e    NaN
        dtype: float64
        """
    def rsub(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Subtraction of series and other, element-wise (binary operator `rsub`).

        Equivalent to ``other - series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.sub : Element-wise Subtraction, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.subtract(b, fill_value=0)
        a    0.0
        b    1.0
        c    1.0
        d   -1.0
        e    NaN
        dtype: float64
        """
    def mul(self, other, level: Level | None, fill_value: float | None, axis: Axis = ...) -> Series:
        """
        Return Multiplication of series and other, element-wise (binary operator `mul`).

        Equivalent to ``series * other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rmul : Reverse of the Multiplication operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.multiply(b, fill_value=0)
        a    1.0
        b    0.0
        c    0.0
        d    0.0
        e    NaN
        dtype: float64
        """
    def multiply(self, other, level: Level | None, fill_value: float | None, axis: Axis = ...) -> Series:
        """
        Return Multiplication of series and other, element-wise (binary operator `mul`).

        Equivalent to ``series * other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rmul : Reverse of the Multiplication operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.multiply(b, fill_value=0)
        a    1.0
        b    0.0
        c    0.0
        d    0.0
        e    NaN
        dtype: float64
        """
    def rmul(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Multiplication of series and other, element-wise (binary operator `rmul`).

        Equivalent to ``other * series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.mul : Element-wise Multiplication, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.multiply(b, fill_value=0)
        a    1.0
        b    0.0
        c    0.0
        d    0.0
        e    NaN
        dtype: float64
        """
    def truediv(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Floating division of series and other, element-wise (binary operator `truediv`).

        Equivalent to ``series / other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rtruediv : Reverse of the Floating division operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divide(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def div(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Floating division of series and other, element-wise (binary operator `truediv`).

        Equivalent to ``series / other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rtruediv : Reverse of the Floating division operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divide(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def divide(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Floating division of series and other, element-wise (binary operator `truediv`).

        Equivalent to ``series / other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rtruediv : Reverse of the Floating division operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divide(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def rtruediv(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Floating division of series and other, element-wise (binary operator `rtruediv`).

        Equivalent to ``other / series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.truediv : Element-wise Floating division, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divide(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def rdiv(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Floating division of series and other, element-wise (binary operator `rtruediv`).

        Equivalent to ``other / series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.truediv : Element-wise Floating division, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divide(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def floordiv(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Integer division of series and other, element-wise (binary operator `floordiv`).

        Equivalent to ``series // other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rfloordiv : Reverse of the Integer division operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.floordiv(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def rfloordiv(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Integer division of series and other, element-wise (binary operator `rfloordiv`).

        Equivalent to ``other // series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.floordiv : Element-wise Integer division, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.floordiv(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
    def mod(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Modulo of series and other, element-wise (binary operator `mod`).

        Equivalent to ``series % other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rmod : Reverse of the Modulo operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.mod(b, fill_value=0)
        a    0.0
        b    NaN
        c    NaN
        d    0.0
        e    NaN
        dtype: float64
        """
    def rmod(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Modulo of series and other, element-wise (binary operator `rmod`).

        Equivalent to ``other % series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.mod : Element-wise Modulo, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.mod(b, fill_value=0)
        a    0.0
        b    NaN
        c    NaN
        d    0.0
        e    NaN
        dtype: float64
        """
    def pow(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Exponential power of series and other, element-wise (binary operator `pow`).

        Equivalent to ``series ** other``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rpow : Reverse of the Exponential power operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.pow(b, fill_value=0)
        a    1.0
        b    1.0
        c    1.0
        d    0.0
        e    NaN
        dtype: float64
        """
    def rpow(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Exponential power of series and other, element-wise (binary operator `rpow`).

        Equivalent to ``other ** series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.pow : Element-wise Exponential power, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.pow(b, fill_value=0)
        a    1.0
        b    1.0
        c    1.0
        d    0.0
        e    NaN
        dtype: float64
        """
    def divmod(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Integer division and modulo of series and other, element-wise (binary operator `divmod`).

        Equivalent to ``divmod(series, other)``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        2-Tuple of Series
            The result of the operation.

        See Also
        --------
        Series.rdivmod : Reverse of the Integer division and modulo operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divmod(b, fill_value=0)
        (a    1.0
         b    inf
         c    inf
         d    0.0
         e    NaN
         dtype: float64,
         a    0.0
         b    NaN
         c    NaN
         d    0.0
         e    NaN
         dtype: float64)
        """
    def rdivmod(self, other, level, fill_value, axis: Axis = ...) -> Series:
        """
        Return Integer division and modulo of series and other, element-wise (binary operator `rdivmod`).

        Equivalent to ``other divmod series``, but with support to substitute a fill_value for
        missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        2-Tuple of Series
            The result of the operation.

        See Also
        --------
        Series.divmod : Element-wise Integer division and modulo, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divmod(b, fill_value=0)
        (a    1.0
         b    inf
         c    inf
         d    0.0
         e    NaN
         dtype: float64,
         a    0.0
         b    NaN
         c    NaN
         d    0.0
         e    NaN
         dtype: float64)
        """
    def _reduce(self, op, name: str, *, axis: Axis = ..., skipna: bool = ..., numeric_only: bool = ..., filter_type, **kwds):
        """
        Perform a reduction operation.

        If we have an ndarray as a value, then simply perform the operation,
        otherwise delegate to the object.
        """
    def any(self, *, axis: Axis = ..., bool_only: bool = ..., skipna: bool = ..., **kwargs) -> bool:
        '''
        Return whether any element is True, potentially over an axis.

        Returns False unless there is at least one element within a series or
        along a Dataframe axis that is True or equivalent (e.g. non-zero or
        non-empty).

        Parameters
        ----------
        axis : {0 or \'index\', 1 or \'columns\', None}, default 0
            Indicate which axis or axes should be reduced. For `Series` this parameter
            is unused and defaults to 0.

            * 0 / \'index\' : reduce the index, return a Series whose index is the
              original column labels.
            * 1 / \'columns\' : reduce the columns, return a Series whose index is the
              original index.
            * None : reduce all axes, return a scalar.

        bool_only : bool, default False
            Include only boolean columns. Not implemented for Series.
        skipna : bool, default True
            Exclude NA/null values. If the entire row/column is NA and skipna is
            True, then the result will be False, as for an empty row/column.
            If skipna is False, then NA are treated as True, because these are not
            equal to zero.
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        scalar or Series
            If level is specified, then, Series is returned; otherwise, scalar
            is returned.

        See Also
        --------
        numpy.any : Numpy version of this method.
        Series.any : Return whether any element is True.
        Series.all : Return whether all elements are True.
        DataFrame.any : Return whether any element is True over requested axis.
        DataFrame.all : Return whether all elements are True over requested axis.

        Examples
        --------
        **Series**

        For Series input, the output is a scalar indicating whether any element
        is True.

        >>> pd.Series([False, False]).any()
        False
        >>> pd.Series([True, False]).any()
        True
        >>> pd.Series([], dtype="float64").any()
        False
        >>> pd.Series([np.nan]).any()
        False
        >>> pd.Series([np.nan]).any(skipna=False)
        True

        **DataFrame**

        Whether each column contains at least one True element (the default).

        >>> df = pd.DataFrame({"A": [1, 2], "B": [0, 2], "C": [0, 0]})
        >>> df
           A  B  C
        0  1  0  0
        1  2  2  0

        >>> df.any()
        A     True
        B     True
        C    False
        dtype: bool

        Aggregating over the columns.

        >>> df = pd.DataFrame({"A": [True, False], "B": [1, 2]})
        >>> df
               A  B
        0   True  1
        1  False  2

        >>> df.any(axis=\'columns\')
        0    True
        1    True
        dtype: bool

        >>> df = pd.DataFrame({"A": [True, False], "B": [1, 0]})
        >>> df
               A  B
        0   True  1
        1  False  0

        >>> df.any(axis=\'columns\')
        0    True
        1    False
        dtype: bool

        Aggregating over the entire DataFrame with ``axis=None``.

        >>> df.any(axis=None)
        True

        `any` for an empty DataFrame is an empty Series.

        >>> pd.DataFrame([]).any()
        Series([], dtype: bool)
        '''
    def all(self, axis: Axis = ..., bool_only: bool = ..., skipna: bool = ..., **kwargs) -> bool:
        '''
        Return whether all elements are True, potentially over an axis.

        Returns True unless there at least one element within a series or
        along a Dataframe axis that is False or equivalent (e.g. zero or
        empty).

        Parameters
        ----------
        axis : {0 or \'index\', 1 or \'columns\', None}, default 0
            Indicate which axis or axes should be reduced. For `Series` this parameter
            is unused and defaults to 0.

            * 0 / \'index\' : reduce the index, return a Series whose index is the
              original column labels.
            * 1 / \'columns\' : reduce the columns, return a Series whose index is the
              original index.
            * None : reduce all axes, return a scalar.

        bool_only : bool, default False
            Include only boolean columns. Not implemented for Series.
        skipna : bool, default True
            Exclude NA/null values. If the entire row/column is NA and skipna is
            True, then the result will be True, as for an empty row/column.
            If skipna is False, then NA are treated as True, because these are not
            equal to zero.
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        scalar or Series
            If level is specified, then, Series is returned; otherwise, scalar
            is returned.

        See Also
        --------
        Series.all : Return True if all elements are True.
        DataFrame.any : Return True if one (or more) elements are True.

        Examples
        --------
        **Series**

        >>> pd.Series([True, True]).all()
        True
        >>> pd.Series([True, False]).all()
        False
        >>> pd.Series([], dtype="float64").all()
        True
        >>> pd.Series([np.nan]).all()
        True
        >>> pd.Series([np.nan]).all(skipna=False)
        True

        **DataFrames**

        Create a dataframe from a dictionary.

        >>> df = pd.DataFrame({\'col1\': [True, True], \'col2\': [True, False]})
        >>> df
           col1   col2
        0  True   True
        1  True  False

        Default behaviour checks if values in each column all return True.

        >>> df.all()
        col1     True
        col2    False
        dtype: bool

        Specify ``axis=\'columns\'`` to check if values in each row all return True.

        >>> df.all(axis=\'columns\')
        0     True
        1    False
        dtype: bool

        Or ``axis=None`` for whether every value is True.

        >>> df.all(axis=None)
        False
        '''
    def min(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return the minimum of the values over the requested axis.

        If you want the *index* of the minimum, use ``idxmin``. This is the equivalent of the ``numpy.ndarray`` method ``argmin``.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays([
        ...     ['warm', 'warm', 'cold', 'cold'],
        ...     ['dog', 'falcon', 'fish', 'spider']],
        ...     names=['blooded', 'animal'])
        >>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                 falcon    2
        cold     fish      0
                 spider    8
        Name: legs, dtype: int64

        >>> s.min()
        0
        """
    def max(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return the maximum of the values over the requested axis.

        If you want the *index* of the maximum, use ``idxmax``. This is the equivalent of the ``numpy.ndarray`` method ``argmax``.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays([
        ...     ['warm', 'warm', 'cold', 'cold'],
        ...     ['dog', 'falcon', 'fish', 'spider']],
        ...     names=['blooded', 'animal'])
        >>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                 falcon    2
        cold     fish      0
                 spider    8
        Name: legs, dtype: int64

        >>> s.max()
        8
        """
    def sum(self, axis: Axis | None, skipna: bool = ..., numeric_only: bool = ..., min_count: int = ..., **kwargs):
        '''
        Return the sum of the values over the requested axis.

        This is equivalent to the method ``numpy.sum``.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sum with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays([
        ...     [\'warm\', \'warm\', \'cold\', \'cold\'],
        ...     [\'dog\', \'falcon\', \'fish\', \'spider\']],
        ...     names=[\'blooded\', \'animal\'])
        >>> s = pd.Series([4, 2, 0, 8], name=\'legs\', index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                 falcon    2
        cold     fish      0
                 spider    8
        Name: legs, dtype: int64

        >>> s.sum()
        14

        By default, the sum of an empty or all-NA Series is ``0``.

        >>> pd.Series([], dtype="float64").sum()  # min_count=0 is the default
        0.0

        This can be controlled with the ``min_count`` parameter. For example, if
        you\'d like the sum of an empty series to be NaN, pass ``min_count=1``.

        >>> pd.Series([], dtype="float64").sum(min_count=1)
        nan

        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.

        >>> pd.Series([np.nan]).sum()
        0.0

        >>> pd.Series([np.nan]).sum(min_count=1)
        nan
        '''
    def prod(self, axis: Axis | None, skipna: bool = ..., numeric_only: bool = ..., min_count: int = ..., **kwargs):
        '''
        Return the product of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.prod with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        By default, the product of an empty or all-NA Series is ``1``

        >>> pd.Series([], dtype="float64").prod()
        1.0

        This can be controlled with the ``min_count`` parameter

        >>> pd.Series([], dtype="float64").prod(min_count=1)
        nan

        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.

        >>> pd.Series([np.nan]).prod()
        1.0

        >>> pd.Series([np.nan]).prod(min_count=1)
        nan
        '''
    def mean(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return the mean of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

                    Examples
                    --------
                    >>> s = pd.Series([1, 2, 3])
                    >>> s.mean()
                    2.0

                    With a DataFrame

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
                    >>> df
                           a   b
                    tiger  1   2
                    zebra  2   3
                    >>> df.mean()
                    a   1.5
                    b   2.5
                    dtype: float64

                    Using axis=1

                    >>> df.mean(axis=1)
                    tiger   1.5
                    zebra   2.5
                    dtype: float64

                    In this case, `numeric_only` should be set to `True` to avoid
                    getting an error.

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
                    ...                   index=['tiger', 'zebra'])
                    >>> df.mean(numeric_only=True)
                    a   1.5
                    dtype: float64
        """
    def median(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return the median of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

                    Examples
                    --------
                    >>> s = pd.Series([1, 2, 3])
                    >>> s.median()
                    2.0

                    With a DataFrame

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
                    >>> df
                           a   b
                    tiger  1   2
                    zebra  2   3
                    >>> df.median()
                    a   1.5
                    b   2.5
                    dtype: float64

                    Using axis=1

                    >>> df.median(axis=1)
                    tiger   1.5
                    zebra   2.5
                    dtype: float64

                    In this case, `numeric_only` should be set to `True`
                    to avoid getting an error.

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
                    ...                   index=['tiger', 'zebra'])
                    >>> df.median(numeric_only=True)
                    a   1.5
                    dtype: float64
        """
    def sem(self, axis: Axis | None, skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
        axis : {index (0)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sem with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        Returns
        -------
        scalar or Series (if level specified) 

                    Examples
                    --------
                    >>> s = pd.Series([1, 2, 3])
                    >>> s.sem().round(6)
                    0.57735

                    With a DataFrame

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
                    >>> df
                           a   b
                    tiger  1   2
                    zebra  2   3
                    >>> df.sem()
                    a   0.5
                    b   0.5
                    dtype: float64

                    Using axis=1

                    >>> df.sem(axis=1)
                    tiger   0.5
                    zebra   0.5
                    dtype: float64

                    In this case, `numeric_only` should be set to `True`
                    to avoid getting an error.

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
                    ...                   index=['tiger', 'zebra'])
                    >>> df.sem(numeric_only=True)
                    a   0.5
                    dtype: float64
        """
    def var(self, axis: Axis | None, skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.var with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        Returns
        -------
        scalar or Series (if level specified) 

        Examples
        --------
        >>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
        ...                    'age': [21, 25, 62, 43],
        ...                    'height': [1.61, 1.87, 1.49, 2.01]}
        ...                   ).set_index('person_id')
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.var()
        age       352.916667
        height      0.056367
        dtype: float64

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

        >>> df.var(ddof=0)
        age       264.687500
        height      0.042275
        dtype: float64
        """
    def std(self, axis: Axis | None, skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.std with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        Returns
        -------
        scalar or Series (if level specified) 

        Notes
        -----
        To have the same behaviour as `numpy.std`, use `ddof=0` (instead of the
        default `ddof=1`)

        Examples
        --------
        >>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
        ...                    'age': [21, 25, 62, 43],
        ...                    'height': [1.61, 1.87, 1.49, 2.01]}
        ...                   ).set_index('person_id')
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        The standard deviation of the columns can be found as follows:

        >>> df.std()
        age       18.786076
        height     0.237417
        dtype: float64

        Alternatively, `ddof=0` can be set to normalize by N instead of N-1:

        >>> df.std(ddof=0)
        age       16.269219
        height     0.205609
        dtype: float64
        """
    def skew(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased skew over requested axis.

        Normalized by N-1.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

                    Examples
                    --------
                    >>> s = pd.Series([1, 2, 3])
                    >>> s.skew()
                    0.0

                    With a DataFrame

                    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 3, 5]},
                    ...                   index=['tiger', 'zebra', 'cow'])
                    >>> df
                            a   b   c
                    tiger   1   2   1
                    zebra   2   3   3
                    cow     3   4   5
                    >>> df.skew()
                    a   0.0
                    b   0.0
                    c   0.0
                    dtype: float64

                    Using axis=1

                    >>> df.skew(axis=1)
                    tiger   1.732051
                    zebra  -1.732051
                    cow     0.000000
                    dtype: float64

                    In this case, `numeric_only` should be set to `True` to avoid
                    getting an error.

                    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['T', 'Z', 'X']},
                    ...                   index=['tiger', 'zebra', 'cow'])
                    >>> df.skew(numeric_only=True)
                    a   0.0
                    dtype: float64
        """
    def kurt(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

                    Examples
                    --------
                    >>> s = pd.Series([1, 2, 2, 3], index=['cat', 'dog', 'dog', 'mouse'])
                    >>> s
                    cat    1
                    dog    2
                    dog    2
                    mouse  3
                    dtype: int64
                    >>> s.kurt()
                    1.5

                    With a DataFrame

                    >>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},
                    ...                   index=['cat', 'dog', 'dog', 'mouse'])
                    >>> df
                           a   b
                      cat  1   3
                      dog  2   4
                      dog  2   4
                    mouse  3   4
                    >>> df.kurt()
                    a   1.5
                    b   4.0
                    dtype: float64

                    With axis=None

                    >>> df.kurt(axis=None).round(6)
                    -0.988693

                    Using axis=1

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},
                    ...                   index=['cat', 'dog'])
                    >>> df.kurt(axis=1)
                    cat   -6.0
                    dog   -6.0
                    dtype: float64
        """
    def kurtosis(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

                    Examples
                    --------
                    >>> s = pd.Series([1, 2, 2, 3], index=['cat', 'dog', 'dog', 'mouse'])
                    >>> s
                    cat    1
                    dog    2
                    dog    2
                    mouse  3
                    dtype: int64
                    >>> s.kurt()
                    1.5

                    With a DataFrame

                    >>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},
                    ...                   index=['cat', 'dog', 'dog', 'mouse'])
                    >>> df
                           a   b
                      cat  1   3
                      dog  2   4
                      dog  2   4
                    mouse  3   4
                    >>> df.kurt()
                    a   1.5
                    b   4.0
                    dtype: float64

                    With axis=None

                    >>> df.kurt(axis=None).round(6)
                    -0.988693

                    Using axis=1

                    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},
                    ...                   index=['cat', 'dog'])
                    >>> df.kurt(axis=1)
                    cat   -6.0
                    dog   -6.0
                    dtype: float64
        """
    def product(self, axis: Axis | None, skipna: bool = ..., numeric_only: bool = ..., min_count: int = ..., **kwargs):
        '''
        Return the product of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.prod with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or scalar

        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        By default, the product of an empty or all-NA Series is ``1``

        >>> pd.Series([], dtype="float64").prod()
        1.0

        This can be controlled with the ``min_count`` parameter

        >>> pd.Series([], dtype="float64").prod(min_count=1)
        nan

        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.

        >>> pd.Series([np.nan]).prod()
        1.0

        >>> pd.Series([np.nan]).prod(min_count=1)
        nan
        '''
    def cummin(self, axis: Axis | None, skipna: bool = ..., *args, **kwargs):
        """
        Return cumulative minimum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative
        minimum.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The index or the name of the axis. 0 is equivalent to None or 'index'.
            For `Series` this parameter is unused and defaults to 0.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        *args, **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        scalar or Series
            Return cumulative minimum of scalar or Series.

        See Also
        --------
        core.window.expanding.Expanding.min : Similar functionality
            but ignores ``NaN`` values.
        Series.min : Return the minimum over
            Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        **Series**

        >>> s = pd.Series([2, np.nan, 5, -1, 0])
        >>> s
        0    2.0
        1    NaN
        2    5.0
        3   -1.0
        4    0.0
        dtype: float64

        By default, NA values are ignored.

        >>> s.cummin()
        0    2.0
        1    NaN
        2    2.0
        3   -1.0
        4   -1.0
        dtype: float64

        To include NA values in the operation, use ``skipna=False``

        >>> s.cummin(skipna=False)
        0    2.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        **DataFrame**

        >>> df = pd.DataFrame([[2.0, 1.0],
        ...                    [3.0, np.nan],
        ...                    [1.0, 0.0]],
        ...                   columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the minimum
        in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

        >>> df.cummin()
             A    B
        0  2.0  1.0
        1  2.0  NaN
        2  1.0  0.0

        To iterate over columns and find the minimum in each row,
        use ``axis=1``

        >>> df.cummin(axis=1)
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0
        """
    def cummax(self, axis: Axis | None, skipna: bool = ..., *args, **kwargs):
        """
        Return cumulative maximum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative
        maximum.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The index or the name of the axis. 0 is equivalent to None or 'index'.
            For `Series` this parameter is unused and defaults to 0.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        *args, **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        scalar or Series
            Return cumulative maximum of scalar or Series.

        See Also
        --------
        core.window.expanding.Expanding.max : Similar functionality
            but ignores ``NaN`` values.
        Series.max : Return the maximum over
            Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        **Series**

        >>> s = pd.Series([2, np.nan, 5, -1, 0])
        >>> s
        0    2.0
        1    NaN
        2    5.0
        3   -1.0
        4    0.0
        dtype: float64

        By default, NA values are ignored.

        >>> s.cummax()
        0    2.0
        1    NaN
        2    5.0
        3    5.0
        4    5.0
        dtype: float64

        To include NA values in the operation, use ``skipna=False``

        >>> s.cummax(skipna=False)
        0    2.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        **DataFrame**

        >>> df = pd.DataFrame([[2.0, 1.0],
        ...                    [3.0, np.nan],
        ...                    [1.0, 0.0]],
        ...                   columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the maximum
        in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

        >>> df.cummax()
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  3.0  1.0

        To iterate over columns and find the maximum in each row,
        use ``axis=1``

        >>> df.cummax(axis=1)
             A    B
        0  2.0  2.0
        1  3.0  NaN
        2  1.0  1.0
        """
    def cumsum(self, axis: Axis | None, skipna: bool = ..., *args, **kwargs):
        """
        Return cumulative sum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative
        sum.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The index or the name of the axis. 0 is equivalent to None or 'index'.
            For `Series` this parameter is unused and defaults to 0.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        *args, **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        scalar or Series
            Return cumulative sum of scalar or Series.

        See Also
        --------
        core.window.expanding.Expanding.sum : Similar functionality
            but ignores ``NaN`` values.
        Series.sum : Return the sum over
            Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        **Series**

        >>> s = pd.Series([2, np.nan, 5, -1, 0])
        >>> s
        0    2.0
        1    NaN
        2    5.0
        3   -1.0
        4    0.0
        dtype: float64

        By default, NA values are ignored.

        >>> s.cumsum()
        0    2.0
        1    NaN
        2    7.0
        3    6.0
        4    6.0
        dtype: float64

        To include NA values in the operation, use ``skipna=False``

        >>> s.cumsum(skipna=False)
        0    2.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        **DataFrame**

        >>> df = pd.DataFrame([[2.0, 1.0],
        ...                    [3.0, np.nan],
        ...                    [1.0, 0.0]],
        ...                   columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the sum
        in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

        >>> df.cumsum()
             A    B
        0  2.0  1.0
        1  5.0  NaN
        2  6.0  1.0

        To iterate over columns and find the sum in each row,
        use ``axis=1``

        >>> df.cumsum(axis=1)
             A    B
        0  2.0  3.0
        1  3.0  NaN
        2  1.0  1.0
        """
    def cumprod(self, axis: Axis | None, skipna: bool = ..., *args, **kwargs):
        """
        Return cumulative product over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative
        product.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The index or the name of the axis. 0 is equivalent to None or 'index'.
            For `Series` this parameter is unused and defaults to 0.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        *args, **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        scalar or Series
            Return cumulative product of scalar or Series.

        See Also
        --------
        core.window.expanding.Expanding.prod : Similar functionality
            but ignores ``NaN`` values.
        Series.prod : Return the product over
            Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        **Series**

        >>> s = pd.Series([2, np.nan, 5, -1, 0])
        >>> s
        0    2.0
        1    NaN
        2    5.0
        3   -1.0
        4    0.0
        dtype: float64

        By default, NA values are ignored.

        >>> s.cumprod()
        0     2.0
        1     NaN
        2    10.0
        3   -10.0
        4    -0.0
        dtype: float64

        To include NA values in the operation, use ``skipna=False``

        >>> s.cumprod(skipna=False)
        0    2.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        **DataFrame**

        >>> df = pd.DataFrame([[2.0, 1.0],
        ...                    [3.0, np.nan],
        ...                    [1.0, 0.0]],
        ...                   columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the product
        in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

        >>> df.cumprod()
             A    B
        0  2.0  1.0
        1  6.0  NaN
        2  6.0  0.0

        To iterate over columns and find the product in each row,
        use ``axis=1``

        >>> df.cumprod(axis=1)
             A    B
        0  2.0  2.0
        1  3.0  NaN
        2  1.0  0.0
        """
    @property
    def hasnans(self): ...
    @property
    def _constructor(self): ...
    @property
    def _constructor_expanddim(self): ...
    @property
    def _can_hold_na(self): ...
    @property
    def dtype(self): ...
    @property
    def dtypes(self): ...
    @property
    def values(self): ...
    @property
    def _values(self): ...
    @property
    def _references(self): ...
    @property
    def array(self): ...
    @property
    def axes(self): ...
    @property
    def _is_cached(self): ...

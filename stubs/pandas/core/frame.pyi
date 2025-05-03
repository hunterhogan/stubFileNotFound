import collections
import datetime
import fmt as fmt
import lib as lib
import np
import np.rec
import npt
import pandas as pandas
import pandas._libs.algos as libalgos
import pandas._libs.lib
import pandas._libs.properties as properties
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algorithms
import pandas.core.arraylike
import pandas.core.arrays.sparse.accessor
import pandas.core.common as com
import pandas.core.generic
import pandas.core.methods.selectn as selectn
import pandas.core.nanops as nanops
import pandas.core.ops as ops
import pandas.core.roperator as roperator
import pandas.core.series
import pandas.io.formats.console as console
import pandas.plotting._core
from _typeshed import Incomplete
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._config.config import _get_option as _get_option, get_option as get_option
from pandas._libs.hashtable import duplicated as duplicated
from pandas._libs.lib import is_float as is_float, is_integer as is_integer, is_iterator as is_iterator, is_list_like as is_list_like, is_range_indexer as is_range_indexer, is_scalar as is_scalar
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.accessor import CachedAccessor as CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result as reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi as take_2d_multi
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.arrays.period import PeriodArray as PeriodArray
from pandas.core.arrays.sparse.accessor import SparseFrameAccessor as SparseFrameAccessor
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, sanitize_array as sanitize_array, sanitize_masked_array as sanitize_masked_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import can_hold_element as can_hold_element, construct_1d_arraylike_from_scalar as construct_1d_arraylike_from_scalar, construct_2d_arraylike_from_scalar as construct_2d_arraylike_from_scalar, find_common_type as find_common_type, infer_dtype_from_scalar as infer_dtype_from_scalar, invalidate_string_dtypes as invalidate_string_dtypes, maybe_box_native as maybe_box_native, maybe_downcast_to_dtype as maybe_downcast_to_dtype
from pandas.core.dtypes.common import infer_dtype_from_object as infer_dtype_from_object, is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.inference import is_array_like as is_array_like, is_dataclass as is_dataclass, is_dict_like as is_dict_like, is_hashable as is_hashable, is_sequence as is_sequence
from pandas.core.dtypes.missing import isna as isna, notna as notna
from pandas.core.generic import NDFrame as NDFrame, make_doc as make_doc
from pandas.core.indexers.utils import check_key_length as check_key_length
from pandas.core.indexes.api import default_index as default_index
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index, ensure_index_from_sequences as ensure_index_from_sequences
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex, maybe_droplevels as maybe_droplevels
from pandas.core.indexes.period import PeriodIndex as PeriodIndex
from pandas.core.indexing import check_bool_indexer as check_bool_indexer, check_dict_or_set_indexers as check_dict_or_set_indexers
from pandas.core.internals.array_manager import ArrayManager as ArrayManager
from pandas.core.internals.construction import arrays_to_mgr as arrays_to_mgr, dataclasses_to_dicts as dataclasses_to_dicts, dict_to_mgr as dict_to_mgr, mgr_to_mgr as mgr_to_mgr, ndarray_to_mgr as ndarray_to_mgr, nested_data_to_arrays as nested_data_to_arrays, rec_array_to_mgr as rec_array_to_mgr, reorder_arrays as reorder_arrays, to_arrays as to_arrays, treat_as_nested as treat_as_nested
from pandas.core.internals.managers import BlockManager as BlockManager
from pandas.core.reshape.melt import melt as melt
from pandas.core.series import Series as Series
from pandas.core.sorting import get_group_index as get_group_index, lexsort_indexer as lexsort_indexer, nargsort as nargsort
from pandas.errors import ChainedAssignmentError as ChainedAssignmentError, InvalidIndexError as InvalidIndexError, LossySetitemError as LossySetitemError
from pandas.io.common import get_handle as get_handle
from pandas.io.formats.info import DataFrameInfo as DataFrameInfo
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level, rewrite_warning as rewrite_warning
from pandas.util._validators import validate_ascending as validate_ascending, validate_bool_kwarg as validate_bool_kwarg, validate_percentile as validate_percentile
from typing import Any, Callable, ClassVar, Literal

TYPE_CHECKING: bool
PYPY: bool
REF_COUNT: int
_chained_assignment_method_msg: str
_chained_assignment_msg: str
_chained_assignment_warning_method_msg: str
_chained_assignment_warning_msg: str
_shared_docs: dict
INFO_DOCSTRING: str
frame_sub_kwargs: dict
_shared_doc_kwargs: dict
_merge_doc: str

class DataFrame(pandas.core.generic.NDFrame, pandas.core.arraylike.OpsMixin):
    _internal_names_set: ClassVar[set] = ...
    _typ: ClassVar[str] = ...
    _HANDLED_TYPES: ClassVar[tuple] = ...
    _accessors: ClassVar[set] = ...
    _hidden_attrs: ClassVar[frozenset] = ...
    __pandas_priority__: ClassVar[int] = ...
    _constructor_sliced: ClassVar[type[pandas.core.series.Series]] = ...
    _agg_see_also_doc: ClassVar[str] = ...
    _agg_examples_doc: ClassVar[str] = ...
    _AXIS_ORDERS: ClassVar[list] = ...
    _AXIS_TO_AXIS_NUMBER: ClassVar[dict] = ...
    _AXIS_LEN: ClassVar[int] = ...
    _info_axis_number: ClassVar[int] = ...
    _info_axis_name: ClassVar[str] = ...
    plot: ClassVar[type[pandas.plotting._core.PlotAccessor]] = ...
    sparse: ClassVar[type[pandas.core.arrays.sparse.accessor.SparseFrameAccessor]] = ...
    index: Incomplete
    columns: Incomplete
    def _constructor_from_mgr(self, mgr, axes) -> DataFrame: ...
    def _constructor_sliced_from_mgr(self, mgr, axes) -> Series: ...
    def __init__(self, data, index: Axes | None, columns: Axes | None, dtype: Dtype | None, copy: bool | None) -> None: ...
    def __dataframe__(self, nan_as_null: bool = ..., allow_copy: bool = ...) -> DataFrameXchg:
        """
        Return the dataframe interchange object implementing the interchange protocol.

        Parameters
        ----------
        nan_as_null : bool, default False
            `nan_as_null` is DEPRECATED and has no effect. Please avoid using
            it; it will be removed in a future release.
        allow_copy : bool, default True
            Whether to allow memory copying when exporting. If set to False
            it would cause non-zero-copy exports to fail.

        Returns
        -------
        DataFrame interchange object
            The object which consuming library can use to ingress the dataframe.

        Notes
        -----
        Details on the interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html

        Examples
        --------
        >>> df_not_necessarily_pandas = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> interchange_object = df_not_necessarily_pandas.__dataframe__()
        >>> interchange_object.column_names()
        Index(['A', 'B'], dtype='object')
        >>> df_pandas = (pd.api.interchange.from_dataframe
        ...              (interchange_object.select_columns_by_name(['A'])))
        >>> df_pandas
             A
        0    1
        1    2

        These methods (``column_names``, ``select_columns_by_name``) should work
        for any dataframe library which implements the interchange protocol.
        """
    def __dataframe_consortium_standard__(self, *, api_version: str | None) -> Any:
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of pandas.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
    def __arrow_c_stream__(self, requested_schema):
        """
        Export the pandas DataFrame as an Arrow C stream PyCapsule.

        This relies on pyarrow to convert the pandas DataFrame to the Arrow
        format (and follows the default behaviour of ``pyarrow.Table.from_pandas``
        in its handling of the index, i.e. store the index as a column except
        for RangeIndex).
        This conversion is not necessarily zero-copy.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the dataframe should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.

        Returns
        -------
        PyCapsule
        """
    def _repr_fits_vertical_(self) -> bool:
        """
        Check length against max_rows.
        """
    def _repr_fits_horizontal_(self) -> bool:
        """
        Check if full repr fits in horizontal boundaries imposed by the display
        options width and max_columns.
        """
    def _info_repr(self) -> bool:
        """
        True if the repr should show the info view.
        """
    def _repr_html_(self) -> str | None:
        """
        Return a html representation for a particular DataFrame.

        Mainly for IPython notebook.
        """
    def to_string(self, buf: FilePath | WriteBuffer[str] | None, *, columns: Axes | None, col_space: int | list[int] | dict[Hashable, int] | None, header: bool | SequenceNotStr[str] = ..., index: bool = ..., na_rep: str = ..., formatters: fmt.FormattersType | None, float_format: fmt.FloatFormatType | None, sparsify: bool | None, index_names: bool = ..., justify: str | None, max_rows: int | None, max_cols: int | None, show_dimensions: bool = ..., decimal: str = ..., line_width: int | None, min_rows: int | None, max_colwidth: int | None, encoding: str | None) -> str | None:
        '''
        Render a DataFrame to a console-friendly tabular output.

                Parameters
                ----------
                buf : str, Path or StringIO-like, optional, default None
                    Buffer to write to. If None, the output is returned as a string.
                columns : array-like, optional, default None
                    The subset of columns to write. Writes all columns by default.
                col_space : int, list or dict of int, optional
                    The minimum width of each column. If a list of ints is given every integers corresponds with one column. If a dict is given, the key references the column, while the value defines the space to use..
                header : bool or list of str, optional
                    Write out the column names. If a list of columns is given, it is assumed to be aliases for the column names.
                index : bool, optional, default True
                    Whether to print index (row) labels.
                na_rep : str, optional, default \'NaN\'
                    String representation of ``NaN`` to use.
                formatters : list, tuple or dict of one-param. functions, optional
                    Formatter functions to apply to columns\' elements by position or
                    name.
                    The result of each function must be a unicode string.
                    List/tuple must be of length equal to the number of columns.
                float_format : one-parameter function, optional, default None
                    Formatter function to apply to columns\' elements if they are
                    floats. This function must return a unicode string and will be
                    applied only to the non-``NaN`` elements, with ``NaN`` being
                    handled by ``na_rep``.
                sparsify : bool, optional, default True
                    Set to False for a DataFrame with a hierarchical index to print
                    every multiindex key at each row.
                index_names : bool, optional, default True
                    Prints the names of the indexes.
                justify : str, default None
                    How to justify the column labels. If None uses the option from
                    the print configuration (controlled by set_option), \'right\' out
                    of the box. Valid values are

                    * left
                    * right
                    * center
                    * justify
                    * justify-all
                    * start
                    * end
                    * inherit
                    * match-parent
                    * initial
                    * unset.
                max_rows : int, optional
                    Maximum number of rows to display in the console.
                max_cols : int, optional
                    Maximum number of columns to display in the console.
                show_dimensions : bool, default False
                    Display DataFrame dimensions (number of rows by number of columns).
                decimal : str, default \'.\'
                    Character recognized as decimal separator, e.g. \',\' in Europe.
    
        line_width : int, optional
            Width to wrap a line in characters.
        min_rows : int, optional
            The number of rows to display in the console in a truncated repr
            (when number of rows is above `max_rows`).
        max_colwidth : int, optional
            Max width to truncate each column in characters. By default, no limit.
        encoding : str, default "utf-8"
            Set character encoding.

                Returns
                -------
                str or None
                    If buf is None, returns the result as a string. Otherwise returns
                    None.
    
        See Also
        --------
        to_html : Convert DataFrame to HTML.

        Examples
        --------
        >>> d = {\'col1\': [1, 2, 3], \'col2\': [4, 5, 6]}
        >>> df = pd.DataFrame(d)
        >>> print(df.to_string())
           col1  col2
        0     1     4
        1     2     5
        2     3     6
        '''
    def _get_values_for_csv(self, *, float_format: FloatFormatType | None, date_format: str | None, decimal: str, na_rep: str, quoting) -> Self: ...
    def items(self) -> Iterable[tuple[Hashable, Series]]:
        """
        Iterate over (column name, Series) pairs.

        Iterates over the DataFrame columns, returning a tuple with
        the column name and the content as a Series.

        Yields
        ------
        label : object
            The column names for the DataFrame being iterated over.
        content : Series
            The column entries belonging to each label, as a Series.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as
            (index, Series) pairs.
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples
            of the values.

        Examples
        --------
        >>> df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],
        ...                   'population': [1864, 22000, 80000]},
        ...                   index=['panda', 'polar', 'koala'])
        >>> df
                species   population
        panda   bear      1864
        polar   bear      22000
        koala   marsupial 80000
        >>> for label, content in df.items():
        ...     print(f'label: {label}')
        ...     print(f'content: {content}', sep='\\n')
        ...
        label: species
        content:
        panda         bear
        polar         bear
        koala    marsupial
        Name: species, dtype: object
        label: population
        content:
        panda     1864
        polar    22000
        koala    80000
        Name: population, dtype: int64
        """
    def iterrows(self) -> Iterable[tuple[Hashable, Series]]:
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        See Also
        --------
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples of the values.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames).

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.

        Examples
        --------

        >>> df = pd.DataFrame([[1, 1.5]], columns=['int', 'float'])
        >>> row = next(df.iterrows())[1]
        >>> row
        int      1.0
        float    1.5
        Name: 0, dtype: float64
        >>> print(row['int'].dtype)
        float64
        >>> print(df['int'].dtype)
        int64
        """
    def itertuples(self, index: bool = ..., name: str | None = ...) -> Iterable[tuple[Any, ...]]:
        '''
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str or None, default "Pandas"
            The name of the returned namedtuples or None to return regular
            tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)
            pairs.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        The column names will be renamed to positional names if they are
        invalid Python identifiers, repeated, or start with an underscore.

        Examples
        --------
        >>> df = pd.DataFrame({\'num_legs\': [4, 2], \'num_wings\': [0, 2]},
        ...                   index=[\'dog\', \'hawk\'])
        >>> df
              num_legs  num_wings
        dog          4          0
        hawk         2          2
        >>> for row in df.itertuples():
        ...     print(row)
        ...
        Pandas(Index=\'dog\', num_legs=4, num_wings=0)
        Pandas(Index=\'hawk\', num_legs=2, num_wings=2)

        By setting the `index` parameter to False we can remove the index
        as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        ...
        Pandas(num_legs=4, num_wings=0)
        Pandas(num_legs=2, num_wings=2)

        With the `name` parameter set we set a custom name for the yielded
        namedtuples:

        >>> for row in df.itertuples(name=\'Animal\'):
        ...     print(row)
        ...
        Animal(Index=\'dog\', num_legs=4, num_wings=0)
        Animal(Index=\'hawk\', num_legs=2, num_wings=2)
        '''
    def __len__(self) -> int:
        """
        Returns length of info axis, but here we use the index.
        """
    def dot(self, other: AnyArrayLike | DataFrame) -> DataFrame | Series:
        """
        Compute the matrix multiplication between the DataFrame and other.

        This method computes the matrix product between the DataFrame and the
        values of an other Series, DataFrame or a numpy array.

        It can also be called using ``self @ other``.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the matrix product with.

        Returns
        -------
        Series or DataFrame
            If other is a Series, return the matrix product between self and
            other as a Series. If other is a DataFrame or a numpy.array, return
            the matrix product of self and other in a DataFrame of a np.array.

        See Also
        --------
        Series.dot: Similar method for Series.

        Notes
        -----
        The dimensions of DataFrame and other must be compatible in order to
        compute the matrix multiplication. In addition, the column names of
        DataFrame and the index of other must contain the same values, as they
        will be aligned prior to the multiplication.

        The dot method for Series computes the inner product, instead of the
        matrix product here.

        Examples
        --------
        Here we multiply a DataFrame with a Series.

        >>> df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        >>> s = pd.Series([1, 1, 2, 1])
        >>> df.dot(s)
        0    -4
        1     5
        dtype: int64

        Here we multiply a DataFrame with another DataFrame.

        >>> other = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
        >>> df.dot(other)
            0   1
        0   1   4
        1   2   2

        Note that the dot method give the same result as @

        >>> df @ other
            0   1
        0   1   4
        1   2   2

        The dot method works also if other is an np.array.

        >>> arr = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
        >>> df.dot(arr)
            0   1
        0   1   4
        1   2   2

        Note how shuffling of the objects does not change the result.

        >>> s2 = s.reindex([1, 0, 2, 3])
        >>> df.dot(s2)
        0    -4
        1     5
        dtype: int64
        """
    def __matmul__(self, other: AnyArrayLike | DataFrame) -> DataFrame | Series:
        """
        Matrix multiplication using binary `@` operator.
        """
    def __rmatmul__(self, other) -> DataFrame:
        """
        Matrix multiplication using binary `@` operator.
        """
    @classmethod
    def from_dict(cls, data: dict, orient: FromDictOrient = ..., dtype: Dtype | None, columns: Axes | None) -> DataFrame:
        '''
        Construct DataFrame from dict of array-like or dicts.

        Creates DataFrame object from dictionary by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {\'columns\', \'index\', \'tight\'}, default \'columns\'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass \'columns\'
            (default). Otherwise if the keys should be rows, pass \'index\'.
            If \'tight\', assume a dict with keys [\'index\', \'columns\', \'data\',
            \'index_names\', \'column_names\'].

            .. versionadded:: 1.4.0
               \'tight\' as an allowed value for the ``orient`` argument

        dtype : dtype, default None
            Data type to force after DataFrame construction, otherwise infer.
        columns : list, default None
            Column labels to use when ``orient=\'index\'``. Raises a ValueError
            if used with ``orient=\'columns\'`` or ``orient=\'tight\'``.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_records : DataFrame from structured ndarray, sequence
            of tuples or dicts, or DataFrame.
        DataFrame : DataFrame object creation using constructor.
        DataFrame.to_dict : Convert the DataFrame to a dictionary.

        Examples
        --------
        By default the keys of the dict become the DataFrame columns:

        >>> data = {\'col_1\': [3, 2, 1, 0], \'col_2\': [\'a\', \'b\', \'c\', \'d\']}
        >>> pd.DataFrame.from_dict(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Specify ``orient=\'index\'`` to create the DataFrame using dictionary
        keys as rows:

        >>> data = {\'row_1\': [3, 2, 1, 0], \'row_2\': [\'a\', \'b\', \'c\', \'d\']}
        >>> pd.DataFrame.from_dict(data, orient=\'index\')
               0  1  2  3
        row_1  3  2  1  0
        row_2  a  b  c  d

        When using the \'index\' orientation, the column names can be
        specified manually:

        >>> pd.DataFrame.from_dict(data, orient=\'index\',
        ...                        columns=[\'A\', \'B\', \'C\', \'D\'])
               A  B  C  D
        row_1  3  2  1  0
        row_2  a  b  c  d

        Specify ``orient=\'tight\'`` to create the DataFrame using a \'tight\'
        format:

        >>> data = {\'index\': [(\'a\', \'b\'), (\'a\', \'c\')],
        ...         \'columns\': [(\'x\', 1), (\'y\', 2)],
        ...         \'data\': [[1, 3], [2, 4]],
        ...         \'index_names\': [\'n1\', \'n2\'],
        ...         \'column_names\': [\'z1\', \'z2\']}
        >>> pd.DataFrame.from_dict(data, orient=\'tight\')
        z1     x  y
        z2     1  2
        n1 n2
        a  b   1  3
           c   2  4
        '''
    def to_numpy(self, dtype: npt.DTypeLike | None, copy: bool = ..., na_value: object = ...) -> np.ndarray:
        '''
        Convert the DataFrame to a NumPy array.

        By default, the dtype of the returned array will be the common NumPy
        dtype of all types in the DataFrame. For example, if the dtypes are
        ``float16`` and ``float32``, the results dtype will be ``float32``.
        This may require copying data and coercing values, which may be
        expensive.

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
            on `dtype` and the dtypes of the DataFrame columns.

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
        array([[1, 3],
               [2, 4]])

        With heterogeneous data, the lowest common type will have to
        be used.

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
        >>> df.to_numpy()
        array([[1. , 3. ],
               [2. , 4.5]])

        For a mix of numeric and non-numeric types, the output array will
        have object dtype.

        >>> df[\'C\'] = pd.date_range(\'2000\', periods=2)
        >>> df.to_numpy()
        array([[1, 3.0, Timestamp(\'2000-01-01 00:00:00\')],
               [2, 4.5, Timestamp(\'2000-01-02 00:00:00\')]], dtype=object)
        '''
    def _create_data_for_split_and_tight_to_dict(self, are_all_object_dtype_cols: bool, object_dtype_indices: list[int]) -> list:
        '''
        Simple helper method to create data for to ``to_dict(orient="split")`` and
        ``to_dict(orient="tight")`` to create the main output data
        '''
    def to_dict(self, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'records', 'index'] = ..., *, into: type[MutableMappingT] | MutableMappingT = ..., index: bool = ...) -> MutableMappingT | list[MutableMappingT]:
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters
        (see below).

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'tight' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
              'index_names' -> [index.names], 'column_names' -> [column.names]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

            .. versionadded:: 1.4.0
                'tight' as an allowed value for the ``orient`` argument

        into : class, default dict
            The collections.abc.MutableMapping subclass used for all Mappings
            in the return value.  Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        index : bool, default True
            Whether to include the index item (and index_names item if `orient`
            is 'tight') in the returned dictionary. Can only be ``False``
            when `orient` is 'split' or 'tight'.

            .. versionadded:: 2.0.0

        Returns
        -------
        dict, list or collections.abc.MutableMapping
            Return a collections.abc.MutableMapping object representing the
            DataFrame. The resulting transformation depends on the `orient`
            parameter.

        See Also
        --------
        DataFrame.from_dict: Create a DataFrame from a dictionary.
        DataFrame.to_json: Convert a DataFrame to JSON format.

        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2],
        ...                    'col2': [0.5, 0.75]},
        ...                   index=['row1', 'row2'])
        >>> df
              col1  col2
        row1     1  0.50
        row2     2  0.75
        >>> df.to_dict()
        {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}

        You can specify the return orientation.

        >>> df.to_dict('series')
        {'col1': row1    1
                 row2    2
        Name: col1, dtype: int64,
        'col2': row1    0.50
                row2    0.75
        Name: col2, dtype: float64}

        >>> df.to_dict('split')
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]]}

        >>> df.to_dict('records')
        [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]

        >>> df.to_dict('index')
        {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}

        >>> df.to_dict('tight')
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]], 'index_names': [None], 'column_names': [None]}

        You can also specify the mapping type.

        >>> from collections import OrderedDict, defaultdict
        >>> df.to_dict(into=OrderedDict)
        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])),
                     ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

        If you want a `defaultdict`, you need to initialize it:

        >>> dd = defaultdict(list)
        >>> df.to_dict('records', into=dd)
        [defaultdict(<class 'list'>, {'col1': 1, 'col2': 0.5}),
         defaultdict(<class 'list'>, {'col1': 2, 'col2': 0.75})]
        """
    def to_gbq(self, destination_table: str, *, project_id: str | None, chunksize: int | None, reauth: bool = ..., if_exists: ToGbqIfexist = ..., auth_local_webserver: bool = ..., table_schema: list[dict[str, str]] | None, location: str | None, progress_bar: bool = ..., credentials) -> None:
        '''
        Write a DataFrame to a Google BigQuery table.

        .. deprecated:: 2.2.0

           Please use ``pandas_gbq.to_gbq`` instead.

        This function requires the `pandas-gbq package
        <https://pandas-gbq.readthedocs.io>`__.

        See the `How to authenticate with Google BigQuery
        <https://pandas-gbq.readthedocs.io/en/latest/howto/authentication.html>`__
        guide for authentication instructions.

        Parameters
        ----------
        destination_table : str
            Name of table to be written, in the form ``dataset.tablename``.
        project_id : str, optional
            Google BigQuery Account project ID. Optional when available from
            the environment.
        chunksize : int, optional
            Number of rows to be inserted in each chunk from the dataframe.
            Set to ``None`` to load the whole dataframe at once.
        reauth : bool, default False
            Force Google BigQuery to re-authenticate the user. This is useful
            if multiple accounts are used.
        if_exists : str, default \'fail\'
            Behavior when the destination table exists. Value can be one of:

            ``\'fail\'``
                If table exists raise pandas_gbq.gbq.TableCreationError.
            ``\'replace\'``
                If table exists, drop it, recreate it, and insert data.
            ``\'append\'``
                If table exists, insert data. Create if does not exist.
        auth_local_webserver : bool, default True
            Use the `local webserver flow`_ instead of the `console flow`_
            when getting user credentials.

            .. _local webserver flow:
                https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_local_server
            .. _console flow:
                https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_console

            *New in version 0.2.0 of pandas-gbq*.

            .. versionchanged:: 1.5.0
               Default value is changed to ``True``. Google has deprecated the
               ``auth_local_webserver = False`` `"out of band" (copy-paste)
               flow
               <https://developers.googleblog.com/2022/02/making-oauth-flows-safer.html?m=1#disallowed-oob>`_.
        table_schema : list of dicts, optional
            List of BigQuery table fields to which according DataFrame
            columns conform to, e.g. ``[{\'name\': \'col1\', \'type\':
            \'STRING\'},...]``. If schema is not provided, it will be
            generated according to dtypes of DataFrame columns. See
            BigQuery API documentation on available names of a field.

            *New in version 0.3.1 of pandas-gbq*.
        location : str, optional
            Location where the load job should run. See the `BigQuery locations
            documentation
            <https://cloud.google.com/bigquery/docs/dataset-locations>`__ for a
            list of available locations. The location must match that of the
            target dataset.

            *New in version 0.5.0 of pandas-gbq*.
        progress_bar : bool, default True
            Use the library `tqdm` to show the progress bar for the upload,
            chunk by chunk.

            *New in version 0.5.0 of pandas-gbq*.
        credentials : google.auth.credentials.Credentials, optional
            Credentials for accessing Google APIs. Use this parameter to
            override default credentials, such as to use Compute Engine
            :class:`google.auth.compute_engine.Credentials` or Service
            Account :class:`google.oauth2.service_account.Credentials`
            directly.

            *New in version 0.8.0 of pandas-gbq*.

        See Also
        --------
        pandas_gbq.to_gbq : This function in the pandas-gbq library.
        read_gbq : Read a DataFrame from Google BigQuery.

        Examples
        --------
        Example taken from `Google BigQuery documentation
        <https://cloud.google.com/bigquery/docs/samples/bigquery-pandas-gbq-to-gbq-simple>`_

        >>> project_id = "my-project"
        >>> table_id = \'my_dataset.my_table\'
        >>> df = pd.DataFrame({
        ...                   "my_string": ["a", "b", "c"],
        ...                   "my_int64": [1, 2, 3],
        ...                   "my_float64": [4.0, 5.0, 6.0],
        ...                   "my_bool1": [True, False, True],
        ...                   "my_bool2": [False, True, False],
        ...                   "my_dates": pd.date_range("now", periods=3),
        ...                   }
        ...                   )

        >>> df.to_gbq(table_id, project_id=project_id)  # doctest: +SKIP
        '''
    @classmethod
    def from_records(cls, data, index, exclude, columns, coerce_float: bool = ..., nrows: int | None) -> DataFrame:
        """
        Convert structured or record ndarray to DataFrame.

        Creates a DataFrame object from a structured ndarray, sequence of
        tuples or dicts, or DataFrame.

        Parameters
        ----------
        data : structured ndarray, sequence of tuples or dicts, or DataFrame
            Structured input data.

            .. deprecated:: 2.1.0
                Passing a DataFrame is deprecated.
        index : str, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use.
        exclude : sequence, default None
            Columns or fields to exclude.
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns).
        coerce_float : bool, default False
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        nrows : int, default None
            Number of rows to read if data is an iterator.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_dict : DataFrame from dict of array-like or dicts.
        DataFrame : DataFrame object creation using constructor.

        Examples
        --------
        Data can be provided as a structured ndarray:

        >>> data = np.array([(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')],
        ...                 dtype=[('col_1', 'i4'), ('col_2', 'U1')])
        >>> pd.DataFrame.from_records(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Data can be provided as a list of dicts:

        >>> data = [{'col_1': 3, 'col_2': 'a'},
        ...         {'col_1': 2, 'col_2': 'b'},
        ...         {'col_1': 1, 'col_2': 'c'},
        ...         {'col_1': 0, 'col_2': 'd'}]
        >>> pd.DataFrame.from_records(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Data can be provided as a list of tuples with corresponding columns:

        >>> data = [(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')]
        >>> pd.DataFrame.from_records(data, columns=['col_1', 'col_2'])
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d
        """
    def to_records(self, index: bool = ..., column_dtypes, index_dtypes) -> np.rec.recarray:
        '''
        Convert DataFrame to a NumPy record array.

        Index will be included as the first field of the record array if
        requested.

        Parameters
        ----------
        index : bool, default True
            Include index in resulting record array, stored in \'index\'
            field or using the index label, if set.
        column_dtypes : str, type, dict, default None
            If a string or type, the data type to store all columns. If
            a dictionary, a mapping of column names and indices (zero-indexed)
            to specific data types.
        index_dtypes : str, type, dict, default None
            If a string or type, the data type to store all index levels. If
            a dictionary, a mapping of index level names and indices
            (zero-indexed) to specific data types.

            This mapping is applied only if `index=True`.

        Returns
        -------
        numpy.rec.recarray
            NumPy ndarray with the DataFrame labels as fields and each row
            of the DataFrame as entries.

        See Also
        --------
        DataFrame.from_records: Convert structured or record ndarray
            to DataFrame.
        numpy.rec.recarray: An ndarray that allows field access using
            attributes, analogous to typed columns in a
            spreadsheet.

        Examples
        --------
        >>> df = pd.DataFrame({\'A\': [1, 2], \'B\': [0.5, 0.75]},
        ...                   index=[\'a\', \'b\'])
        >>> df
           A     B
        a  1  0.50
        b  2  0.75
        >>> df.to_records()
        rec.array([(\'a\', 1, 0.5 ), (\'b\', 2, 0.75)],
                  dtype=[(\'index\', \'O\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])

        If the DataFrame index has no label then the recarray field name
        is set to \'index\'. If the index has a label then this is used as the
        field name:

        >>> df.index = df.index.rename("I")
        >>> df.to_records()
        rec.array([(\'a\', 1, 0.5 ), (\'b\', 2, 0.75)],
                  dtype=[(\'I\', \'O\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])

        The index can be excluded from the record array:

        >>> df.to_records(index=False)
        rec.array([(1, 0.5 ), (2, 0.75)],
                  dtype=[(\'A\', \'<i8\'), (\'B\', \'<f8\')])

        Data types can be specified for the columns:

        >>> df.to_records(column_dtypes={"A": "int32"})
        rec.array([(\'a\', 1, 0.5 ), (\'b\', 2, 0.75)],
                  dtype=[(\'I\', \'O\'), (\'A\', \'<i4\'), (\'B\', \'<f8\')])

        As well as for the index:

        >>> df.to_records(index_dtypes="<S2")
        rec.array([(b\'a\', 1, 0.5 ), (b\'b\', 2, 0.75)],
                  dtype=[(\'I\', \'S2\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])

        >>> index_dtypes = f"<S{df.index.str.len().max()}"
        >>> df.to_records(index_dtypes=index_dtypes)
        rec.array([(b\'a\', 1, 0.5 ), (b\'b\', 2, 0.75)],
                  dtype=[(\'I\', \'S1\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])
        '''
    @classmethod
    def _from_arrays(cls, arrays, columns, index, dtype: Dtype | None, verify_integrity: bool = ...) -> Self:
        """
        Create DataFrame from a list of arrays corresponding to the columns.

        Parameters
        ----------
        arrays : list-like of arrays
            Each array in the list corresponds to one column, in order.
        columns : list-like, Index
            The column names for the resulting DataFrame.
        index : list-like, Index
            The rows labels for the resulting DataFrame.
        dtype : dtype, optional
            Optional dtype to enforce for all arrays.
        verify_integrity : bool, default True
            Validate and homogenize all input. If set to False, it is assumed
            that all elements of `arrays` are actual arrays how they will be
            stored in a block (numpy ndarray or ExtensionArray), have the same
            length as and are aligned with the index, and that `columns` and
            `index` are ensured to be an Index object.

        Returns
        -------
        DataFrame
        """
    def to_stata(self, path: FilePath | WriteBuffer[bytes], *, convert_dates: dict[Hashable, str] | None, write_index: bool = ..., byteorder: ToStataByteorder | None, time_stamp: datetime.datetime | None, data_label: str | None, variable_labels: dict[Hashable, str] | None, version: int | None = ..., convert_strl: Sequence[Hashable] | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None, value_labels: dict[Hashable, dict[float, str]] | None) -> None:
        '''
        Export DataFrame object to Stata dta format.

        Writes the DataFrame to a Stata dataset file.
        "dta" files contain a Stata dataset.

        Parameters
        ----------
        path : str, path object, or buffer
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function.

        convert_dates : dict
            Dictionary mapping columns containing datetime types to stata
            internal format to use when writing the dates. Options are \'tc\',
            \'td\', \'tm\', \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer
            or a name. Datetime columns that do not have a conversion type
            specified will be converted to \'tc\'. Raises NotImplementedError if
            a datetime column has timezone information.
        write_index : bool
            Write the index to Stata dataset.
        byteorder : str
            Can be ">", "<", "little", or "big". default is `sys.byteorder`.
        time_stamp : datetime
            A datetime to use as file creation date.  Default is the current
            time.
        data_label : str, optional
            A label for the data set.  Must be 80 characters or smaller.
        variable_labels : dict
            Dictionary containing columns as keys and variable labels as
            values. Each label must be 80 characters or smaller.
        version : {114, 117, 118, 119, None}, default 114
            Version to use in the output dta file. Set to None to let pandas
            decide between 118 or 119 formats depending on the number of
            columns in the frame. Version 114 can be read by Stata 10 and
            later. Version 117 can be read by Stata 13 or later. Version 118
            is supported in Stata 14 and later. Version 119 is supported in
            Stata 15 and later. Version 114 limits string variables to 244
            characters or fewer while versions 117 and later allow strings
            with lengths up to 2,000,000 characters. Versions 118 and 119
            support Unicode characters, and version 119 supports more than
            32,767 variables.

            Version 119 should usually only be used when the number of
            variables exceeds the capacity of dta format 118. Exporting
            smaller datasets in format 119 may have unintended consequences,
            and, as of November 2020, Stata SE cannot read version 119 files.

        convert_strl : list, optional
            List of column names to convert to string columns to Stata StrL
            format. Only available if version is 117.  Storing strings in the
            StrL format can produce smaller dta files if strings have more than
            8 characters and values are repeated.
        compression : str or dict, default \'infer\'
            For on-the-fly compression of the output data. If \'infer\' and \'path\' is
            path-like, then detect compression from the following extensions: \'.gz\',
            \'.bz2\', \'.zip\', \'.xz\', \'.zst\', \'.tar\', \'.tar.gz\', \'.tar.xz\' or \'.tar.bz2\'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``\'method\'`` set
            to one of {``\'zip\'``, ``\'gzip\'``, ``\'bz2\'``, ``\'zstd\'``, ``\'xz\'``, ``\'tar\'``} and
            other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and to create
            a reproducible gzip archive:
            ``compression={\'method\': \'gzip\', \'compresslevel\': 1, \'mtime\': 1}``.

            .. versionadded:: 1.5.0
                Added support for `.tar` files.

            .. versionchanged:: 1.4.0 Zstandard support.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        value_labels : dict of dicts
            Dictionary containing columns as keys and dictionaries of column value
            to labels as values. Labels for a single variable must be 32,000
            characters or smaller.

            .. versionadded:: 1.4.0

        Raises
        ------
        NotImplementedError
            * If datetimes contain timezone information
            * Column dtype is not representable in Stata
        ValueError
            * Columns listed in convert_dates are neither datetime64[ns]
              or datetime.datetime
            * Column listed in convert_dates is not in DataFrame
            * Categorical label contains more than 32,000 characters

        See Also
        --------
        read_stata : Import Stata data files.
        io.stata.StataWriter : Low-level writer for Stata data files.
        io.stata.StataWriter117 : Low-level writer for version 117 files.

        Examples
        --------
        >>> df = pd.DataFrame({\'animal\': [\'falcon\', \'parrot\', \'falcon\',
        ...                               \'parrot\'],
        ...                    \'speed\': [350, 18, 361, 15]})
        >>> df.to_stata(\'animals.dta\')  # doctest: +SKIP
        '''
    def to_feather(self, path: FilePath | WriteBuffer[bytes], **kwargs) -> None:
        '''
        Write a DataFrame to the binary Feather format.

        Parameters
        ----------
        path : str, path object, file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. If a string or a path,
            it will be used as Root Directory path when writing a partitioned dataset.
        **kwargs :
            Additional keywords passed to :func:`pyarrow.feather.write_feather`.
            This includes the `compression`, `compression_level`, `chunksize`
            and `version` keywords.

        Notes
        -----
        This function writes the dataframe as a `feather file
        <https://arrow.apache.org/docs/python/feather.html>`_. Requires a default
        index. For saving the DataFrame with your custom index use a method that
        supports custom indices e.g. `to_parquet`.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        >>> df.to_feather("file.feather")  # doctest: +SKIP
        '''
    def to_markdown(self, buf: FilePath | WriteBuffer[str] | None, *, mode: str = ..., index: bool = ..., storage_options: StorageOptions | None, **kwargs) -> str | None:
        '''
        Print DataFrame in Markdown-friendly format.

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
            DataFrame in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        Examples
                --------
                >>> df = pd.DataFrame(
                ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
                ... )
                >>> print(df.to_markdown())
                |    | animal_1   | animal_2   |
                |---:|:-----------|:-----------|
                |  0 | elk        | dog        |
                |  1 | pig        | quetzal    |

                Output markdown with a tabulate option.

                >>> print(df.to_markdown(tablefmt="grid"))
                +----+------------+------------+
                |    | animal_1   | animal_2   |
                +====+============+============+
                |  0 | elk        | dog        |
                +----+------------+------------+
                |  1 | pig        | quetzal    |
                +----+------------+------------+
        '''
    def to_parquet(self, path: FilePath | WriteBuffer[bytes] | None, *, engine: Literal['auto', 'pyarrow', 'fastparquet'] = ..., compression: str | None = ..., index: bool | None, partition_cols: list[str] | None, storage_options: StorageOptions | None, **kwargs) -> bytes | None:
        '''
        Write a DataFrame to the binary parquet format.

        This function writes the dataframe as a `parquet file
        <https://parquet.apache.org/>`_. You can choose different parquet
        backends, and have the option of compression. See
        :ref:`the user guide <io.parquet>` for more details.

        Parameters
        ----------
        path : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. If None, the result is
            returned as bytes. If a string or path, it will be used as Root Directory
            path when writing a partitioned dataset.
        engine : {\'auto\', \'pyarrow\', \'fastparquet\'}, default \'auto\'
            Parquet library to use. If \'auto\', then the option
            ``io.parquet.engine`` is used. The default ``io.parquet.engine``
            behavior is to try \'pyarrow\', falling back to \'fastparquet\' if
            \'pyarrow\' is unavailable.
        compression : str or None, default \'snappy\'
            Name of the compression to use. Use ``None`` for no compression.
            Supported options: \'snappy\', \'gzip\', \'brotli\', \'lz4\', \'zstd\'.
        index : bool, default None
            If ``True``, include the dataframe\'s index(es) in the file output.
            If ``False``, they will not be written to the file.
            If ``None``, similar to ``True`` the dataframe\'s index(es)
            will be saved. However, instead of being saved as values,
            the RangeIndex will be stored as a range in the metadata so it
            doesn\'t require much space and is faster. Other indexes will
            be included as columns in the file output.
        partition_cols : list, optional, default None
            Column names by which to partition the dataset.
            Columns are partitioned in the order they are given.
            Must be None if path is not a string.
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
            Additional arguments passed to the parquet library. See
            :ref:`pandas io <io.parquet>` for more details.

        Returns
        -------
        bytes if no path argument is provided else None

        See Also
        --------
        read_parquet : Read a parquet file.
        DataFrame.to_orc : Write an orc file.
        DataFrame.to_csv : Write a csv file.
        DataFrame.to_sql : Write to a sql table.
        DataFrame.to_hdf : Write to hdf.

        Notes
        -----
        This function requires either the `fastparquet
        <https://pypi.org/project/fastparquet>`_ or `pyarrow
        <https://arrow.apache.org/docs/python/>`_ library.

        Examples
        --------
        >>> df = pd.DataFrame(data={\'col1\': [1, 2], \'col2\': [3, 4]})
        >>> df.to_parquet(\'df.parquet.gzip\',
        ...               compression=\'gzip\')  # doctest: +SKIP
        >>> pd.read_parquet(\'df.parquet.gzip\')  # doctest: +SKIP
           col1  col2
        0     1     3
        1     2     4

        If you want to get a buffer to the parquet content you can use a io.BytesIO
        object, as long as you don\'t use partition_cols, which creates multiple files.

        >>> import io
        >>> f = io.BytesIO()
        >>> df.to_parquet(f)
        >>> f.seek(0)
        0
        >>> content = f.read()
        '''
    def to_orc(self, path: FilePath | WriteBuffer[bytes] | None, *, engine: Literal['pyarrow'] = ..., index: bool | None, engine_kwargs: dict[str, Any] | None) -> bytes | None:
        """
        Write a DataFrame to the ORC format.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        path : str, file-like object or None, default None
            If a string, it will be used as Root Directory path
            when writing a partitioned dataset. By file-like object,
            we refer to objects with a write() method, such as a file handle
            (e.g. via builtin open function). If path is None,
            a bytes object is returned.
        engine : {'pyarrow'}, default 'pyarrow'
            ORC library to use.
        index : bool, optional
            If ``True``, include the dataframe's index(es) in the file output.
            If ``False``, they will not be written to the file.
            If ``None``, similar to ``infer`` the dataframe's index(es)
            will be saved. However, instead of being saved as values,
            the RangeIndex will be stored as a range in the metadata so it
            doesn't require much space and is faster. Other indexes will
            be included as columns in the file output.
        engine_kwargs : dict[str, Any] or None, default None
            Additional keyword arguments passed to :func:`pyarrow.orc.write_table`.

        Returns
        -------
        bytes if no path argument is provided else None

        Raises
        ------
        NotImplementedError
            Dtype of one or more columns is category, unsigned integers, interval,
            period or sparse.
        ValueError
            engine is not pyarrow.

        See Also
        --------
        read_orc : Read a ORC file.
        DataFrame.to_parquet : Write a parquet file.
        DataFrame.to_csv : Write a csv file.
        DataFrame.to_sql : Write to a sql table.
        DataFrame.to_hdf : Write to hdf.

        Notes
        -----
        * Before using this function you should read the :ref:`user guide about
          ORC <io.orc>` and :ref:`install optional dependencies <install.warn_orc>`.
        * This function requires `pyarrow <https://arrow.apache.org/docs/python/>`_
          library.
        * For supported dtypes please refer to `supported ORC features in Arrow
          <https://arrow.apache.org/docs/cpp/orc.html#data-types>`__.
        * Currently timezones in datetime columns are not preserved when a
          dataframe is converted into ORC files.

        Examples
        --------
        >>> df = pd.DataFrame(data={'col1': [1, 2], 'col2': [4, 3]})
        >>> df.to_orc('df.orc')  # doctest: +SKIP
        >>> pd.read_orc('df.orc')  # doctest: +SKIP
           col1  col2
        0     1     4
        1     2     3

        If you want to get a buffer to the orc content you can write it to io.BytesIO

        >>> import io
        >>> b = io.BytesIO(df.to_orc())  # doctest: +SKIP
        >>> b.seek(0)  # doctest: +SKIP
        0
        >>> content = b.read()  # doctest: +SKIP
        """
    def to_html(self, buf: FilePath | WriteBuffer[str] | None, *, columns: Axes | None, col_space: ColspaceArgType | None, header: bool = ..., index: bool = ..., na_rep: str = ..., formatters: FormattersType | None, float_format: FloatFormatType | None, sparsify: bool | None, index_names: bool = ..., justify: str | None, max_rows: int | None, max_cols: int | None, show_dimensions: bool | str = ..., decimal: str = ..., bold_rows: bool = ..., classes: str | list | tuple | None, escape: bool = ..., notebook: bool = ..., border: int | bool | None, table_id: str | None, render_links: bool = ..., encoding: str | None) -> str | None:
        '''
        Render a DataFrame as an HTML table.

                Parameters
                ----------
                buf : str, Path or StringIO-like, optional, default None
                    Buffer to write to. If None, the output is returned as a string.
                columns : array-like, optional, default None
                    The subset of columns to write. Writes all columns by default.
                col_space : str or int, list or dict of int or str, optional
                    The minimum width of each column in CSS length units.  An int is assumed to be px units..
                header : bool, optional
                    Whether to print column labels, default True.
                index : bool, optional, default True
                    Whether to print index (row) labels.
                na_rep : str, optional, default \'NaN\'
                    String representation of ``NaN`` to use.
                formatters : list, tuple or dict of one-param. functions, optional
                    Formatter functions to apply to columns\' elements by position or
                    name.
                    The result of each function must be a unicode string.
                    List/tuple must be of length equal to the number of columns.
                float_format : one-parameter function, optional, default None
                    Formatter function to apply to columns\' elements if they are
                    floats. This function must return a unicode string and will be
                    applied only to the non-``NaN`` elements, with ``NaN`` being
                    handled by ``na_rep``.
                sparsify : bool, optional, default True
                    Set to False for a DataFrame with a hierarchical index to print
                    every multiindex key at each row.
                index_names : bool, optional, default True
                    Prints the names of the indexes.
                justify : str, default None
                    How to justify the column labels. If None uses the option from
                    the print configuration (controlled by set_option), \'right\' out
                    of the box. Valid values are

                    * left
                    * right
                    * center
                    * justify
                    * justify-all
                    * start
                    * end
                    * inherit
                    * match-parent
                    * initial
                    * unset.
                max_rows : int, optional
                    Maximum number of rows to display in the console.
                max_cols : int, optional
                    Maximum number of columns to display in the console.
                show_dimensions : bool, default False
                    Display DataFrame dimensions (number of rows by number of columns).
                decimal : str, default \'.\'
                    Character recognized as decimal separator, e.g. \',\' in Europe.
    
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            `<table>` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        encoding : str, default "utf-8"
            Set character encoding.

                Returns
                -------
                str or None
                    If buf is None, returns the result as a string. Otherwise returns
                    None.
    
        See Also
        --------
        to_string : Convert DataFrame to a string.

        Examples
        --------
        >>> df = pd.DataFrame(data={\'col1\': [1, 2], \'col2\': [4, 3]})
        >>> html_string = \'\'\'<table border="1" class="dataframe">
        ...   <thead>
        ...     <tr style="text-align: right;">
        ...       <th></th>
        ...       <th>col1</th>
        ...       <th>col2</th>
        ...     </tr>
        ...   </thead>
        ...   <tbody>
        ...     <tr>
        ...       <th>0</th>
        ...       <td>1</td>
        ...       <td>4</td>
        ...     </tr>
        ...     <tr>
        ...       <th>1</th>
        ...       <td>2</td>
        ...       <td>3</td>
        ...     </tr>
        ...   </tbody>
        ... </table>\'\'\'
        >>> assert html_string == df.to_html()
        '''
    def to_xml(self, path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None, *, index: bool = ..., root_name: str | None = ..., row_name: str | None = ..., na_rep: str | None, attr_cols: list[str] | None, elem_cols: list[str] | None, namespaces: dict[str | None, str] | None, prefix: str | None, encoding: str = ..., xml_declaration: bool | None = ..., pretty_print: bool | None = ..., parser: XMLParsers | None = ..., stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None) -> str | None:
        '''
        Render a DataFrame to an XML document.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        path_or_buffer : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a ``write()`` function. If None, the result is returned
            as a string.
        index : bool, default True
            Whether to include index in XML document.
        root_name : str, default \'data\'
            The name of root element in XML document.
        row_name : str, default \'row\'
            The name of row element in XML document.
        na_rep : str, optional
            Missing data representation.
        attr_cols : list-like, optional
            List of columns to write as attributes in row element.
            Hierarchical columns will be flattened with underscore
            delimiting the different levels.
        elem_cols : list-like, optional
            List of columns to write as children in row element. By default,
            all columns output as children of row element. Hierarchical
            columns will be flattened with underscore delimiting the
            different levels.
        namespaces : dict, optional
            All namespaces to be defined in root element. Keys of dict
            should be prefix names and values of dict corresponding URIs.
            Default namespaces should be given empty string key. For
            example, ::

                namespaces = {"": "https://example.com"}

        prefix : str, optional
            Namespace prefix to be used for every element and/or attribute
            in document. This should be one of the keys in ``namespaces``
            dict.
        encoding : str, default \'utf-8\'
            Encoding of the resulting document.
        xml_declaration : bool, default True
            Whether to include the XML declaration at start of document.
        pretty_print : bool, default True
            Whether output should be pretty printed with indentation and
            line breaks.
        parser : {\'lxml\',\'etree\'}, default \'lxml\'
            Parser module to use for building of tree. Only \'lxml\' and
            \'etree\' are supported. With \'lxml\', the ability to use XSLT
            stylesheet is supported.
        stylesheet : str, path object or file-like object, optional
            A URL, file-like object, or a raw string containing an XSLT
            script used to transform the raw XML output. Script should use
            layout of elements and attributes from original output. This
            argument requires ``lxml`` to be installed. Only XSLT 1.0
            scripts and not later versions is currently supported.
        compression : str or dict, default \'infer\'
            For on-the-fly compression of the output data. If \'infer\' and \'path_or_buffer\' is
            path-like, then detect compression from the following extensions: \'.gz\',
            \'.bz2\', \'.zip\', \'.xz\', \'.zst\', \'.tar\', \'.tar.gz\', \'.tar.xz\' or \'.tar.bz2\'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``\'method\'`` set
            to one of {``\'zip\'``, ``\'gzip\'``, ``\'bz2\'``, ``\'zstd\'``, ``\'xz\'``, ``\'tar\'``} and
            other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and to create
            a reproducible gzip archive:
            ``compression={\'method\': \'gzip\', \'compresslevel\': 1, \'mtime\': 1}``.

            .. versionadded:: 1.5.0
                Added support for `.tar` files.

            .. versionchanged:: 1.4.0 Zstandard support.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        Returns
        -------
        None or str
            If ``io`` is None, returns the resulting XML format as a
            string. Otherwise returns None.

        See Also
        --------
        to_json : Convert the pandas object to a JSON string.
        to_html : Convert DataFrame to a html.

        Examples
        --------
        >>> df = pd.DataFrame({\'shape\': [\'square\', \'circle\', \'triangle\'],
        ...                    \'degrees\': [360, 360, 180],
        ...                    \'sides\': [4, np.nan, 3]})

        >>> df.to_xml()  # doctest: +SKIP
        <?xml version=\'1.0\' encoding=\'utf-8\'?>
        <data>
          <row>
            <index>0</index>
            <shape>square</shape>
            <degrees>360</degrees>
            <sides>4.0</sides>
          </row>
          <row>
            <index>1</index>
            <shape>circle</shape>
            <degrees>360</degrees>
            <sides/>
          </row>
          <row>
            <index>2</index>
            <shape>triangle</shape>
            <degrees>180</degrees>
            <sides>3.0</sides>
          </row>
        </data>

        >>> df.to_xml(attr_cols=[
        ...           \'index\', \'shape\', \'degrees\', \'sides\'
        ...           ])  # doctest: +SKIP
        <?xml version=\'1.0\' encoding=\'utf-8\'?>
        <data>
          <row index="0" shape="square" degrees="360" sides="4.0"/>
          <row index="1" shape="circle" degrees="360"/>
          <row index="2" shape="triangle" degrees="180" sides="3.0"/>
        </data>

        >>> df.to_xml(namespaces={"doc": "https://example.com"},
        ...           prefix="doc")  # doctest: +SKIP
        <?xml version=\'1.0\' encoding=\'utf-8\'?>
        <doc:data xmlns:doc="https://example.com">
          <doc:row>
            <doc:index>0</doc:index>
            <doc:shape>square</doc:shape>
            <doc:degrees>360</doc:degrees>
            <doc:sides>4.0</doc:sides>
          </doc:row>
          <doc:row>
            <doc:index>1</doc:index>
            <doc:shape>circle</doc:shape>
            <doc:degrees>360</doc:degrees>
            <doc:sides/>
          </doc:row>
          <doc:row>
            <doc:index>2</doc:index>
            <doc:shape>triangle</doc:shape>
            <doc:degrees>180</doc:degrees>
            <doc:sides>3.0</doc:sides>
          </doc:row>
        </doc:data>
        '''
    def info(self, verbose: bool | None, buf: WriteBuffer[str] | None, max_cols: int | None, memory_usage: bool | str | None, show_counts: bool | None) -> None:
        '''
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and columns, non-null values and memory usage.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary. By default, the setting in
            ``pandas.options.display.max_info_columns`` is followed.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        max_cols : int, optional
            When to switch from the verbose to the truncated output. If the
            DataFrame has more than `max_cols` columns, the truncated output
            is used. By default, the setting in
            ``pandas.options.display.max_info_columns`` is used.
        memory_usage : bool, str, optional
            Specifies whether total memory usage of the DataFrame
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
            This method prints a summary of a DataFrame and returns None.

        See Also
        --------
        DataFrame.describe: Generate descriptive statistics of DataFrame
            columns.
        DataFrame.memory_usage: Memory usage of DataFrame columns.

        Examples
        --------
        >>> int_values = [1, 2, 3, 4, 5]
        >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']
        >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
        ...                   "float_col": float_values})
        >>> df
            int_col text_col  float_col
        0        1    alpha       0.00
        1        2     beta       0.25
        2        3    gamma       0.50
        3        4    delta       0.75
        4        5  epsilon       1.00

        Prints information of all columns:

        >>> df.info(verbose=True)
        <class \'pandas.core.frame.DataFrame\'>
        RangeIndex: 5 entries, 0 to 4
        Data columns (total 3 columns):
         #   Column     Non-Null Count  Dtype
        ---  ------     --------------  -----
         0   int_col    5 non-null      int64
         1   text_col   5 non-null      object
         2   float_col  5 non-null      float64
        dtypes: float64(1), int64(1), object(1)
        memory usage: 248.0+ bytes

        Prints a summary of columns count and its dtypes but not per column
        information:

        >>> df.info(verbose=False)
        <class \'pandas.core.frame.DataFrame\'>
        RangeIndex: 5 entries, 0 to 4
        Columns: 3 entries, int_col to float_col
        dtypes: float64(1), int64(1), object(1)
        memory usage: 248.0+ bytes

        Pipe output of DataFrame.info to buffer instead of sys.stdout, get
        buffer content and writes to a text file:

        >>> import io
        >>> buffer = io.StringIO()
        >>> df.info(buf=buffer)
        >>> s = buffer.getvalue()
        >>> with open("df_info.txt", "w",
        ...           encoding="utf-8") as f:  # doctest: +SKIP
        ...     f.write(s)
        260

        The `memory_usage` parameter allows deep introspection mode, specially
        useful for big DataFrames and fine-tune memory optimization:

        >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)
        >>> df = pd.DataFrame({
        ...     \'column_1\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),
        ...     \'column_2\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),
        ...     \'column_3\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)
        ... })
        >>> df.info()
        <class \'pandas.core.frame.DataFrame\'>
        RangeIndex: 1000000 entries, 0 to 999999
        Data columns (total 3 columns):
         #   Column    Non-Null Count    Dtype
        ---  ------    --------------    -----
         0   column_1  1000000 non-null  object
         1   column_2  1000000 non-null  object
         2   column_3  1000000 non-null  object
        dtypes: object(3)
        memory usage: 22.9+ MB

        >>> df.info(memory_usage=\'deep\')
        <class \'pandas.core.frame.DataFrame\'>
        RangeIndex: 1000000 entries, 0 to 999999
        Data columns (total 3 columns):
         #   Column    Non-Null Count    Dtype
        ---  ------    --------------    -----
         0   column_1  1000000 non-null  object
         1   column_2  1000000 non-null  object
         2   column_3  1000000 non-null  object
        dtypes: object(3)
        memory usage: 165.9 MB
        '''
    def memory_usage(self, index: bool = ..., deep: bool = ...) -> Series:
        """
        Return the memory usage of each column in bytes.

        The memory usage can optionally include the contribution of
        the index and elements of `object` dtype.

        This value is displayed in `DataFrame.info` by default. This can be
        suppressed by setting ``pandas.options.display.memory_usage`` to False.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the DataFrame's
            index in returned Series. If ``index=True``, the memory usage of
            the index is the first item in the output.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned values.

        Returns
        -------
        Series
            A Series whose index is the original column names and whose values
            is the memory usage of each column in bytes.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of an
            ndarray.
        Series.memory_usage : Bytes consumed by a Series.
        Categorical : Memory-efficient array for string values with
            many repeated values.
        DataFrame.info : Concise summary of a DataFrame.

        Notes
        -----
        See the :ref:`Frequently Asked Questions <df-memory-usage>` for more
        details.

        Examples
        --------
        >>> dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']
        >>> data = dict([(t, np.ones(shape=5000, dtype=int).astype(t))
        ...              for t in dtypes])
        >>> df = pd.DataFrame(data)
        >>> df.head()
           int64  float64            complex128  object  bool
        0      1      1.0              1.0+0.0j       1  True
        1      1      1.0              1.0+0.0j       1  True
        2      1      1.0              1.0+0.0j       1  True
        3      1      1.0              1.0+0.0j       1  True
        4      1      1.0              1.0+0.0j       1  True

        >>> df.memory_usage()
        Index           128
        int64         40000
        float64       40000
        complex128    80000
        object        40000
        bool           5000
        dtype: int64

        >>> df.memory_usage(index=False)
        int64         40000
        float64       40000
        complex128    80000
        object        40000
        bool           5000
        dtype: int64

        The memory footprint of `object` dtype columns is ignored by default:

        >>> df.memory_usage(deep=True)
        Index            128
        int64          40000
        float64        40000
        complex128     80000
        object        180000
        bool            5000
        dtype: int64

        Use a Categorical for efficient storage of an object-dtype column with
        many repeated values.

        >>> df['object'].astype('category').memory_usage(deep=True)
        5244
        """
    def transpose(self, *args, copy: bool = ...) -> DataFrame:
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows as columns
        and vice-versa. The property :attr:`.T` is an accessor to the method
        :meth:`transpose`.

        Parameters
        ----------
        *args : tuple, optional
            Accepted for compatibility with NumPy.
        copy : bool, default False
            Whether to copy the data after transposing, even for DataFrames
            with a single dtype.

            Note that a copy is always required for mixed dtype DataFrames,
            or for DataFrames with any extension types.

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
        DataFrame
            The transposed DataFrame.

        See Also
        --------
        numpy.transpose : Permute the dimensions of a given array.

        Notes
        -----
        Transposing a DataFrame with mixed dtypes will result in a homogeneous
        DataFrame with the `object` dtype. In such a case, a copy of the data
        is always made.

        Examples
        --------
        **Square DataFrame with homogeneous dtype**

        >>> d1 = {'col1': [1, 2], 'col2': [3, 4]}
        >>> df1 = pd.DataFrame(data=d1)
        >>> df1
           col1  col2
        0     1     3
        1     2     4

        >>> df1_transposed = df1.T  # or df1.transpose()
        >>> df1_transposed
              0  1
        col1  1  2
        col2  3  4

        When the dtype is homogeneous in the original DataFrame, we get a
        transposed DataFrame with the same dtype:

        >>> df1.dtypes
        col1    int64
        col2    int64
        dtype: object
        >>> df1_transposed.dtypes
        0    int64
        1    int64
        dtype: object

        **Non-square DataFrame with mixed dtypes**

        >>> d2 = {'name': ['Alice', 'Bob'],
        ...       'score': [9.5, 8],
        ...       'employed': [False, True],
        ...       'kids': [0, 0]}
        >>> df2 = pd.DataFrame(data=d2)
        >>> df2
            name  score  employed  kids
        0  Alice    9.5     False     0
        1    Bob    8.0      True     0

        >>> df2_transposed = df2.T  # or df2.transpose()
        >>> df2_transposed
                      0     1
        name      Alice   Bob
        score       9.5   8.0
        employed  False  True
        kids          0     0

        When the DataFrame has mixed dtypes, we get a transposed DataFrame with
        the `object` dtype:

        >>> df2.dtypes
        name         object
        score       float64
        employed       bool
        kids          int64
        dtype: object
        >>> df2_transposed.dtypes
        0    object
        1    object
        dtype: object
        """
    def _ixs(self, i: int, axis: AxisInt = ...) -> Series:
        """
        Parameters
        ----------
        i : int
        axis : int

        Returns
        -------
        Series
        """
    def _get_column_array(self, i: int) -> ArrayLike:
        """
        Get the values of the i'th column (ndarray or ExtensionArray, as stored
        in the Block)

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution (for read-only purposes).
        """
    def _iter_column_arrays(self) -> Iterator[ArrayLike]:
        """
        Iterate over the arrays of all columns in order.
        This returns the values as stored in the Block (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution (for read-only purposes).
        """
    def _getitem_nocopy(self, key: list):
        """
        Behaves like __getitem__, but returns a view in cases where __getitem__
        would make a copy.
        """
    def __getitem__(self, key): ...
    def _getitem_bool_array(self, key): ...
    def _getitem_multilevel(self, key): ...
    def _get_value(self, index, col, takeable: bool = ...) -> Scalar:
        """
        Quickly retrieve single value at passed column and index.

        Parameters
        ----------
        index : row label
        col : column label
        takeable : interpret the index/col as indexers, default False

        Returns
        -------
        scalar

        Notes
        -----
        Assumes that both `self.index._index_as_unique` and
        `self.columns._index_as_unique`; Caller is responsible for checking.
        """
    def isetitem(self, loc, value) -> None:
        """
        Set the given value in the column with position `loc`.

        This is a positional analogue to ``__setitem__``.

        Parameters
        ----------
        loc : int or sequence of ints
            Index position for the column.
        value : scalar or arraylike
            Value(s) for the column.

        Notes
        -----
        ``frame.isetitem(loc, value)`` is an in-place method as it will
        modify the DataFrame in place (not returning a new object). In contrast to
        ``frame.iloc[:, i] = value`` which will try to update the existing values in
        place, ``frame.isetitem(loc, value)`` will not update the values of the column
        itself in place, it will instead insert a new array.

        In cases where ``frame.columns`` is unique, this is equivalent to
        ``frame[frame.columns[i]] = value``.
        """
    def __setitem__(self, key, value) -> None: ...
    def _setitem_slice(self, key: slice, value) -> None: ...
    def _setitem_array(self, key, value): ...
    def _iset_not_inplace(self, key, value): ...
    def _setitem_frame(self, key, value): ...
    def _set_item_frame_value(self, key, value: DataFrame) -> None: ...
    def _iset_item_mgr(self, loc: int | slice | np.ndarray, value, inplace: bool = ..., refs: BlockValuesRefs | None) -> None: ...
    def _set_item_mgr(self, key, value: ArrayLike, refs: BlockValuesRefs | None) -> None: ...
    def _iset_item(self, loc: int, value: Series, inplace: bool = ...) -> None: ...
    def _set_item(self, key, value) -> None:
        """
        Add series to DataFrame in specified column.

        If series is a numpy-array (not a Series/TimeSeries), it must be the
        same length as the DataFrames index or an error will be thrown.

        Series/TimeSeries will be conformed to the DataFrames index to
        ensure homogeneity.
        """
    def _set_value(self, index: IndexLabel, col, value: Scalar, takeable: bool = ...) -> None:
        """
        Put single value at passed column and index.

        Parameters
        ----------
        index : Label
            row label
        col : Label
            column label
        value : scalar
        takeable : bool, default False
            Sets whether or not index/col interpreted as indexers
        """
    def _ensure_valid_index(self, value) -> None:
        """
        Ensure that if we don't have an index, that we can create one from the
        passed value.
        """
    def _box_col_values(self, values: SingleDataManager, loc: int) -> Series:
        """
        Provide boxed values for a column.
        """
    def _clear_item_cache(self) -> None: ...
    def _get_item_cache(self, item: Hashable) -> Series:
        """Return the cached item, item represents a label indexer."""
    def _reset_cacher(self) -> None: ...
    def _maybe_cache_changed(self, item, value: Series, inplace: bool) -> None:
        """
        The object has called back to us saying maybe it has changed.
        """
    def query(self, expr: str, *, inplace: bool = ..., **kwargs) -> DataFrame | None:
        '''
        Query the columns of a DataFrame with a boolean expression.

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            You can refer to variables
            in the environment by prefixing them with an \'@\' character like
            ``@a + b``.

            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks. Thus, column names containing spaces
            or punctuations (besides underscores) or starting with digits must be
            surrounded by backticks. (For example, a column named "Area (cm^2)" would
            be referenced as ```Area (cm^2)```). Column names which are Python keywords
            (like "list", "for", "import", etc) cannot be used.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

        inplace : bool
            Whether to modify the DataFrame rather than creating a new one.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by :meth:`DataFrame.query`.

        Returns
        -------
        DataFrame or None
            DataFrame resulting from the provided query expression or
            None if ``inplace=True``.

        See Also
        --------
        eval : Evaluate a string describing operations on
            DataFrame columns.
        DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

        Notes
        -----
        The result of the evaluation of this expression is first passed to
        :attr:`DataFrame.loc` and if that fails because of a
        multidimensional key (e.g., a DataFrame) then the result will be passed
        to :meth:`DataFrame.__getitem__`.

        This method uses the top-level :func:`eval` function to
        evaluate the passed query.

        The :meth:`~pandas.DataFrame.query` method uses a slightly
        modified Python syntax by default. For example, the ``&`` and ``|``
        (bitwise) operators have the precedence of their boolean cousins,
        :keyword:`and` and :keyword:`or`. This *is* syntactically valid Python,
        however the semantics are different.

        You can change the semantics of the expression by passing the keyword
        argument ``parser=\'python\'``. This enforces the same semantics as
        evaluation in Python space. Likewise, you can pass ``engine=\'python\'``
        to evaluate an expression using Python itself as a backend. This is not
        recommended as it is inefficient compared to using ``numexpr`` as the
        engine.

        The :attr:`DataFrame.index` and
        :attr:`DataFrame.columns` attributes of the
        :class:`~pandas.DataFrame` instance are placed in the query namespace
        by default, which allows you to treat both the index and columns of the
        frame as a column in the frame.
        The identifier ``index`` is used for the frame index; you can also
        use the name of the index to identify it in a query. Please note that
        Python keywords may not be used as identifiers.

        For further details and examples see the ``query`` documentation in
        :ref:`indexing <indexing.query>`.

        *Backtick quoted variables*

        Backtick quoted variables are parsed as literal Python code and
        are converted internally to a Python valid identifier.
        This can lead to the following problems.

        During parsing a number of disallowed characters inside the backtick
        quoted string are replaced by strings that are allowed as a Python identifier.
        These characters include all operators in Python, the space character, the
        question mark, the exclamation mark, the dollar sign, and the euro sign.
        For other characters that fall outside the ASCII range (U+0001..U+007F)
        and those that are not further specified in PEP 3131,
        the query parser will raise an error.
        This excludes whitespace different than the space character,
        but also the hashtag (as it is used for comments) and the backtick
        itself (backtick can also not be escaped).

        In a special case, quotes that make a pair around a backtick can
        confuse the parser.
        For example, ```it\'s` > `that\'s``` will raise an error,
        as it forms a quoted string (``\'s > `that\'``) with a backtick inside.

        See also the Python documentation about lexical analysis
        (https://docs.python.org/3/reference/lexical_analysis.html)
        in combination with the source code in :mod:`pandas.core.computation.parsing`.

        Examples
        --------
        >>> df = pd.DataFrame({\'A\': range(1, 6),
        ...                    \'B\': range(10, 0, -2),
        ...                    \'C C\': range(10, 5, -1)})
        >>> df
           A   B  C C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6
        >>> df.query(\'A > B\')
           A  B  C C
        4  5  2    6

        The previous expression is equivalent to

        >>> df[df.A > df.B]
           A  B  C C
        4  5  2    6

        For columns with spaces in their name, you can use backtick quoting.

        >>> df.query(\'B == `C C`\')
           A   B  C C
        0  1  10   10

        The previous expression is equivalent to

        >>> df[df.B == df[\'C C\']]
           A   B  C C
        0  1  10   10
        '''
    def eval(self, expr: str, *, inplace: bool = ..., **kwargs) -> Any | None:
        """
        Evaluate a string describing operations on DataFrame columns.

        Operates on columns only, not specific rows or elements.  This allows
        `eval` to run arbitrary code, which can make you vulnerable to code
        injection if you pass user input to this function.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing DataFrame. Otherwise,
            a new DataFrame is returned.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by
            :meth:`~pandas.DataFrame.query`.

        Returns
        -------
        ndarray, scalar, pandas object, or None
            The result of the evaluation or None if ``inplace=True``.

        See Also
        --------
        DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
        DataFrame.assign : Can evaluate an expression or function to create new
            values for a column.
        eval : Evaluate a Python expression as a string using various
            backends.

        Notes
        -----
        For more details see the API documentation for :func:`~eval`.
        For detailed examples see :ref:`enhancing performance with eval
        <enhancingperf.eval>`.

        Examples
        --------
        >>> df = pd.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2
        >>> df.eval('A + B')
        0    11
        1    10
        2     9
        3     8
        4     7
        dtype: int64

        Assignment is allowed though by default the original DataFrame is not
        modified.

        >>> df.eval('C = A + B')
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2

        Multiple columns can be assigned to using multi-line expressions:

        >>> df.eval(
        ...     '''
        ... C = A + B
        ... D = A - B
        ... '''
        ... )
           A   B   C  D
        0  1  10  11 -9
        1  2   8  10 -6
        2  3   6   9 -3
        3  4   4   8  0
        4  5   2   7  3
        """
    def select_dtypes(self, include, exclude) -> Self:
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Parameters
        ----------
        include, exclude : scalar or list-like
            A selection of dtypes or strings to be included/excluded. At least
            one of these parameters must be supplied.

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes in ``include`` and
            excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            * If both of ``include`` and ``exclude`` are empty
            * If ``include`` and ``exclude`` have overlapping elements
            * If any kind of string dtype is passed in.

        See Also
        --------
        DataFrame.dtypes: Return Series with the data type of each column.

        Notes
        -----
        * To select all *numeric* types, use ``np.number`` or ``'number'``
        * To select strings you must use the ``object`` dtype, but note that
          this will return *all* object dtype columns
        * See the `numpy dtype hierarchy
          <https://numpy.org/doc/stable/reference/arrays.scalars.html>`__
        * To select datetimes, use ``np.datetime64``, ``'datetime'`` or
          ``'datetime64'``
        * To select timedeltas, use ``np.timedelta64``, ``'timedelta'`` or
          ``'timedelta64'``
        * To select Pandas categorical dtypes, use ``'category'``
        * To select Pandas datetimetz dtypes, use ``'datetimetz'``
          or ``'datetime64[ns, tz]'``

        Examples
        --------
        >>> df = pd.DataFrame({'a': [1, 2] * 3,
        ...                    'b': [True, False] * 3,
        ...                    'c': [1.0, 2.0] * 3})
        >>> df
                a      b  c
        0       1   True  1.0
        1       2  False  2.0
        2       1   True  1.0
        3       2  False  2.0
        4       1   True  1.0
        5       2  False  2.0

        >>> df.select_dtypes(include='bool')
           b
        0  True
        1  False
        2  True
        3  False
        4  True
        5  False

        >>> df.select_dtypes(include=['float64'])
           c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0

        >>> df.select_dtypes(exclude=['int64'])
               b    c
        0   True  1.0
        1  False  2.0
        2   True  1.0
        3  False  2.0
        4   True  1.0
        5  False  2.0
        """
    def insert(self, loc: int, column: Hashable, value: Scalar | AnyArrayLike, allow_duplicates: bool | lib.NoDefault = ...) -> None:
        '''
        Insert column into DataFrame at specified location.

        Raises a ValueError if `column` is already contained in the DataFrame,
        unless `allow_duplicates` is set to True.

        Parameters
        ----------
        loc : int
            Insertion index. Must verify 0 <= loc <= len(columns).
        column : str, number, or hashable object
            Label of the inserted column.
        value : Scalar, Series, or array-like
            Content of the inserted column.
        allow_duplicates : bool, optional, default lib.no_default
            Allow duplicate column labels to be created.

        See Also
        --------
        Index.insert : Insert new item by index.

        Examples
        --------
        >>> df = pd.DataFrame({\'col1\': [1, 2], \'col2\': [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4
        >>> df.insert(1, "newcol", [99, 99])
        >>> df
           col1  newcol  col2
        0     1      99     3
        1     2      99     4
        >>> df.insert(0, "col1", [100, 100], allow_duplicates=True)
        >>> df
           col1  col1  newcol  col2
        0   100     1      99     3
        1   100     2      99     4

        Notice that pandas uses index alignment in case of `value` from type `Series`:

        >>> df.insert(0, "col0", pd.Series([5, 6], index=[1, 2]))
        >>> df
           col0  col1  col1  newcol  col2
        0   NaN   100     1      99     3
        1   5.0   100     2      99     4
        '''
    def assign(self, **kwargs) -> DataFrame:
        """
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable or Series}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though pandas doesn't check it).
            If the values are not callable, (e.g. a Series, scalar, or array),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible.
        Later items in '\\*\\*kwargs' may refer to newly created or modified
        columns in 'df'; items are computed and assigned into 'df' in order.

        Examples
        --------
        >>> df = pd.DataFrame({'temp_c': [17.0, 25.0]},
        ...                   index=['Portland', 'Berkeley'])
        >>> df
                  temp_c
        Portland    17.0
        Berkeley    25.0

        Where the value is a callable, evaluated on `df`:

        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence:

        >>> df.assign(temp_f=df['temp_c'] * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        You can create multiple columns within the same assign where one
        of the columns depends on another one defined within the same assign:

        >>> df.assign(temp_f=lambda x: x['temp_c'] * 9 / 5 + 32,
        ...           temp_k=lambda x: (x['temp_f'] + 459.67) * 5 / 9)
                  temp_c  temp_f  temp_k
        Portland    17.0    62.6  290.15
        Berkeley    25.0    77.0  298.15
        """
    def _sanitize_column(self, value) -> tuple[ArrayLike, BlockValuesRefs | None]:
        """
        Ensures new columns (which go into the BlockManager as new blocks) are
        always copied (or a reference is being tracked to them under CoW)
        and converted into an array.

        Parameters
        ----------
        value : scalar, Series, or array-like

        Returns
        -------
        tuple of numpy.ndarray or ExtensionArray and optional BlockValuesRefs
        """
    def _reindex_multi(self, axes: dict[str, Index], copy: bool, fill_value) -> DataFrame:
        """
        We are guaranteed non-Nones in the axes.
        """
    def set_axis(self, labels, *, axis: Axis = ..., copy: bool | None) -> DataFrame:
        '''
        Assign desired index to given axis.

        Indexes for column or row labels can be changed by assigning
        a list-like or Index.

        Parameters
        ----------
        labels : list-like, Index
            The values for the new index.

        axis : {0 or \'index\', 1 or \'columns\'}, default 0
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
        DataFrame
            An object of type DataFrame.

        See Also
        --------
        DataFrame.rename_axis : Alter the name of the index or columns.

                Examples
                --------
                >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

                Change the row labels.

                >>> df.set_axis([\'a\', \'b\', \'c\'], axis=\'index\')
                   A  B
                a  1  4
                b  2  5
                c  3  6

                Change the column labels.

                >>> df.set_axis([\'I\', \'II\'], axis=\'columns\')
                   I  II
                0  1   4
                1  2   5
                2  3   6
        '''
    def reindex(self, labels, *, index, columns, axis: Axis | None, method: ReindexMethod | None, copy: bool | None, level: Level | None, fill_value: Scalar | None = ..., limit: int | None, tolerance) -> DataFrame:
        '''
        Conform DataFrame to new index with optional filling logic.

        Places NA/NaN in locations having no value in the previous index. A new object
        is produced unless the new index is equivalent to the current one and
        ``copy=False``.

        Parameters
        ----------

        labels : array-like, optional
            New labels / index to conform the axis specified by \'axis\' to.
        index : array-like, optional
            New labels for the index. Preferably an Index object to avoid
            duplicating data.
        columns : array-like, optional
            New labels for the columns. Preferably an Index object to avoid
            duplicating data.
        axis : int or str, optional
            Axis to target. Can be either the axis name (\'index\', \'columns\')
            or number (0, 1).
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
        DataFrame with changed index.

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
    def drop(self, labels: IndexLabel | None, *, axis: Axis = ..., index: IndexLabel | None, columns: IndexLabel | None, level: Level | None, inplace: bool = ..., errors: IgnoreRaise = ...) -> DataFrame | None:
        """
        Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding
        axis, or by directly specifying index or column names. When using a
        multi-index, labels on different levels can be removed by specifying
        the level. See the :ref:`user guide <advanced.shown_levels>`
        for more information about the now unused levels.

        Parameters
        ----------
        labels : single label or list-like
            Index or column labels to drop. A tuple will be used as a single
            label and not treated as a list-like.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
        index : single label or list-like
            Alternative to specifying axis (``labels, axis=0``
            is equivalent to ``index=labels``).
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If False, return a copy. Otherwise, do operation
            in place and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        DataFrame or None
            Returns DataFrame or None DataFrame with the specified
            index or column labels removed or None if inplace=True.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis.

        See Also
        --------
        DataFrame.loc : Label-location based indexer for selection by label.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.
        DataFrame.drop_duplicates : Return DataFrame with duplicate rows
            removed, optionally only considering certain columns.
        Series.drop : Return Series with specified index labels removed.

        Examples
        --------
        >>> df = pd.DataFrame(np.arange(12).reshape(3, 4),
        ...                   columns=['A', 'B', 'C', 'D'])
        >>> df
           A  B   C   D
        0  0  1   2   3
        1  4  5   6   7
        2  8  9  10  11

        Drop columns

        >>> df.drop(['B', 'C'], axis=1)
           A   D
        0  0   3
        1  4   7
        2  8  11

        >>> df.drop(columns=['B', 'C'])
           A   D
        0  0   3
        1  4   7
        2  8  11

        Drop a row by index

        >>> df.drop([0, 1])
           A  B   C   D
        2  8  9  10  11

        Drop columns and/or rows of MultiIndex DataFrame

        >>> midx = pd.MultiIndex(levels=[['llama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> df = pd.DataFrame(index=midx, columns=['big', 'small'],
        ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
        ...                         [250, 150], [1.5, 0.8], [320, 250],
        ...                         [1, 0.8], [0.3, 0.2]])
        >>> df
                        big     small
        llama   speed   45.0    30.0
                weight  200.0   100.0
                length  1.5     1.0
        cow     speed   30.0    20.0
                weight  250.0   150.0
                length  1.5     0.8
        falcon  speed   320.0   250.0
                weight  1.0     0.8
                length  0.3     0.2

        Drop a specific index combination from the MultiIndex
        DataFrame, i.e., drop the combination ``'falcon'`` and
        ``'weight'``, which deletes only the corresponding row

        >>> df.drop(index=('falcon', 'weight'))
                        big     small
        llama   speed   45.0    30.0
                weight  200.0   100.0
                length  1.5     1.0
        cow     speed   30.0    20.0
                weight  250.0   150.0
                length  1.5     0.8
        falcon  speed   320.0   250.0
                length  0.3     0.2

        >>> df.drop(index='cow', columns='small')
                        big
        llama   speed   45.0
                weight  200.0
                length  1.5
        falcon  speed   320.0
                weight  1.0
                length  0.3

        >>> df.drop(index='length', level=1)
                        big     small
        llama   speed   45.0    30.0
                weight  200.0   100.0
        cow     speed   30.0    20.0
                weight  250.0   150.0
        falcon  speed   320.0   250.0
                weight  1.0     0.8
        """
    def rename(self, mapper: Renamer | None, *, index: Renamer | None, columns: Renamer | None, axis: Axis | None, copy: bool | None, inplace: bool = ..., level: Level | None, errors: IgnoreRaise = ...) -> DataFrame | None:
        '''
        Rename columns or index labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don\'t throw an
        error.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        mapper : dict-like or function
            Dict-like or function transformations to apply to
            that axis\' values. Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index`` and
            ``columns``.
        index : dict-like or function
            Alternative to specifying axis (``mapper, axis=0``
            is equivalent to ``index=mapper``).
        columns : dict-like or function
            Alternative to specifying axis (``mapper, axis=1``
            is equivalent to ``columns=mapper``).
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            Axis to target with ``mapper``. Can be either the axis name
            (\'index\', \'columns\') or number (0, 1). The default is \'index\'.
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
            Whether to modify the DataFrame rather than creating a new one.
            If True then value of copy is ignored.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified
            level.
        errors : {\'ignore\', \'raise\'}, default \'ignore\'
            If \'raise\', raise a `KeyError` when a dict-like `mapper`, `index`,
            or `columns` contains labels that are not present in the Index
            being transformed.
            If \'ignore\', existing keys will be renamed and extra keys will be
            ignored.

        Returns
        -------
        DataFrame or None
            DataFrame with the renamed axis labels or None if ``inplace=True``.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis and
            "errors=\'raise\'".

        See Also
        --------
        DataFrame.rename_axis : Set the name of the axis.

        Examples
        --------
        ``DataFrame.rename`` supports two calling conventions

        * ``(index=index_mapper, columns=columns_mapper, ...)``
        * ``(mapper, axis={\'index\', \'columns\'}, ...)``

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Rename columns using a mapping:

        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> df.rename(columns={"A": "a", "B": "c"})
           a  c
        0  1  4
        1  2  5
        2  3  6

        Rename index using a mapping:

        >>> df.rename(index={0: "x", 1: "y", 2: "z"})
           A  B
        x  1  4
        y  2  5
        z  3  6

        Cast index labels to a different type:

        >>> df.index
        RangeIndex(start=0, stop=3, step=1)
        >>> df.rename(index=str).index
        Index([\'0\', \'1\', \'2\'], dtype=\'object\')

        >>> df.rename(columns={"A": "a", "B": "b", "C": "c"}, errors="raise")
        Traceback (most recent call last):
        KeyError: [\'C\'] not found in axis

        Using axis-style parameters:

        >>> df.rename(str.lower, axis=\'columns\')
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> df.rename({1: 2, 2: 4}, axis=\'index\')
           A  B
        0  1  4
        2  2  5
        4  3  6
        '''
    def pop(self, item: Hashable) -> Series:
        """
        Return item and drop from frame. Raise KeyError if not found.

        Parameters
        ----------
        item : label
            Label of column to be popped.

        Returns
        -------
        Series

        Examples
        --------
        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan)],
        ...                   columns=('name', 'class', 'max_speed'))
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        >>> df.pop('class')
        0      bird
        1      bird
        2    mammal
        3    mammal
        Name: class, dtype: object

        >>> df
             name  max_speed
        0  falcon      389.0
        1  parrot       24.0
        2    lion       80.5
        3  monkey        NaN
        """
    def _replace_columnwise(self, mapping: dict[Hashable, tuple[Any, Any]], inplace: bool, regex):
        """
        Dispatch to Series.replace column-wise.

        Parameters
        ----------
        mapping : dict
            of the form {col: (target, value)}
        inplace : bool
        regex : bool or same types as `to_replace` in DataFrame.replace

        Returns
        -------
        DataFrame or None
        """
    def shift(self, periods: int | Sequence[int] = ..., freq: Frequency | None, axis: Axis = ..., fill_value: Hashable = ..., suffix: str | None) -> DataFrame:
        '''
        Shift index by desired number of periods with an optional time `freq`.

        When `freq` is not passed, shift the index without realigning the data.
        If `freq` is passed (in this case, the index must be date or datetime,
        or it will raise a `NotImplementedError`), the index will be
        increased using the periods and the `freq`. `freq` can be inferred
        when specified as "infer" as long as either freq or inferred_freq
        attribute is set in the index.

        Parameters
        ----------
        periods : int or Sequence
            Number of periods to shift. Can be positive or negative.
            If an iterable of ints, the data will be shifted once by each int.
            This is equivalent to shifting by one value at a time and
            concatenating all resulting frames. The resulting columns will have
            the shift suffixed to their column names. For multiple periods,
            axis must not be 1.
        freq : DateOffset, tseries.offsets, timedelta, or str, optional
            Offset to use from the tseries module or time rule (e.g. \'EOM\').
            If `freq` is specified then the index values are shifted but the
            data is not realigned. That is, use `freq` if you would like to
            extend the index when shifting and preserve the original data.
            If `freq` is specified as "infer" then it will be inferred from
            the freq or inferred_freq attributes of the index. If neither of
            those attributes exist, a ValueError is thrown.
        axis : {0 or \'index\', 1 or \'columns\', None}, default None
            Shift direction. For `Series` this parameter is unused and defaults to 0.
        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            the default depends on the dtype of `self`.
            For numeric data, ``np.nan`` is used.
            For datetime, timedelta, or period data, etc. :attr:`NaT` is used.
            For extension dtypes, ``self.dtype.na_value`` is used.
        suffix : str, optional
            If str and periods is an iterable, this is added after the column
            name and before the shift value for each shifted column name.

        Returns
        -------
        DataFrame
            Copy of input object, shifted.

        See Also
        --------
        Index.shift : Shift values of Index.
        DatetimeIndex.shift : Shift values of DatetimeIndex.
        PeriodIndex.shift : Shift values of PeriodIndex.

        Examples
        --------
        >>> df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],
        ...                    "Col2": [13, 23, 18, 33, 48],
        ...                    "Col3": [17, 27, 22, 37, 52]},
        ...                   index=pd.date_range("2020-01-01", "2020-01-05"))
        >>> df
                    Col1  Col2  Col3
        2020-01-01    10    13    17
        2020-01-02    20    23    27
        2020-01-03    15    18    22
        2020-01-04    30    33    37
        2020-01-05    45    48    52

        >>> df.shift(periods=3)
                    Col1  Col2  Col3
        2020-01-01   NaN   NaN   NaN
        2020-01-02   NaN   NaN   NaN
        2020-01-03   NaN   NaN   NaN
        2020-01-04  10.0  13.0  17.0
        2020-01-05  20.0  23.0  27.0

        >>> df.shift(periods=1, axis="columns")
                    Col1  Col2  Col3
        2020-01-01   NaN    10    13
        2020-01-02   NaN    20    23
        2020-01-03   NaN    15    18
        2020-01-04   NaN    30    33
        2020-01-05   NaN    45    48

        >>> df.shift(periods=3, fill_value=0)
                    Col1  Col2  Col3
        2020-01-01     0     0     0
        2020-01-02     0     0     0
        2020-01-03     0     0     0
        2020-01-04    10    13    17
        2020-01-05    20    23    27

        >>> df.shift(periods=3, freq="D")
                    Col1  Col2  Col3
        2020-01-04    10    13    17
        2020-01-05    20    23    27
        2020-01-06    15    18    22
        2020-01-07    30    33    37
        2020-01-08    45    48    52

        >>> df.shift(periods=3, freq="infer")
                    Col1  Col2  Col3
        2020-01-04    10    13    17
        2020-01-05    20    23    27
        2020-01-06    15    18    22
        2020-01-07    30    33    37
        2020-01-08    45    48    52

        >>> df[\'Col1\'].shift(periods=[0, 1, 2])
                    Col1_0  Col1_1  Col1_2
        2020-01-01      10     NaN     NaN
        2020-01-02      20    10.0     NaN
        2020-01-03      15    20.0    10.0
        2020-01-04      30    15.0    20.0
        2020-01-05      45    30.0    15.0
        '''
    def set_index(self, keys, *, drop: bool = ..., append: bool = ..., inplace: bool = ..., verify_integrity: bool = ...) -> DataFrame | None:
        '''
        Set the DataFrame index using existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index`, ``np.ndarray``, and
            instances of :class:`~collections.abc.Iterator`.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        verify_integrity : bool, default False
            Check the new index for duplicates. Otherwise defer the check until
            necessary. Setting to False will improve the performance of this
            method.

        Returns
        -------
        DataFrame or None
            Changed row labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.reset_index : Opposite of set_index.
        DataFrame.reindex : Change to new indices or expand indices.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame({\'month\': [1, 4, 7, 10],
        ...                    \'year\': [2012, 2014, 2013, 2014],
        ...                    \'sale\': [55, 40, 84, 31]})
        >>> df
           month  year  sale
        0      1  2012    55
        1      4  2014    40
        2      7  2013    84
        3     10  2014    31

        Set the index to become the \'month\' column:

        >>> df.set_index(\'month\')
               year  sale
        month
        1      2012    55
        4      2014    40
        7      2013    84
        10     2014    31

        Create a MultiIndex using columns \'year\' and \'month\':

        >>> df.set_index([\'year\', \'month\'])
                    sale
        year  month
        2012  1     55
        2014  4     40
        2013  7     84
        2014  10    31

        Create a MultiIndex using an Index and a column:

        >>> df.set_index([pd.Index([1, 2, 3, 4]), \'year\'])
                 month  sale
           year
        1  2012  1      55
        2  2014  4      40
        3  2013  7      84
        4  2014  10     31

        Create a MultiIndex using two Series:

        >>> s = pd.Series([1, 2, 3, 4])
        >>> df.set_index([s, s**2])
              month  year  sale
        1 1       1  2012    55
        2 4       4  2014    40
        3 9       7  2013    84
        4 16     10  2014    31
        '''
    def reset_index(self, level: IndexLabel | None, *, drop: bool = ..., inplace: bool = ..., col_level: Hashable = ..., col_fill: Hashable = ..., allow_duplicates: bool | lib.NoDefault = ..., names: Hashable | Sequence[Hashable] | None) -> DataFrame | None:
        """
        Reset the index, or a level of it.

        Reset the index of the DataFrame, and use the default one instead.
        If the DataFrame has a MultiIndex, this method can remove one or more
        levels.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by
            default.
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        col_level : int or str, default 0
            If the columns have multiple levels, determines which level the
            labels are inserted into. By default it is inserted into the first
            level.
        col_fill : object, default ''
            If the columns have multiple levels, determines how the other
            levels are named. If None then the index name is repeated.
        allow_duplicates : bool, optional, default lib.no_default
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        names : int, str or 1-dimensional list, default None
            Using the given string, rename the DataFrame column which contains the
            index data. If the DataFrame has a MultiIndex, this has to be a list or
            tuple with length equal to the number of levels.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame or None
            DataFrame with the new index or None if ``inplace=True``.

        See Also
        --------
        DataFrame.set_index : Opposite of reset_index.
        DataFrame.reindex : Change to new indices or expand indices.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class  max_speed
        falcon    bird      389.0
        parrot    bird       24.0
        lion    mammal       80.5
        monkey  mammal        NaN

        When we reset the index, the old index is added as a column, and a
        new sequential index is used:

        >>> df.reset_index()
            index   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        We can use the `drop` parameter to avoid the old index being added as
        a column:

        >>> df.reset_index(drop=True)
            class  max_speed
        0    bird      389.0
        1    bird       24.0
        2  mammal       80.5
        3  mammal        NaN

        You can also use `reset_index` with `MultiIndex`.

        >>> index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
        ...                                    ('bird', 'parrot'),
        ...                                    ('mammal', 'lion'),
        ...                                    ('mammal', 'monkey')],
        ...                                   names=['class', 'name'])
        >>> columns = pd.MultiIndex.from_tuples([('speed', 'max'),
        ...                                      ('species', 'type')])
        >>> df = pd.DataFrame([(389.0, 'fly'),
        ...                    (24.0, 'fly'),
        ...                    (80.5, 'run'),
        ...                    (np.nan, 'jump')],
        ...                   index=index,
        ...                   columns=columns)
        >>> df
                       speed species
                         max    type
        class  name
        bird   falcon  389.0     fly
               parrot   24.0     fly
        mammal lion     80.5     run
               monkey    NaN    jump

        Using the `names` parameter, choose a name for the index column:

        >>> df.reset_index(names=['classes', 'names'])
          classes   names  speed species
                             max    type
        0    bird  falcon  389.0     fly
        1    bird  parrot   24.0     fly
        2  mammal    lion   80.5     run
        3  mammal  monkey    NaN    jump

        If the index has multiple levels, we can reset a subset of them:

        >>> df.reset_index(level='class')
                 class  speed species
                          max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        If we are not dropping the index, by default, it is placed in the top
        level. We can place it in another level:

        >>> df.reset_index(level='class', col_level=1)
                        speed species
                 class    max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        When the index is inserted under another level, we can specify under
        which one with the parameter `col_fill`:

        >>> df.reset_index(level='class', col_level=1, col_fill='species')
                      species  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump

        If we specify a nonexistent level for `col_fill`, it is created:

        >>> df.reset_index(level='class', col_level=1, col_fill='genus')
                        genus  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump
        """
    def isna(self) -> DataFrame:
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
        DataFrame
            Mask of bool values for each element in DataFrame that
            indicates whether an element is an NA value.

        See Also
        --------
        DataFrame.isnull : Alias of isna.
        DataFrame.notna : Boolean inverse of isna.
        DataFrame.dropna : Omit axes labels with missing values.
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
    def isnull(self) -> DataFrame:
        """
        DataFrame.isnull is an alias for DataFrame.isna.

        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
        values.
        Everything else gets mapped to False values. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).

        Returns
        -------
        DataFrame
            Mask of bool values for each element in DataFrame that
            indicates whether an element is an NA value.

        See Also
        --------
        DataFrame.isnull : Alias of isna.
        DataFrame.notna : Boolean inverse of isna.
        DataFrame.dropna : Omit axes labels with missing values.
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
    def notna(self) -> DataFrame:
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
        DataFrame
            Mask of bool values for each element in DataFrame that
            indicates whether an element is not an NA value.

        See Also
        --------
        DataFrame.notnull : Alias of notna.
        DataFrame.isna : Boolean inverse of notna.
        DataFrame.dropna : Omit axes labels with missing values.
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
    def notnull(self) -> DataFrame:
        """
        DataFrame.notnull is an alias for DataFrame.notna.

        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        DataFrame
            Mask of bool values for each element in DataFrame that
            indicates whether an element is not an NA value.

        See Also
        --------
        DataFrame.notnull : Alias of notna.
        DataFrame.isna : Boolean inverse of notna.
        DataFrame.dropna : Omit axes labels with missing values.
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
    def dropna(self, *, axis: Axis = ..., how: AnyAll | lib.NoDefault = ..., thresh: int | lib.NoDefault = ..., subset: IndexLabel | None, inplace: bool = ..., ignore_index: bool = ...) -> DataFrame | None:
        '''
        Remove missing values.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or \'index\' : Drop rows which contain missing values.
            * 1, or \'columns\' : Drop columns which contain missing value.

            Only a single axis is allowed.

        how : {\'any\', \'all\'}, default \'any\'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * \'any\' : If any NA values are present, drop that row or column.
            * \'all\' : If all values are NA, drop that row or column.

        thresh : int, optional
            Require that many non-NA values. Cannot be combined with how.
        subset : column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        DataFrame or None
            DataFrame with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        DataFrame.isna: Indicate missing values.
        DataFrame.notna : Indicate existing (non-missing) values.
        DataFrame.fillna : Replace missing values.
        Series.dropna : Drop missing values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> df = pd.DataFrame({"name": [\'Alfred\', \'Batman\', \'Catwoman\'],
        ...                    "toy": [np.nan, \'Batmobile\', \'Bullwhip\'],
        ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),
        ...                             pd.NaT]})
        >>> df
               name        toy       born
        0    Alfred        NaN        NaT
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT

        Drop the rows where at least one element is missing.

        >>> df.dropna()
             name        toy       born
        1  Batman  Batmobile 1940-04-25

        Drop the columns where at least one element is missing.

        >>> df.dropna(axis=\'columns\')
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are missing.

        >>> df.dropna(how=\'all\')
               name        toy       born
        0    Alfred        NaN        NaT
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT

        Keep only the rows with at least 2 non-NA values.

        >>> df.dropna(thresh=2)
               name        toy       born
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT

        Define in which columns to look for missing values.

        >>> df.dropna(subset=[\'name\', \'toy\'])
               name        toy       born
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT
        '''
    def drop_duplicates(self, subset: Hashable | Sequence[Hashable] | None, *, keep: DropKeep = ..., inplace: bool = ..., ignore_index: bool = ...) -> DataFrame | None:
        """
        Return DataFrame with duplicate rows removed.

        Considering certain columns is optional. Indexes, including time indexes
        are ignored.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', ``False``}, default 'first'
            Determines which duplicates (if any) to keep.

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            Whether to modify the DataFrame rather than creating a new one.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or None if ``inplace=True``.

        See Also
        --------
        DataFrame.value_counts: Count unique combinations of columns.

        Examples
        --------
        Consider dataset containing ramen rating.

        >>> df = pd.DataFrame({
        ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        ...     'rating': [4, 4, 3.5, 15, 5]
        ... })
        >>> df
            brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        By default, it removes duplicate rows based on all columns.

        >>> df.drop_duplicates()
            brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        To remove duplicates on specific column(s), use ``subset``.

        >>> df.drop_duplicates(subset=['brand'])
            brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5

        To remove duplicates and keep last occurrences, use ``keep``.

        >>> df.drop_duplicates(subset=['brand', 'style'], keep='last')
            brand style  rating
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        4  Indomie  pack     5.0
        """
    def duplicated(self, subset: Hashable | Sequence[Hashable] | None, keep: DropKeep = ...) -> Series:
        """
        Return boolean Series denoting duplicate rows.

        Considering certain columns is optional.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to mark.

            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        Series
            Boolean series for each duplicated rows.

        See Also
        --------
        Index.duplicated : Equivalent method on index.
        Series.duplicated : Equivalent method on Series.
        Series.drop_duplicates : Remove duplicate values from Series.
        DataFrame.drop_duplicates : Remove duplicate values from DataFrame.

        Examples
        --------
        Consider dataset containing ramen rating.

        >>> df = pd.DataFrame({
        ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        ...     'rating': [4, 4, 3.5, 15, 5]
        ... })
        >>> df
            brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        By default, for each set of duplicated values, the first occurrence
        is set on False and all others on True.

        >>> df.duplicated()
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True.

        >>> df.duplicated(keep='last')
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        By setting ``keep`` on False, all duplicates are True.

        >>> df.duplicated(keep=False)
        0     True
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        To find duplicates on specific column(s), use ``subset``.

        >>> df.duplicated(subset=['brand'])
        0    False
        1     True
        2    False
        3     True
        4     True
        dtype: bool
        """
    def sort_values(self, by: IndexLabel, *, axis: Axis = ..., ascending: bool | list[bool] | tuple[bool, ...] = ..., inplace: bool = ..., kind: SortKind = ..., na_position: str = ..., ignore_index: bool = ..., key: ValueKeyFunc | None) -> DataFrame | None:
        '''
        Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.

            - if `axis` is 0 or `\'index\'` then `by` may contain index
              levels and/or column labels.
            - if `axis` is 1 or `\'columns\'` then `by` may contain column
              levels and/or index labels.
        axis : "{0 or \'index\', 1 or \'columns\'}", default 0
             Axis to be sorted.
        ascending : bool or list of bool, default True
             Sort ascending vs. descending. Specify list for multiple sort
             orders.  If this is a list of bools, must match the length of
             the by.
        inplace : bool, default False
             If True, perform operation in-place.
        kind : {\'quicksort\', \'mergesort\', \'heapsort\', \'stable\'}, default \'quicksort\'
             Choice of sorting algorithm. See also :func:`numpy.sort` for more
             information. `mergesort` and `stable` are the only stable algorithms. For
             DataFrames, this option is only applied when sorting on a single
             column or label.
        na_position : {\'first\', \'last\'}, default \'last\'
             Puts NaNs at the beginning if `first`; `last` puts NaNs at the
             end.
        ignore_index : bool, default False
             If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            Apply the key function to the values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return a Series with the same shape as the input.
            It will be applied to each column in `by` independently.

        Returns
        -------
        DataFrame or None
            DataFrame with sorted values or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index : Sort a DataFrame by the index.
        Series.sort_values : Similar method for a Series.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     \'col1\': [\'A\', \'A\', \'B\', np.nan, \'D\', \'C\'],
        ...     \'col2\': [2, 1, 9, 8, 7, 4],
        ...     \'col3\': [0, 1, 9, 4, 2, 3],
        ...     \'col4\': [\'a\', \'B\', \'c\', \'D\', \'e\', \'F\']
        ... })
        >>> df
          col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F

        Sort by col1

        >>> df.sort_values(by=[\'col1\'])
          col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        5    C     4     3    F
        4    D     7     2    e
        3  NaN     8     4    D

        Sort by multiple columns

        >>> df.sort_values(by=[\'col1\', \'col2\'])
          col1  col2  col3 col4
        1    A     1     1    B
        0    A     2     0    a
        2    B     9     9    c
        5    C     4     3    F
        4    D     7     2    e
        3  NaN     8     4    D

        Sort Descending

        >>> df.sort_values(by=\'col1\', ascending=False)
          col1  col2  col3 col4
        4    D     7     2    e
        5    C     4     3    F
        2    B     9     9    c
        0    A     2     0    a
        1    A     1     1    B
        3  NaN     8     4    D

        Putting NAs first

        >>> df.sort_values(by=\'col1\', ascending=False, na_position=\'first\')
          col1  col2  col3 col4
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F
        2    B     9     9    c
        0    A     2     0    a
        1    A     1     1    B

        Sorting with a key function

        >>> df.sort_values(by=\'col4\', key=lambda col: col.str.lower())
           col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F

        Natural sort with the key argument,
        using the `natsort <https://github.com/SethMMorton/natsort>` package.

        >>> df = pd.DataFrame({
        ...    "time": [\'0hr\', \'128hr\', \'72hr\', \'48hr\', \'96hr\'],
        ...    "value": [10, 20, 30, 40, 50]
        ... })
        >>> df
            time  value
        0    0hr     10
        1  128hr     20
        2   72hr     30
        3   48hr     40
        4   96hr     50
        >>> from natsort import index_natsorted
        >>> df.sort_values(
        ...     by="time",
        ...     key=lambda x: np.argsort(index_natsorted(df["time"]))
        ... )
            time  value
        0    0hr     10
        3   48hr     40
        2   72hr     30
        4   96hr     50
        1  128hr     20
        '''
    def sort_index(self, *, axis: Axis = ..., level: IndexLabel | None, ascending: bool | Sequence[bool] = ..., inplace: bool = ..., kind: SortKind = ..., na_position: NaPosition = ..., sort_remaining: bool = ..., ignore_index: bool = ..., key: IndexKeyFunc | None) -> DataFrame | None:
        '''
        Sort object by labels (along an axis).

        Returns a new DataFrame sorted by label if `inplace` argument is
        ``False``, otherwise updates the original DataFrame and returns None.

        Parameters
        ----------
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            The axis along which to sort.  The value 0 identifies the rows,
            and 1 identifies the columns.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
        ascending : bool or list-like of bools, default True
            Sort ascending vs. descending. When the index is a MultiIndex the
            sort direction can be controlled for each level individually.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        kind : {\'quicksort\', \'mergesort\', \'heapsort\', \'stable\'}, default \'quicksort\'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. `mergesort` and `stable` are the only stable algorithms. For
            DataFrames, this option is only applied when sorting on a single
            column or label.
        na_position : {\'first\', \'last\'}, default \'last\'
            Puts NaNs at the beginning if `first`; `last` puts NaNs at the end.
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
            ``Index`` and return an ``Index`` of the same shape. For MultiIndex
            inputs, the key is applied *per level*.

        Returns
        -------
        DataFrame or None
            The original DataFrame sorted by the labels or None if ``inplace=True``.

        See Also
        --------
        Series.sort_index : Sort Series by the index.
        DataFrame.sort_values : Sort DataFrame by the value.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> df = pd.DataFrame([1, 2, 3, 4, 5], index=[100, 29, 234, 1, 150],
        ...                   columns=[\'A\'])
        >>> df.sort_index()
             A
        1    4
        29   2
        100  1
        150  5
        234  3

        By default, it sorts in ascending order, to sort in descending order,
        use ``ascending=False``

        >>> df.sort_index(ascending=False)
             A
        234  3
        150  5
        100  1
        29   2
        1    4

        A key function can be specified which is applied to the index before
        sorting. For a ``MultiIndex`` this is applied to each level separately.

        >>> df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[\'A\', \'b\', \'C\', \'d\'])
        >>> df.sort_index(key=lambda x: x.str.lower())
           a
        A  1
        b  2
        C  3
        d  4
        '''
    def value_counts(self, subset: IndexLabel | None, normalize: bool = ..., sort: bool = ..., ascending: bool = ..., dropna: bool = ...) -> Series:
        '''
        Return a Series containing the frequency of each distinct row in the Dataframe.

        Parameters
        ----------
        subset : label or list of labels, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies when True. Sort by DataFrame column values when False.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don\'t include counts of rows that contain NA values.

            .. versionadded:: 1.3.0

        Returns
        -------
        Series

        See Also
        --------
        Series.value_counts: Equivalent method on Series.

        Notes
        -----
        The returned Series will have a MultiIndex with one level per input
        column but an Index (non-multi) for a single label. By default, rows
        that contain any NA values are omitted from the result. By default,
        the resulting Series will be in descending order so that the first
        element is the most frequently-occurring row.

        Examples
        --------
        >>> df = pd.DataFrame({\'num_legs\': [2, 4, 4, 6],
        ...                    \'num_wings\': [2, 0, 0, 0]},
        ...                   index=[\'falcon\', \'dog\', \'cat\', \'ant\'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0
        cat            4          0
        ant            6          0

        >>> df.value_counts()
        num_legs  num_wings
        4         0            2
        2         2            1
        6         0            1
        Name: count, dtype: int64

        >>> df.value_counts(sort=False)
        num_legs  num_wings
        2         2            1
        4         0            2
        6         0            1
        Name: count, dtype: int64

        >>> df.value_counts(ascending=True)
        num_legs  num_wings
        2         2            1
        6         0            1
        4         0            2
        Name: count, dtype: int64

        >>> df.value_counts(normalize=True)
        num_legs  num_wings
        4         0            0.50
        2         2            0.25
        6         0            0.25
        Name: proportion, dtype: float64

        With `dropna` set to `False` we can also count rows with NA values.

        >>> df = pd.DataFrame({\'first_name\': [\'John\', \'Anne\', \'John\', \'Beth\'],
        ...                    \'middle_name\': [\'Smith\', pd.NA, pd.NA, \'Louise\']})
        >>> df
          first_name middle_name
        0       John       Smith
        1       Anne        <NA>
        2       John        <NA>
        3       Beth      Louise

        >>> df.value_counts()
        first_name  middle_name
        Beth        Louise         1
        John        Smith          1
        Name: count, dtype: int64

        >>> df.value_counts(dropna=False)
        first_name  middle_name
        Anne        NaN            1
        Beth        Louise         1
        John        Smith          1
                    NaN            1
        Name: count, dtype: int64

        >>> df.value_counts("first_name")
        first_name
        John    2
        Anne    1
        Beth    1
        Name: count, dtype: int64
        '''
    def nlargest(self, n: int, columns: IndexLabel, keep: NsmallestNlargestKeep = ...) -> DataFrame:
        '''
        Return the first `n` rows ordered by `columns` in descending order.

        Return the first `n` rows with the largest values in `columns`, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to
        ``df.sort_values(columns, ascending=False).head(n)``, but more
        performant.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.
        keep : {\'first\', \'last\', \'all\'}, default \'first\'
            Where there are duplicate values:

            - ``first`` : prioritize the first occurrence(s)
            - ``last`` : prioritize the last occurrence(s)
            - ``all`` : keep all the ties of the smallest item even if it means
              selecting more than ``n`` items.

        Returns
        -------
        DataFrame
            The first `n` rows ordered by the given columns in descending
            order.

        See Also
        --------
        DataFrame.nsmallest : Return the first `n` rows ordered by `columns` in
            ascending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Notes
        -----
        This function cannot be used with all column types. For example, when
        specifying columns with `object` or `category` dtypes, ``TypeError`` is
        raised.

        Examples
        --------
        >>> df = pd.DataFrame({\'population\': [59000000, 65000000, 434000,
        ...                                   434000, 434000, 337000, 11300,
        ...                                   11300, 11300],
        ...                    \'GDP\': [1937894, 2583560 , 12011, 4520, 12128,
        ...                            17036, 182, 38, 311],
        ...                    \'alpha-2\': ["IT", "FR", "MT", "MV", "BN",
        ...                                "IS", "NR", "TV", "AI"]},
        ...                   index=["Italy", "France", "Malta",
        ...                          "Maldives", "Brunei", "Iceland",
        ...                          "Nauru", "Tuvalu", "Anguilla"])
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru          11300      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI

        In the following example, we will use ``nlargest`` to select the three
        rows having the largest values in column "population".

        >>> df.nlargest(3, \'population\')
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Malta       434000    12011      MT

        When using ``keep=\'last\'``, ties are resolved in reverse order:

        >>> df.nlargest(3, \'population\', keep=\'last\')
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Brunei      434000    12128      BN

        When using ``keep=\'all\'``, the number of element kept can go beyond ``n``
        if there are duplicate values for the smallest element, all the
        ties are kept:

        >>> df.nlargest(3, \'population\', keep=\'all\')
                  population      GDP alpha-2
        France      65000000  2583560      FR
        Italy       59000000  1937894      IT
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN

        However, ``nlargest`` does not keep ``n`` distinct largest elements:

        >>> df.nlargest(5, \'population\', keep=\'all\')
                  population      GDP alpha-2
        France      65000000  2583560      FR
        Italy       59000000  1937894      IT
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN

        To order by the largest values in column "population" and then "GDP",
        we can specify multiple columns like in the next example.

        >>> df.nlargest(3, [\'population\', \'GDP\'])
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Brunei      434000    12128      BN
        '''
    def nsmallest(self, n: int, columns: IndexLabel, keep: NsmallestNlargestKeep = ...) -> DataFrame:
        '''
        Return the first `n` rows ordered by `columns` in ascending order.

        Return the first `n` rows with the smallest values in `columns`, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to
        ``df.sort_values(columns, ascending=True).head(n)``, but more
        performant.

        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.
        keep : {\'first\', \'last\', \'all\'}, default \'first\'
            Where there are duplicate values:

            - ``first`` : take the first occurrence.
            - ``last`` : take the last occurrence.
            - ``all`` : keep all the ties of the largest item even if it means
              selecting more than ``n`` items.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.nlargest : Return the first `n` rows ordered by `columns` in
            descending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Examples
        --------
        >>> df = pd.DataFrame({\'population\': [59000000, 65000000, 434000,
        ...                                   434000, 434000, 337000, 337000,
        ...                                   11300, 11300],
        ...                    \'GDP\': [1937894, 2583560 , 12011, 4520, 12128,
        ...                            17036, 182, 38, 311],
        ...                    \'alpha-2\': ["IT", "FR", "MT", "MV", "BN",
        ...                                "IS", "NR", "TV", "AI"]},
        ...                   index=["Italy", "France", "Malta",
        ...                          "Maldives", "Brunei", "Iceland",
        ...                          "Nauru", "Tuvalu", "Anguilla"])
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru         337000      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI

        In the following example, we will use ``nsmallest`` to select the
        three rows having the smallest values in column "population".

        >>> df.nsmallest(3, \'population\')
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS

        When using ``keep=\'last\'``, ties are resolved in reverse order:

        >>> df.nsmallest(3, \'population\', keep=\'last\')
                  population  GDP alpha-2
        Anguilla       11300  311      AI
        Tuvalu         11300   38      TV
        Nauru         337000  182      NR

        When using ``keep=\'all\'``, the number of element kept can go beyond ``n``
        if there are duplicate values for the largest element, all the
        ties are kept.

        >>> df.nsmallest(3, \'population\', keep=\'all\')
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS
        Nauru         337000    182      NR

        However, ``nsmallest`` does not keep ``n`` distinct
        smallest elements:

        >>> df.nsmallest(4, \'population\', keep=\'all\')
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS
        Nauru         337000    182      NR

        To order by the smallest values in column "population" and then "GDP", we can
        specify multiple columns like in the next example.

        >>> df.nsmallest(3, [\'population\', \'GDP\'])
                  population  GDP alpha-2
        Tuvalu         11300   38      TV
        Anguilla       11300  311      AI
        Nauru         337000  182      NR
        '''
    def swaplevel(self, i: Axis = ..., j: Axis = ..., axis: Axis = ...) -> DataFrame:
        '''
        Swap levels i and j in a :class:`MultiIndex`.

        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i, j : int or str
            Levels of the indices to be swapped. Can pass level name as string.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
                    The axis to swap levels on. 0 or \'index\' for row-wise, 1 or
                    \'columns\' for column-wise.

        Returns
        -------
        DataFrame
            DataFrame with levels swapped in MultiIndex.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"Grade": ["A", "B", "A", "C"]},
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> df
                                            Grade
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C

        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behaviour.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.

        >>> df.swaplevel()
                                            Grade
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C

        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.

        >>> df.swaplevel(0)
                                            Grade
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C

        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.

        >>> df.swaplevel(0, 1)
                                            Grade
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C
        '''
    def reorder_levels(self, order: Sequence[int | str], axis: Axis = ...) -> DataFrame:
        '''
        Rearrange index levels using input order. May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            Where to reorder levels.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> data = {
        ...     "class": ["Mammals", "Mammals", "Reptiles"],
        ...     "diet": ["Omnivore", "Carnivore", "Carnivore"],
        ...     "species": ["Humans", "Dogs", "Snakes"],
        ... }
        >>> df = pd.DataFrame(data, columns=["class", "diet", "species"])
        >>> df = df.set_index(["class", "diet"])
        >>> df
                                          species
        class      diet
        Mammals    Omnivore                Humans
                   Carnivore                 Dogs
        Reptiles   Carnivore               Snakes

        Let\'s reorder the levels of the index:

        >>> df.reorder_levels(["diet", "class"])
                                          species
        diet      class
        Omnivore  Mammals                  Humans
        Carnivore Mammals                    Dogs
                  Reptiles                 Snakes
        '''
    def _cmp_method(self, other, op): ...
    def _arith_method(self, other, op): ...
    def _logical_method(self, other, op): ...
    def _dispatch_frame_op(self, right, func: Callable, axis: AxisInt | None) -> DataFrame:
        """
        Evaluate the frame operation func(left, right) by evaluating
        column-by-column, dispatching to the Series implementation.

        Parameters
        ----------
        right : scalar, Series, or DataFrame
        func : arithmetic or comparison operator
        axis : {None, 0, 1}

        Returns
        -------
        DataFrame

        Notes
        -----
        Caller is responsible for setting np.errstate where relevant.
        """
    def _combine_frame(self, other: DataFrame, func, fill_value): ...
    def _arith_method_with_reindex(self, right: DataFrame, op) -> DataFrame:
        """
        For DataFrame-with-DataFrame operations that require reindexing,
        operate only on shared columns, then reindex.

        Parameters
        ----------
        right : DataFrame
        op : binary operator

        Returns
        -------
        DataFrame
        """
    def _should_reindex_frame_op(self, right, op, axis: int, fill_value, level) -> bool:
        """
        Check if this is an operation between DataFrames that will need to reindex.
        """
    def _align_for_op(self, other, axis: AxisInt, flex: bool | None = ..., level: Level | None):
        """
        Convert rhs to meet lhs dims if input is list, tuple or np.ndarray.

        Parameters
        ----------
        left : DataFrame
        right : Any
        axis : int
        flex : bool or None, default False
            Whether this is a flex op, in which case we reindex.
            None indicates not to check for alignment.
        level : int or level name, default None

        Returns
        -------
        left : DataFrame
        right : Any
        """
    def _maybe_align_series_as_frame(self, series: Series, axis: AxisInt):
        """
        If the Series operand is not EA-dtype, we can broadcast to 2D and operate
        blockwise.
        """
    def _flex_arith_method(self, other, op, *, axis: Axis = ..., level, fill_value): ...
    def _construct_result(self, result) -> DataFrame:
        """
        Wrap the result of an arithmetic, comparison, or logical operation.

        Parameters
        ----------
        result : DataFrame

        Returns
        -------
        DataFrame
        """
    def __divmod__(self, other) -> tuple[DataFrame, DataFrame]: ...
    def __rdivmod__(self, other) -> tuple[DataFrame, DataFrame]: ...
    def _flex_cmp_method(self, other, op, *, axis: Axis = ..., level): ...
    def eq(self, other, axis: Axis = ..., level) -> DataFrame:
        '''
        Get Equal to of dataframe and other, element-wise (binary operator `eq`).

        Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
        operators.

        Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
        (rows or columns) and level for comparison.

        Parameters
        ----------
        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or \'index\', 1 or \'columns\'}, default \'columns\'
            Whether to compare by the index (0 or \'index\') or columns
            (1 or \'columns\').
        level : int or label
            Broadcast across a level, matching Index values on the passed
            MultiIndex level.

        Returns
        -------
        DataFrame of bool
            Result of the comparison.

        See Also
        --------
        DataFrame.eq : Compare DataFrames for equality elementwise.
        DataFrame.ne : Compare DataFrames for inequality elementwise.
        DataFrame.le : Compare DataFrames for less than inequality
            or equality elementwise.
        DataFrame.lt : Compare DataFrames for strictly less than
            inequality elementwise.
        DataFrame.ge : Compare DataFrames for greater than inequality
            or equality elementwise.
        DataFrame.gt : Compare DataFrames for strictly greater than
            inequality elementwise.

        Notes
        -----
        Mismatched indices will be unioned together.
        `NaN` values are considered different (i.e. `NaN` != `NaN`).

        Examples
        --------
        >>> df = pd.DataFrame({\'cost\': [250, 150, 100],
        ...                    \'revenue\': [100, 250, 300]},
        ...                   index=[\'A\', \'B\', \'C\'])
        >>> df
           cost  revenue
        A   250      100
        B   150      250
        C   100      300

        Comparison with a scalar, using either the operator or method:

        >>> df == 100
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        >>> df.eq(100)
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        When `other` is a :class:`Series`, the columns of a DataFrame are aligned
        with the index of `other` and broadcast:

        >>> df != pd.Series([100, 250], index=["cost", "revenue"])
            cost  revenue
        A   True     True
        B   True    False
        C  False     True

        Use the method to control the broadcast axis:

        >>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis=\'index\')
           cost  revenue
        A  True    False
        B  True     True
        C  True     True
        D  True     True

        When comparing to an arbitrary sequence, the number of columns must
        match the number elements in `other`:

        >>> df == [250, 100]
            cost  revenue
        A   True     True
        B  False    False
        C  False    False

        Use the method to control the axis:

        >>> df.eq([250, 250, 100], axis=\'index\')
            cost  revenue
        A   True    False
        B  False     True
        C   True    False

        Compare to a DataFrame of different shape.

        >>> other = pd.DataFrame({\'revenue\': [300, 250, 100, 150]},
        ...                      index=[\'A\', \'B\', \'C\', \'D\'])
        >>> other
           revenue
        A      300
        B      250
        C      100
        D      150

        >>> df.gt(other)
            cost  revenue
        A  False    False
        B  False    False
        C  False     True
        D  False    False

        Compare to a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({\'cost\': [250, 150, 100, 150, 300, 220],
        ...                              \'revenue\': [100, 250, 300, 200, 175, 225]},
        ...                             index=[[\'Q1\', \'Q1\', \'Q1\', \'Q2\', \'Q2\', \'Q2\'],
        ...                                    [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\']])
        >>> df_multindex
              cost  revenue
        Q1 A   250      100
           B   150      250
           C   100      300
        Q2 A   150      200
           B   300      175
           C   220      225

        >>> df.le(df_multindex, level=1)
               cost  revenue
        Q1 A   True     True
           B   True     True
           C   True     True
        Q2 A  False     True
           B   True    False
           C   True    False
        '''
    def ne(self, other, axis: Axis = ..., level) -> DataFrame:
        '''
        Get Not equal to of dataframe and other, element-wise (binary operator `ne`).

        Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
        operators.

        Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
        (rows or columns) and level for comparison.

        Parameters
        ----------
        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or \'index\', 1 or \'columns\'}, default \'columns\'
            Whether to compare by the index (0 or \'index\') or columns
            (1 or \'columns\').
        level : int or label
            Broadcast across a level, matching Index values on the passed
            MultiIndex level.

        Returns
        -------
        DataFrame of bool
            Result of the comparison.

        See Also
        --------
        DataFrame.eq : Compare DataFrames for equality elementwise.
        DataFrame.ne : Compare DataFrames for inequality elementwise.
        DataFrame.le : Compare DataFrames for less than inequality
            or equality elementwise.
        DataFrame.lt : Compare DataFrames for strictly less than
            inequality elementwise.
        DataFrame.ge : Compare DataFrames for greater than inequality
            or equality elementwise.
        DataFrame.gt : Compare DataFrames for strictly greater than
            inequality elementwise.

        Notes
        -----
        Mismatched indices will be unioned together.
        `NaN` values are considered different (i.e. `NaN` != `NaN`).

        Examples
        --------
        >>> df = pd.DataFrame({\'cost\': [250, 150, 100],
        ...                    \'revenue\': [100, 250, 300]},
        ...                   index=[\'A\', \'B\', \'C\'])
        >>> df
           cost  revenue
        A   250      100
        B   150      250
        C   100      300

        Comparison with a scalar, using either the operator or method:

        >>> df == 100
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        >>> df.eq(100)
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        When `other` is a :class:`Series`, the columns of a DataFrame are aligned
        with the index of `other` and broadcast:

        >>> df != pd.Series([100, 250], index=["cost", "revenue"])
            cost  revenue
        A   True     True
        B   True    False
        C  False     True

        Use the method to control the broadcast axis:

        >>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis=\'index\')
           cost  revenue
        A  True    False
        B  True     True
        C  True     True
        D  True     True

        When comparing to an arbitrary sequence, the number of columns must
        match the number elements in `other`:

        >>> df == [250, 100]
            cost  revenue
        A   True     True
        B  False    False
        C  False    False

        Use the method to control the axis:

        >>> df.eq([250, 250, 100], axis=\'index\')
            cost  revenue
        A   True    False
        B  False     True
        C   True    False

        Compare to a DataFrame of different shape.

        >>> other = pd.DataFrame({\'revenue\': [300, 250, 100, 150]},
        ...                      index=[\'A\', \'B\', \'C\', \'D\'])
        >>> other
           revenue
        A      300
        B      250
        C      100
        D      150

        >>> df.gt(other)
            cost  revenue
        A  False    False
        B  False    False
        C  False     True
        D  False    False

        Compare to a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({\'cost\': [250, 150, 100, 150, 300, 220],
        ...                              \'revenue\': [100, 250, 300, 200, 175, 225]},
        ...                             index=[[\'Q1\', \'Q1\', \'Q1\', \'Q2\', \'Q2\', \'Q2\'],
        ...                                    [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\']])
        >>> df_multindex
              cost  revenue
        Q1 A   250      100
           B   150      250
           C   100      300
        Q2 A   150      200
           B   300      175
           C   220      225

        >>> df.le(df_multindex, level=1)
               cost  revenue
        Q1 A   True     True
           B   True     True
           C   True     True
        Q2 A  False     True
           B   True    False
           C   True    False
        '''
    def le(self, other, axis: Axis = ..., level) -> DataFrame:
        '''
        Get Less than or equal to of dataframe and other, element-wise (binary operator `le`).

        Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
        operators.

        Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
        (rows or columns) and level for comparison.

        Parameters
        ----------
        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or \'index\', 1 or \'columns\'}, default \'columns\'
            Whether to compare by the index (0 or \'index\') or columns
            (1 or \'columns\').
        level : int or label
            Broadcast across a level, matching Index values on the passed
            MultiIndex level.

        Returns
        -------
        DataFrame of bool
            Result of the comparison.

        See Also
        --------
        DataFrame.eq : Compare DataFrames for equality elementwise.
        DataFrame.ne : Compare DataFrames for inequality elementwise.
        DataFrame.le : Compare DataFrames for less than inequality
            or equality elementwise.
        DataFrame.lt : Compare DataFrames for strictly less than
            inequality elementwise.
        DataFrame.ge : Compare DataFrames for greater than inequality
            or equality elementwise.
        DataFrame.gt : Compare DataFrames for strictly greater than
            inequality elementwise.

        Notes
        -----
        Mismatched indices will be unioned together.
        `NaN` values are considered different (i.e. `NaN` != `NaN`).

        Examples
        --------
        >>> df = pd.DataFrame({\'cost\': [250, 150, 100],
        ...                    \'revenue\': [100, 250, 300]},
        ...                   index=[\'A\', \'B\', \'C\'])
        >>> df
           cost  revenue
        A   250      100
        B   150      250
        C   100      300

        Comparison with a scalar, using either the operator or method:

        >>> df == 100
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        >>> df.eq(100)
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        When `other` is a :class:`Series`, the columns of a DataFrame are aligned
        with the index of `other` and broadcast:

        >>> df != pd.Series([100, 250], index=["cost", "revenue"])
            cost  revenue
        A   True     True
        B   True    False
        C  False     True

        Use the method to control the broadcast axis:

        >>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis=\'index\')
           cost  revenue
        A  True    False
        B  True     True
        C  True     True
        D  True     True

        When comparing to an arbitrary sequence, the number of columns must
        match the number elements in `other`:

        >>> df == [250, 100]
            cost  revenue
        A   True     True
        B  False    False
        C  False    False

        Use the method to control the axis:

        >>> df.eq([250, 250, 100], axis=\'index\')
            cost  revenue
        A   True    False
        B  False     True
        C   True    False

        Compare to a DataFrame of different shape.

        >>> other = pd.DataFrame({\'revenue\': [300, 250, 100, 150]},
        ...                      index=[\'A\', \'B\', \'C\', \'D\'])
        >>> other
           revenue
        A      300
        B      250
        C      100
        D      150

        >>> df.gt(other)
            cost  revenue
        A  False    False
        B  False    False
        C  False     True
        D  False    False

        Compare to a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({\'cost\': [250, 150, 100, 150, 300, 220],
        ...                              \'revenue\': [100, 250, 300, 200, 175, 225]},
        ...                             index=[[\'Q1\', \'Q1\', \'Q1\', \'Q2\', \'Q2\', \'Q2\'],
        ...                                    [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\']])
        >>> df_multindex
              cost  revenue
        Q1 A   250      100
           B   150      250
           C   100      300
        Q2 A   150      200
           B   300      175
           C   220      225

        >>> df.le(df_multindex, level=1)
               cost  revenue
        Q1 A   True     True
           B   True     True
           C   True     True
        Q2 A  False     True
           B   True    False
           C   True    False
        '''
    def lt(self, other, axis: Axis = ..., level) -> DataFrame:
        '''
        Get Less than of dataframe and other, element-wise (binary operator `lt`).

        Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
        operators.

        Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
        (rows or columns) and level for comparison.

        Parameters
        ----------
        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or \'index\', 1 or \'columns\'}, default \'columns\'
            Whether to compare by the index (0 or \'index\') or columns
            (1 or \'columns\').
        level : int or label
            Broadcast across a level, matching Index values on the passed
            MultiIndex level.

        Returns
        -------
        DataFrame of bool
            Result of the comparison.

        See Also
        --------
        DataFrame.eq : Compare DataFrames for equality elementwise.
        DataFrame.ne : Compare DataFrames for inequality elementwise.
        DataFrame.le : Compare DataFrames for less than inequality
            or equality elementwise.
        DataFrame.lt : Compare DataFrames for strictly less than
            inequality elementwise.
        DataFrame.ge : Compare DataFrames for greater than inequality
            or equality elementwise.
        DataFrame.gt : Compare DataFrames for strictly greater than
            inequality elementwise.

        Notes
        -----
        Mismatched indices will be unioned together.
        `NaN` values are considered different (i.e. `NaN` != `NaN`).

        Examples
        --------
        >>> df = pd.DataFrame({\'cost\': [250, 150, 100],
        ...                    \'revenue\': [100, 250, 300]},
        ...                   index=[\'A\', \'B\', \'C\'])
        >>> df
           cost  revenue
        A   250      100
        B   150      250
        C   100      300

        Comparison with a scalar, using either the operator or method:

        >>> df == 100
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        >>> df.eq(100)
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        When `other` is a :class:`Series`, the columns of a DataFrame are aligned
        with the index of `other` and broadcast:

        >>> df != pd.Series([100, 250], index=["cost", "revenue"])
            cost  revenue
        A   True     True
        B   True    False
        C  False     True

        Use the method to control the broadcast axis:

        >>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis=\'index\')
           cost  revenue
        A  True    False
        B  True     True
        C  True     True
        D  True     True

        When comparing to an arbitrary sequence, the number of columns must
        match the number elements in `other`:

        >>> df == [250, 100]
            cost  revenue
        A   True     True
        B  False    False
        C  False    False

        Use the method to control the axis:

        >>> df.eq([250, 250, 100], axis=\'index\')
            cost  revenue
        A   True    False
        B  False     True
        C   True    False

        Compare to a DataFrame of different shape.

        >>> other = pd.DataFrame({\'revenue\': [300, 250, 100, 150]},
        ...                      index=[\'A\', \'B\', \'C\', \'D\'])
        >>> other
           revenue
        A      300
        B      250
        C      100
        D      150

        >>> df.gt(other)
            cost  revenue
        A  False    False
        B  False    False
        C  False     True
        D  False    False

        Compare to a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({\'cost\': [250, 150, 100, 150, 300, 220],
        ...                              \'revenue\': [100, 250, 300, 200, 175, 225]},
        ...                             index=[[\'Q1\', \'Q1\', \'Q1\', \'Q2\', \'Q2\', \'Q2\'],
        ...                                    [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\']])
        >>> df_multindex
              cost  revenue
        Q1 A   250      100
           B   150      250
           C   100      300
        Q2 A   150      200
           B   300      175
           C   220      225

        >>> df.le(df_multindex, level=1)
               cost  revenue
        Q1 A   True     True
           B   True     True
           C   True     True
        Q2 A  False     True
           B   True    False
           C   True    False
        '''
    def ge(self, other, axis: Axis = ..., level) -> DataFrame:
        '''
        Get Greater than or equal to of dataframe and other, element-wise (binary operator `ge`).

        Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
        operators.

        Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
        (rows or columns) and level for comparison.

        Parameters
        ----------
        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or \'index\', 1 or \'columns\'}, default \'columns\'
            Whether to compare by the index (0 or \'index\') or columns
            (1 or \'columns\').
        level : int or label
            Broadcast across a level, matching Index values on the passed
            MultiIndex level.

        Returns
        -------
        DataFrame of bool
            Result of the comparison.

        See Also
        --------
        DataFrame.eq : Compare DataFrames for equality elementwise.
        DataFrame.ne : Compare DataFrames for inequality elementwise.
        DataFrame.le : Compare DataFrames for less than inequality
            or equality elementwise.
        DataFrame.lt : Compare DataFrames for strictly less than
            inequality elementwise.
        DataFrame.ge : Compare DataFrames for greater than inequality
            or equality elementwise.
        DataFrame.gt : Compare DataFrames for strictly greater than
            inequality elementwise.

        Notes
        -----
        Mismatched indices will be unioned together.
        `NaN` values are considered different (i.e. `NaN` != `NaN`).

        Examples
        --------
        >>> df = pd.DataFrame({\'cost\': [250, 150, 100],
        ...                    \'revenue\': [100, 250, 300]},
        ...                   index=[\'A\', \'B\', \'C\'])
        >>> df
           cost  revenue
        A   250      100
        B   150      250
        C   100      300

        Comparison with a scalar, using either the operator or method:

        >>> df == 100
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        >>> df.eq(100)
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        When `other` is a :class:`Series`, the columns of a DataFrame are aligned
        with the index of `other` and broadcast:

        >>> df != pd.Series([100, 250], index=["cost", "revenue"])
            cost  revenue
        A   True     True
        B   True    False
        C  False     True

        Use the method to control the broadcast axis:

        >>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis=\'index\')
           cost  revenue
        A  True    False
        B  True     True
        C  True     True
        D  True     True

        When comparing to an arbitrary sequence, the number of columns must
        match the number elements in `other`:

        >>> df == [250, 100]
            cost  revenue
        A   True     True
        B  False    False
        C  False    False

        Use the method to control the axis:

        >>> df.eq([250, 250, 100], axis=\'index\')
            cost  revenue
        A   True    False
        B  False     True
        C   True    False

        Compare to a DataFrame of different shape.

        >>> other = pd.DataFrame({\'revenue\': [300, 250, 100, 150]},
        ...                      index=[\'A\', \'B\', \'C\', \'D\'])
        >>> other
           revenue
        A      300
        B      250
        C      100
        D      150

        >>> df.gt(other)
            cost  revenue
        A  False    False
        B  False    False
        C  False     True
        D  False    False

        Compare to a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({\'cost\': [250, 150, 100, 150, 300, 220],
        ...                              \'revenue\': [100, 250, 300, 200, 175, 225]},
        ...                             index=[[\'Q1\', \'Q1\', \'Q1\', \'Q2\', \'Q2\', \'Q2\'],
        ...                                    [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\']])
        >>> df_multindex
              cost  revenue
        Q1 A   250      100
           B   150      250
           C   100      300
        Q2 A   150      200
           B   300      175
           C   220      225

        >>> df.le(df_multindex, level=1)
               cost  revenue
        Q1 A   True     True
           B   True     True
           C   True     True
        Q2 A  False     True
           B   True    False
           C   True    False
        '''
    def gt(self, other, axis: Axis = ..., level) -> DataFrame:
        '''
        Get Greater than of dataframe and other, element-wise (binary operator `gt`).

        Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
        operators.

        Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
        (rows or columns) and level for comparison.

        Parameters
        ----------
        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or \'index\', 1 or \'columns\'}, default \'columns\'
            Whether to compare by the index (0 or \'index\') or columns
            (1 or \'columns\').
        level : int or label
            Broadcast across a level, matching Index values on the passed
            MultiIndex level.

        Returns
        -------
        DataFrame of bool
            Result of the comparison.

        See Also
        --------
        DataFrame.eq : Compare DataFrames for equality elementwise.
        DataFrame.ne : Compare DataFrames for inequality elementwise.
        DataFrame.le : Compare DataFrames for less than inequality
            or equality elementwise.
        DataFrame.lt : Compare DataFrames for strictly less than
            inequality elementwise.
        DataFrame.ge : Compare DataFrames for greater than inequality
            or equality elementwise.
        DataFrame.gt : Compare DataFrames for strictly greater than
            inequality elementwise.

        Notes
        -----
        Mismatched indices will be unioned together.
        `NaN` values are considered different (i.e. `NaN` != `NaN`).

        Examples
        --------
        >>> df = pd.DataFrame({\'cost\': [250, 150, 100],
        ...                    \'revenue\': [100, 250, 300]},
        ...                   index=[\'A\', \'B\', \'C\'])
        >>> df
           cost  revenue
        A   250      100
        B   150      250
        C   100      300

        Comparison with a scalar, using either the operator or method:

        >>> df == 100
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        >>> df.eq(100)
            cost  revenue
        A  False     True
        B  False    False
        C   True    False

        When `other` is a :class:`Series`, the columns of a DataFrame are aligned
        with the index of `other` and broadcast:

        >>> df != pd.Series([100, 250], index=["cost", "revenue"])
            cost  revenue
        A   True     True
        B   True    False
        C  False     True

        Use the method to control the broadcast axis:

        >>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis=\'index\')
           cost  revenue
        A  True    False
        B  True     True
        C  True     True
        D  True     True

        When comparing to an arbitrary sequence, the number of columns must
        match the number elements in `other`:

        >>> df == [250, 100]
            cost  revenue
        A   True     True
        B  False    False
        C  False    False

        Use the method to control the axis:

        >>> df.eq([250, 250, 100], axis=\'index\')
            cost  revenue
        A   True    False
        B  False     True
        C   True    False

        Compare to a DataFrame of different shape.

        >>> other = pd.DataFrame({\'revenue\': [300, 250, 100, 150]},
        ...                      index=[\'A\', \'B\', \'C\', \'D\'])
        >>> other
           revenue
        A      300
        B      250
        C      100
        D      150

        >>> df.gt(other)
            cost  revenue
        A  False    False
        B  False    False
        C  False     True
        D  False    False

        Compare to a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({\'cost\': [250, 150, 100, 150, 300, 220],
        ...                              \'revenue\': [100, 250, 300, 200, 175, 225]},
        ...                             index=[[\'Q1\', \'Q1\', \'Q1\', \'Q2\', \'Q2\', \'Q2\'],
        ...                                    [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\']])
        >>> df_multindex
              cost  revenue
        Q1 A   250      100
           B   150      250
           C   100      300
        Q2 A   150      200
           B   300      175
           C   220      225

        >>> df.le(df_multindex, level=1)
               cost  revenue
        Q1 A   True     True
           B   True     True
           C   True     True
        Q2 A  False     True
           B   True    False
           C   True    False
        '''
    def add(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Addition of dataframe and other, element-wise (binary operator `add`).

        Equivalent to ``dataframe + other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `radd`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def radd(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Addition of dataframe and other, element-wise (binary operator `radd`).

        Equivalent to ``other + dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `add`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def sub(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Subtraction of dataframe and other, element-wise (binary operator `sub`).

        Equivalent to ``dataframe - other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rsub`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def subtract(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Subtraction of dataframe and other, element-wise (binary operator `sub`).

        Equivalent to ``dataframe - other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rsub`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rsub(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Subtraction of dataframe and other, element-wise (binary operator `rsub`).

        Equivalent to ``other - dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `sub`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def mul(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Multiplication of dataframe and other, element-wise (binary operator `mul`).

        Equivalent to ``dataframe * other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rmul`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def multiply(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Multiplication of dataframe and other, element-wise (binary operator `mul`).

        Equivalent to ``dataframe * other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rmul`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rmul(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Multiplication of dataframe and other, element-wise (binary operator `rmul`).

        Equivalent to ``other * dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `mul`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def truediv(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Floating division of dataframe and other, element-wise (binary operator `truediv`).

        Equivalent to ``dataframe / other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rtruediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def div(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Floating division of dataframe and other, element-wise (binary operator `truediv`).

        Equivalent to ``dataframe / other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rtruediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def divide(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Floating division of dataframe and other, element-wise (binary operator `truediv`).

        Equivalent to ``dataframe / other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rtruediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rtruediv(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Floating division of dataframe and other, element-wise (binary operator `rtruediv`).

        Equivalent to ``other / dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `truediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rdiv(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Floating division of dataframe and other, element-wise (binary operator `rtruediv`).

        Equivalent to ``other / dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `truediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def floordiv(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Integer division of dataframe and other, element-wise (binary operator `floordiv`).

        Equivalent to ``dataframe // other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rfloordiv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rfloordiv(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Integer division of dataframe and other, element-wise (binary operator `rfloordiv`).

        Equivalent to ``other // dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `floordiv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def mod(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Modulo of dataframe and other, element-wise (binary operator `mod`).

        Equivalent to ``dataframe % other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rmod`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rmod(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Modulo of dataframe and other, element-wise (binary operator `rmod`).

        Equivalent to ``other % dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `mod`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def pow(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Exponential power of dataframe and other, element-wise (binary operator `pow`).

        Equivalent to ``dataframe ** other``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `rpow`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def rpow(self, other, axis: Axis = ..., level, fill_value) -> DataFrame:
        """
        Get Exponential power of dataframe and other, element-wise (binary operator `rpow`).

        Equivalent to ``other ** dataframe``, but with support to substitute a fill_value
        for missing data in one of the inputs. With reverse version, `pow`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0 or 'index', 1 or 'columns'}
            Whether to compare by the index (0 or 'index') or columns.
            (1 or 'columns'). For Series input, axis to match Series index on.
        level : int or label
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for
            successful DataFrame alignment, with this value before computation.
            If data in both corresponding DataFrame locations is missing
            the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        See Also
        --------
        DataFrame.add : Add DataFrames.
        DataFrame.sub : Subtract DataFrames.
        DataFrame.mul : Multiply DataFrames.
        DataFrame.div : Divide DataFrames (float division).
        DataFrame.truediv : Divide DataFrames (float division).
        DataFrame.floordiv : Divide DataFrames (integer division).
        DataFrame.mod : Calculate modulo (remainder after division).
        DataFrame.pow : Calculate exponential power.

        Notes
        -----
        Mismatched indices will be unioned together.

        Examples
        --------
        >>> df = pd.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        Add a scalar with operator version which return the same
        results.

        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        Divide by constant with reverse version.

        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        Subtract a list and Series by axis with operator version.

        >>> df - [1, 2]
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub([1, 2], axis='columns')
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        >>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
        ...        axis='index')
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359

        Multiply a dictionary by axis.

        >>> df.mul({'angles': 0, 'degrees': 2})
                    angles  degrees
        circle           0      720
        triangle         0      360
        rectangle        0      720

        >>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
                    angles  degrees
        circle           0        0
        triangle         6      360
        rectangle       12     1080

        Multiply a DataFrame of different shape with operator version.

        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other
                   angles
        circle          0
        triangle        3
        rectangle       4

        >>> df * other
                   angles  degrees
        circle          0      NaN
        triangle        9      NaN
        rectangle      16      NaN

        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0      0.0
        triangle        9      0.0
        rectangle      16      0.0

        Divide by a MultiIndex by level.

        >>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
        ...                              'degrees': [360, 180, 360, 360, 540, 720]},
        ...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
        ...                                    ['circle', 'triangle', 'rectangle',
        ...                                     'square', 'pentagon', 'hexagon']])
        >>> df_multindex
                     angles  degrees
        A circle          0      360
          triangle        3      180
          rectangle       4      360
        B square          4      360
          pentagon        5      540
          hexagon         6      720

        >>> df.div(df_multindex, level=1, fill_value=0)
                     angles  degrees
        A circle        NaN      1.0
          triangle      1.0      1.0
          rectangle     1.0      1.0
        B square        0.0      0.0
          pentagon      0.0      0.0
          hexagon       0.0      0.0
        """
    def compare(self, other: DataFrame, align_axis: Axis = ..., keep_shape: bool = ..., keep_equal: bool = ..., result_names: Suffixes = ...) -> DataFrame:
        '''
        Compare to another DataFrame and show the differences.

        Parameters
        ----------
        other : DataFrame
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
        DataFrame
            DataFrame that shows the differences stacked side by side.

            The resulting index will be a MultiIndex with \'self\' and \'other\'
            stacked alternately at the inner level.

        Raises
        ------
        ValueError
            When the two DataFrames don\'t have identical labels or shape.

        See Also
        --------
        Series.compare : Compare with another Series and show differences.
        DataFrame.equals : Test whether two objects contain the same elements.

        Notes
        -----
        Matching NaNs will not appear as a difference.

        Can only compare identically-labeled
        (i.e. same shape, identical row and column labels) DataFrames

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["a", "a", "b", "b", "a"],
        ...         "col2": [1.0, 2.0, 3.0, np.nan, 5.0],
        ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
        ...     },
        ...     columns=["col1", "col2", "col3"],
        ... )
        >>> df
          col1  col2  col3
        0    a   1.0   1.0
        1    a   2.0   2.0
        2    b   3.0   3.0
        3    b   NaN   4.0
        4    a   5.0   5.0

        >>> df2 = df.copy()
        >>> df2.loc[0, \'col1\'] = \'c\'
        >>> df2.loc[2, \'col3\'] = 4.0
        >>> df2
          col1  col2  col3
        0    c   1.0   1.0
        1    a   2.0   2.0
        2    b   3.0   4.0
        3    b   NaN   4.0
        4    a   5.0   5.0

        Align the differences on columns

        >>> df.compare(df2)
          col1       col3
          self other self other
        0    a     c  NaN   NaN
        2  NaN   NaN  3.0   4.0

        Assign result_names

        >>> df.compare(df2, result_names=("left", "right"))
          col1       col3
          left right left right
        0    a     c  NaN   NaN
        2  NaN   NaN  3.0   4.0

        Stack the differences on rows

        >>> df.compare(df2, align_axis=0)
                col1  col3
        0 self     a   NaN
          other    c   NaN
        2 self   NaN   3.0
          other  NaN   4.0

        Keep the equal values

        >>> df.compare(df2, keep_equal=True)
          col1       col3
          self other self other
        0    a     c  1.0   1.0
        2    b     b  3.0   4.0

        Keep all original rows and columns

        >>> df.compare(df2, keep_shape=True)
          col1       col2       col3
          self other self other self other
        0    a     c  NaN   NaN  NaN   NaN
        1  NaN   NaN  NaN   NaN  NaN   NaN
        2  NaN   NaN  NaN   NaN  3.0   4.0
        3  NaN   NaN  NaN   NaN  NaN   NaN
        4  NaN   NaN  NaN   NaN  NaN   NaN

        Keep all original rows and columns and also all original values

        >>> df.compare(df2, keep_shape=True, keep_equal=True)
          col1       col2       col3
          self other self other self other
        0    a     c  1.0   1.0  1.0   1.0
        1    a     a  2.0   2.0  2.0   2.0
        2    b     b  3.0   3.0  3.0   4.0
        3    b     b  NaN   NaN  4.0   4.0
        4    a     a  5.0   5.0  5.0   5.0
        '''
    def combine(self, other: DataFrame, func: Callable[[Series, Series], Series | Hashable], fill_value, overwrite: bool = ...) -> DataFrame:
        """
        Perform column-wise combine with another DataFrame.

        Combines a DataFrame with `other` DataFrame using `func`
        to element-wise combine columns. The row and column indexes of the
        resulting DataFrame will be the union of the two.

        Parameters
        ----------
        other : DataFrame
            The DataFrame to merge column-wise.
        func : function
            Function that takes two series as inputs and return a Series or a
            scalar. Used to merge the two dataframes column by columns.
        fill_value : scalar value, default None
            The value to fill NaNs with prior to passing any column to the
            merge func.
        overwrite : bool, default True
            If True, columns in `self` that do not exist in `other` will be
            overwritten with NaNs.

        Returns
        -------
        DataFrame
            Combination of the provided DataFrames.

        See Also
        --------
        DataFrame.combine_first : Combine two DataFrame objects and default to
            non-null values in frame calling the method.

        Examples
        --------
        Combine using a simple function that chooses the smaller column.

        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        >>> take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
        >>> df1.combine(df2, take_smaller)
           A  B
        0  0  3
        1  0  3

        Example using a true element-wise combine function.

        >>> df1 = pd.DataFrame({'A': [5, 0], 'B': [2, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        >>> df1.combine(df2, np.minimum)
           A  B
        0  1  2
        1  0  3

        Using `fill_value` fills Nones prior to passing the column to the
        merge function.

        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [None, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        >>> df1.combine(df2, take_smaller, fill_value=-5)
           A    B
        0  0 -5.0
        1  0  4.0

        However, if the same element in both dataframes is None, that None
        is preserved

        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [None, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [None, 3]})
        >>> df1.combine(df2, take_smaller, fill_value=-5)
            A    B
        0  0 -5.0
        1  0  3.0

        Example that demonstrates the use of `overwrite` and behavior when
        the axis differ between the dataframes.

        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
        >>> df2 = pd.DataFrame({'B': [3, 3], 'C': [-10, 1], }, index=[1, 2])
        >>> df1.combine(df2, take_smaller)
             A    B     C
        0  NaN  NaN   NaN
        1  NaN  3.0 -10.0
        2  NaN  3.0   1.0

        >>> df1.combine(df2, take_smaller, overwrite=False)
             A    B     C
        0  0.0  NaN   NaN
        1  0.0  3.0 -10.0
        2  NaN  3.0   1.0

        Demonstrating the preference of the passed in dataframe.

        >>> df2 = pd.DataFrame({'B': [3, 3], 'C': [1, 1], }, index=[1, 2])
        >>> df2.combine(df1, take_smaller)
           A    B   C
        0  0.0  NaN NaN
        1  0.0  3.0 NaN
        2  NaN  3.0 NaN

        >>> df2.combine(df1, take_smaller, overwrite=False)
             A    B   C
        0  0.0  NaN NaN
        1  0.0  3.0 1.0
        2  NaN  3.0 1.0
        """
    def combine_first(self, other: DataFrame) -> DataFrame:
        """
        Update null elements with value in the same location in `other`.

        Combine two DataFrame objects by filling null values in one DataFrame
        with non-null values from other DataFrame. The row and column indexes
        of the resulting DataFrame will be the union of the two. The resulting
        dataframe contains the 'first' dataframe values and overrides the
        second one values where both first.loc[index, col] and
        second.loc[index, col] are not missing values, upon calling
        first.combine_first(second).

        Parameters
        ----------
        other : DataFrame
            Provided DataFrame to use to fill null values.

        Returns
        -------
        DataFrame
            The result of combining the provided DataFrame with the other object.

        See Also
        --------
        DataFrame.combine : Perform series-wise operation on two DataFrames
            using a given function.

        Examples
        --------
        >>> df1 = pd.DataFrame({'A': [None, 0], 'B': [None, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        >>> df1.combine_first(df2)
             A    B
        0  1.0  3.0
        1  0.0  4.0

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> df1 = pd.DataFrame({'A': [None, 0], 'B': [4, None]})
        >>> df2 = pd.DataFrame({'B': [3, 3], 'C': [1, 1]}, index=[1, 2])
        >>> df1.combine_first(df2)
             A    B    C
        0  NaN  4.0  NaN
        1  0.0  3.0  1.0
        2  NaN  3.0  1.0
        """
    def update(self, other, join: UpdateJoin = ..., overwrite: bool = ..., filter_func, errors: IgnoreRaise = ...) -> None:
        """
        Modify in place using non-NA values from another DataFrame.

        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or object coercible into a DataFrame
            Should have at least one matching index/column label
            with the original DataFrame. If a Series is passed,
            its name attribute must be set, and that will be
            used as the column name to align with the original DataFrame.
        join : {'left'}, default 'left'
            Only left join is implemented, keeping the index and columns of the
            original object.
        overwrite : bool, default True
            How to handle non-NA values for overlapping keys:

            * True: overwrite original DataFrame's values
              with values from `other`.
            * False: only update values that are NA in
              the original DataFrame.

        filter_func : callable(1d-array) -> bool 1d-array, optional
            Can choose to replace values other than NA. Return True for values
            that should be updated.
        errors : {'raise', 'ignore'}, default 'ignore'
            If 'raise', will raise a ValueError if the DataFrame and `other`
            both contain non-NA data in the same place.

        Returns
        -------
        None
            This method directly changes calling object.

        Raises
        ------
        ValueError
            * When `errors='raise'` and there's overlapping non-NA data.
            * When `errors` is not either `'ignore'` or `'raise'`
        NotImplementedError
            * If `join != 'left'`

        See Also
        --------
        dict.update : Similar method for dictionaries.
        DataFrame.merge : For column(s)-on-column(s) operations.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3],
        ...                    'B': [400, 500, 600]})
        >>> new_df = pd.DataFrame({'B': [4, 5, 6],
        ...                        'C': [7, 8, 9]})
        >>> df.update(new_df)
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6

        The DataFrame's length does not increase as a result of the update,
        only values at matching index/column labels are updated.

        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'],
        ...                    'B': ['x', 'y', 'z']})
        >>> new_df = pd.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']})
        >>> df.update(new_df)
        >>> df
           A  B
        0  a  d
        1  b  e
        2  c  f

        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'],
        ...                    'B': ['x', 'y', 'z']})
        >>> new_df = pd.DataFrame({'B': ['d', 'f']}, index=[0, 2])
        >>> df.update(new_df)
        >>> df
           A  B
        0  a  d
        1  b  y
        2  c  f

        For Series, its name attribute must be set.

        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'],
        ...                    'B': ['x', 'y', 'z']})
        >>> new_column = pd.Series(['d', 'e', 'f'], name='B')
        >>> df.update(new_column)
        >>> df
           A  B
        0  a  d
        1  b  e
        2  c  f

        If `other` contains NaNs the corresponding values are not updated
        in the original dataframe.

        >>> df = pd.DataFrame({'A': [1, 2, 3],
        ...                    'B': [400., 500., 600.]})
        >>> new_df = pd.DataFrame({'B': [4, np.nan, 6]})
        >>> df.update(new_df)
        >>> df
           A      B
        0  1    4.0
        1  2  500.0
        2  3    6.0
        """
    def groupby(self, by, axis: Axis | lib.NoDefault = ..., level: IndexLabel | None, as_index: bool = ..., sort: bool = ..., group_keys: bool = ..., observed: bool | lib.NoDefault = ..., dropna: bool = ...) -> DataFrameGroupBy:
        '''
        Group DataFrame using a mapper or by a Series of columns.

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
        pandas.api.typing.DataFrameGroupBy
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
        >>> df = pd.DataFrame({\'Animal\': [\'Falcon\', \'Falcon\',
        ...                               \'Parrot\', \'Parrot\'],
        ...                    \'Max Speed\': [380., 370., 24., 26.]})
        >>> df
           Animal  Max Speed
        0  Falcon      380.0
        1  Falcon      370.0
        2  Parrot       24.0
        3  Parrot       26.0
        >>> df.groupby([\'Animal\']).mean()
                Max Speed
        Animal
        Falcon      375.0
        Parrot       25.0

        **Hierarchical Indexes**

        We can groupby different levels of a hierarchical index
        using the `level` parameter:

        >>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],
        ...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))
        >>> df = pd.DataFrame({\'Max Speed\': [390., 350., 30., 20.]},
        ...                   index=index)
        >>> df
                        Max Speed
        Animal Type
        Falcon Captive      390.0
               Wild         350.0
        Parrot Captive       30.0
               Wild          20.0
        >>> df.groupby(level=0).mean()
                Max Speed
        Animal
        Falcon      370.0
        Parrot       25.0
        >>> df.groupby(level="Type").mean()
                 Max Speed
        Type
        Captive      210.0
        Wild         185.0

        We can also choose to include NA in group keys or not by setting
        `dropna` parameter, the default setting is `True`.

        >>> l = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]
        >>> df = pd.DataFrame(l, columns=["a", "b", "c"])

        >>> df.groupby(by=["b"]).sum()
            a   c
        b
        1.0 2   3
        2.0 2   5

        >>> df.groupby(by=["b"], dropna=False).sum()
            a   c
        b
        1.0 2   3
        2.0 2   5
        NaN 1   4

        >>> l = [["a", 12, 12], [None, 12.3, 33.], ["b", 12.3, 123], ["a", 1, 1]]
        >>> df = pd.DataFrame(l, columns=["a", "b", "c"])

        >>> df.groupby(by="a").sum()
            b     c
        a
        a   13.0   13.0
        b   12.3  123.0

        >>> df.groupby(by="a", dropna=False).sum()
            b     c
        a
        a   13.0   13.0
        b   12.3  123.0
        NaN 12.3   33.0

        When using ``.apply()``, use ``group_keys`` to include or exclude the
        group keys. The ``group_keys`` argument defaults to ``True`` (include).

        >>> df = pd.DataFrame({\'Animal\': [\'Falcon\', \'Falcon\',
        ...                               \'Parrot\', \'Parrot\'],
        ...                    \'Max Speed\': [380., 370., 24., 26.]})
        >>> df.groupby("Animal", group_keys=True)[[\'Max Speed\']].apply(lambda x: x)
                  Max Speed
        Animal
        Falcon 0      380.0
               1      370.0
        Parrot 2       24.0
               3       26.0

        >>> df.groupby("Animal", group_keys=False)[[\'Max Speed\']].apply(lambda x: x)
           Max Speed
        0      380.0
        1      370.0
        2       24.0
        3       26.0
        '''
    def pivot(self, *, columns, index: pandas._libs.lib._NoDefault = ..., values: pandas._libs.lib._NoDefault = ...) -> DataFrame:
        '''
        Return reshaped DataFrame organized by given index / column values.

        Reshape data (produce a "pivot" table) based on column values. Uses
        unique values from specified `index` / `columns` to form axes of the
        resulting DataFrame. This function does not support data
        aggregation, multiple values will result in a MultiIndex in the
        columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

        Parameters
        ----------
        columns : str or object or a list of str
            Column to use to make new frame\'s columns.
        index : str or object or a list of str, optional
            Column to use to make new frame\'s index. If not given, uses existing index.
        values : str, object or a list of the previous, optional
            Column(s) to use for populating new frame\'s values. If not
            specified, all remaining columns will be used and the result will
            have hierarchically indexed columns.

        Returns
        -------
        DataFrame
            Returns reshaped DataFrame.

        Raises
        ------
        ValueError:
            When there are any `index`, `columns` combinations with multiple
            values. `DataFrame.pivot_table` when you need to aggregate.

        See Also
        --------
        DataFrame.pivot_table : Generalization of pivot that can handle
            duplicate values for one index/column pair.
        DataFrame.unstack : Pivot based on the index values instead of a
            column.
        wide_to_long : Wide panel to long format. Less flexible but more
            user-friendly than melt.

        Notes
        -----
        For finer-tuned control, see hierarchical indexing documentation along
        with the related stack/unstack methods.

        Reference :ref:`the user guide <reshaping.pivot>` for more examples.

        Examples
        --------
        >>> df = pd.DataFrame({\'foo\': [\'one\', \'one\', \'one\', \'two\', \'two\',
        ...                            \'two\'],
        ...                    \'bar\': [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\'],
        ...                    \'baz\': [1, 2, 3, 4, 5, 6],
        ...                    \'zoo\': [\'x\', \'y\', \'z\', \'q\', \'w\', \'t\']})
        >>> df
            foo   bar  baz  zoo
        0   one   A    1    x
        1   one   B    2    y
        2   one   C    3    z
        3   two   A    4    q
        4   two   B    5    w
        5   two   C    6    t

        >>> df.pivot(index=\'foo\', columns=\'bar\', values=\'baz\')
        bar  A   B   C
        foo
        one  1   2   3
        two  4   5   6

        >>> df.pivot(index=\'foo\', columns=\'bar\')[\'baz\']
        bar  A   B   C
        foo
        one  1   2   3
        two  4   5   6

        >>> df.pivot(index=\'foo\', columns=\'bar\', values=[\'baz\', \'zoo\'])
              baz       zoo
        bar   A  B  C   A  B  C
        foo
        one   1  2  3   x  y  z
        two   4  5  6   q  w  t

        You could also assign a list of column names or a list of index names.

        >>> df = pd.DataFrame({
        ...        "lev1": [1, 1, 1, 2, 2, 2],
        ...        "lev2": [1, 1, 2, 1, 1, 2],
        ...        "lev3": [1, 2, 1, 2, 1, 2],
        ...        "lev4": [1, 2, 3, 4, 5, 6],
        ...        "values": [0, 1, 2, 3, 4, 5]})
        >>> df
            lev1 lev2 lev3 lev4 values
        0   1    1    1    1    0
        1   1    1    2    2    1
        2   1    2    1    3    2
        3   2    1    2    4    3
        4   2    1    1    5    4
        5   2    2    2    6    5

        >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
        lev2    1         2
        lev3    1    2    1    2
        lev1
        1     0.0  1.0  2.0  NaN
        2     4.0  3.0  NaN  5.0

        >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")
              lev3    1    2
        lev1  lev2
           1     1  0.0  1.0
                 2  2.0  NaN
           2     1  4.0  3.0
                 2  NaN  5.0

        A ValueError is raised if there are any duplicates.

        >>> df = pd.DataFrame({"foo": [\'one\', \'one\', \'two\', \'two\'],
        ...                    "bar": [\'A\', \'A\', \'B\', \'C\'],
        ...                    "baz": [1, 2, 3, 4]})
        >>> df
           foo bar  baz
        0  one   A    1
        1  one   A    2
        2  two   B    3
        3  two   C    4

        Notice that the first two rows are the same for our `index`
        and `columns` arguments.

        >>> df.pivot(index=\'foo\', columns=\'bar\', values=\'baz\')
        Traceback (most recent call last):
           ...
        ValueError: Index contains duplicate entries, cannot reshape
        '''
    def pivot_table(self, values, index, columns, aggfunc: AggFuncType = ..., fill_value, margins: bool = ..., dropna: bool = ..., margins_name: Level = ..., observed: bool | lib.NoDefault = ..., sort: bool = ...) -> DataFrame:
        '''
        Create a spreadsheet-style pivot table as a DataFrame.

        The levels in the pivot table will be stored in MultiIndex objects
        (hierarchical indexes) on the index and columns of the result DataFrame.

        Parameters
        ----------
        values : list-like or scalar, optional
            Column or columns to aggregate.
        index : column, Grouper, array, or list of the previous
            Keys to group by on the pivot table index. If a list is passed,
            it can contain any of the other types (except list). If an array is
            passed, it must be the same length as the data and will be used in
            the same manner as column values.
        columns : column, Grouper, array, or list of the previous
            Keys to group by on the pivot table column. If a list is passed,
            it can contain any of the other types (except list). If an array is
            passed, it must be the same length as the data and will be used in
            the same manner as column values.
        aggfunc : function, list of functions, dict, default "mean"
            If a list of functions is passed, the resulting pivot table will have
            hierarchical columns whose top level are the function names
            (inferred from the function objects themselves).
            If a dict is passed, the key is column to aggregate and the value is
            function or list of functions. If ``margin=True``, aggfunc will be
            used to calculate the partial aggregates.
        fill_value : scalar, default None
            Value to replace missing values with (in the resulting pivot table,
            after aggregation).
        margins : bool, default False
            If ``margins=True``, special ``All`` columns and rows
            will be added with partial group aggregates across the categories
            on the rows and columns.
        dropna : bool, default True
            Do not include columns whose entries are all NaN. If True,
            rows with a NaN value in any column will be omitted before
            computing margins.
        margins_name : str, default \'All\'
            Name of the row / column that will contain the totals
            when margins is True.
        observed : bool, default False
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.

            .. deprecated:: 2.2.0

                The default value of ``False`` is deprecated and will change to
                ``True`` in a future version of pandas.

        sort : bool, default True
            Specifies if the result should be sorted.

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame
            An Excel style pivot table.

        See Also
        --------
        DataFrame.pivot : Pivot without aggregation that can handle
            non-numeric data.
        DataFrame.melt: Unpivot a DataFrame from wide to long format,
            optionally leaving identifiers set.
        wide_to_long : Wide panel to long format. Less flexible but more
            user-friendly than melt.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.pivot>` for more examples.

        Examples
        --------
        >>> df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
        ...                          "bar", "bar", "bar", "bar"],
        ...                    "B": ["one", "one", "one", "two", "two",
        ...                          "one", "one", "two", "two"],
        ...                    "C": ["small", "large", "large", "small",
        ...                          "small", "large", "small", "small",
        ...                          "large"],
        ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
        ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        >>> df
             A    B      C  D  E
        0  foo  one  small  1  2
        1  foo  one  large  2  4
        2  foo  one  large  2  5
        3  foo  two  small  3  5
        4  foo  two  small  3  6
        5  bar  one  large  4  6
        6  bar  one  small  5  8
        7  bar  two  small  6  9
        8  bar  two  large  7  9

        This first example aggregates values by taking the sum.

        >>> table = pd.pivot_table(df, values=\'D\', index=[\'A\', \'B\'],
        ...                        columns=[\'C\'], aggfunc="sum")
        >>> table
        C        large  small
        A   B
        bar one    4.0    5.0
            two    7.0    6.0
        foo one    4.0    1.0
            two    NaN    6.0

        We can also fill missing values using the `fill_value` parameter.

        >>> table = pd.pivot_table(df, values=\'D\', index=[\'A\', \'B\'],
        ...                        columns=[\'C\'], aggfunc="sum", fill_value=0)
        >>> table
        C        large  small
        A   B
        bar one      4      5
            two      7      6
        foo one      4      1
            two      0      6

        The next example aggregates by taking the mean across multiple columns.

        >>> table = pd.pivot_table(df, values=[\'D\', \'E\'], index=[\'A\', \'C\'],
        ...                        aggfunc={\'D\': "mean", \'E\': "mean"})
        >>> table
                        D         E
        A   C
        bar large  5.500000  7.500000
            small  5.500000  8.500000
        foo large  2.000000  4.500000
            small  2.333333  4.333333

        We can also calculate multiple types of aggregations for any given
        value column.

        >>> table = pd.pivot_table(df, values=[\'D\', \'E\'], index=[\'A\', \'C\'],
        ...                        aggfunc={\'D\': "mean",
        ...                                 \'E\': ["min", "max", "mean"]})
        >>> table
                          D   E
                       mean max      mean  min
        A   C
        bar large  5.500000   9  7.500000    6
            small  5.500000   9  8.500000    8
        foo large  2.000000   5  4.500000    4
            small  2.333333   6  4.333333    2
        '''
    def stack(self, level: IndexLabel = ..., dropna: bool | lib.NoDefault = ..., sort: bool | lib.NoDefault = ..., future_stack: bool = ...):
        """
        Stack the prescribed level(s) from columns to index.

        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to the current
        DataFrame. The new inner-most levels are created by pivoting the
        columns of the current dataframe:

          - if the columns have a single level, the output is a Series;
          - if the columns have multiple levels, the new index
            level(s) is (are) taken from the prescribed level(s) and
            the output is a DataFrame.

        Parameters
        ----------
        level : int, str, list, default -1
            Level(s) to stack from the column axis onto the index
            axis, defined as one index or label, or a list of indices
            or labels.
        dropna : bool, default True
            Whether to drop rows in the resulting Frame/Series with
            missing values. Stacking a column level onto the index
            axis can create combinations of index and column values
            that are missing from the original dataframe. See Examples
            section.
        sort : bool, default True
            Whether to sort the levels of the resulting MultiIndex.
        future_stack : bool, default False
            Whether to use the new implementation that will replace the current
            implementation in pandas 3.0. When True, dropna and sort have no impact
            on the result and must remain unspecified. See :ref:`pandas 2.1.0 Release
            notes <whatsnew_210.enhancements.new_stack>` for more details.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.

        See Also
        --------
        DataFrame.unstack : Unstack prescribed level(s) from index axis
             onto column axis.
        DataFrame.pivot : Reshape dataframe from long format to wide
             format.
        DataFrame.pivot_table : Create a spreadsheet-style pivot table
             as a DataFrame.

        Notes
        -----
        The function is named by analogy with a collection of books
        being reorganized from being side by side on a horizontal
        position (the columns of the dataframe) to being stacked
        vertically on top of each other (in the index of the
        dataframe).

        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        **Single level columns**

        >>> df_single_level_cols = pd.DataFrame([[0, 1], [2, 3]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=['weight', 'height'])

        Stacking a dataframe with a single level column axis returns a Series:

        >>> df_single_level_cols
             weight height
        cat       0      1
        dog       2      3
        >>> df_single_level_cols.stack(future_stack=True)
        cat  weight    0
             height    1
        dog  weight    2
             height    3
        dtype: int64

        **Multi level columns: simple case**

        >>> multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('weight', 'pounds')])
        >>> df_multi_level_cols1 = pd.DataFrame([[1, 2], [2, 4]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol1)

        Stacking a dataframe with a multi-level column axis:

        >>> df_multi_level_cols1
             weight
                 kg    pounds
        cat       1        2
        dog       2        4
        >>> df_multi_level_cols1.stack(future_stack=True)
                    weight
        cat kg           1
            pounds       2
        dog kg           2
            pounds       4

        **Missing values**

        >>> multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('height', 'm')])
        >>> df_multi_level_cols2 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol2)

        It is common to have missing values when stacking a dataframe
        with multi-level columns, as the stacked dataframe typically
        has more values than the original dataframe. Missing values
        are filled with NaNs:

        >>> df_multi_level_cols2
            weight height
                kg      m
        cat    1.0    2.0
        dog    3.0    4.0
        >>> df_multi_level_cols2.stack(future_stack=True)
                weight  height
        cat kg     1.0     NaN
            m      NaN     2.0
        dog kg     3.0     NaN
            m      NaN     4.0

        **Prescribing the level(s) to be stacked**

        The first parameter controls which level or levels are stacked:

        >>> df_multi_level_cols2.stack(0, future_stack=True)
                     kg    m
        cat weight  1.0  NaN
            height  NaN  2.0
        dog weight  3.0  NaN
            height  NaN  4.0
        >>> df_multi_level_cols2.stack([0, 1], future_stack=True)
        cat  weight  kg    1.0
             height  m     2.0
        dog  weight  kg    3.0
             height  m     4.0
        dtype: float64
        """
    def explode(self, column: IndexLabel, ignore_index: bool = ...) -> DataFrame:
        """
        Transform each element of a list-like to a row, replicating index values.

        Parameters
        ----------
        column : IndexLabel
            Column(s) to explode.
            For multiple columns, specify a non-empty list with each element
            be str or tuple, and all specified columns their list-like data
            on same row of the frame must have matching length.

            .. versionadded:: 1.3.0
                Multi-column explode

        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame
            Exploded lists to rows of the subset columns;
            index will be duplicated for these rows.

        Raises
        ------
        ValueError :
            * If columns of the frame are not unique.
            * If specified columns to explode is empty list.
            * If specified columns to explode have not matching count of
              elements rowwise in the frame.

        See Also
        --------
        DataFrame.unstack : Pivot a level of the (necessarily hierarchical)
            index labels.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        Series.explode : Explode a DataFrame from list-like columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of rows in the
        output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [[0, 1, 2], 'foo', [], [3, 4]],
        ...                    'B': 1,
        ...                    'C': [['a', 'b', 'c'], np.nan, [], ['d', 'e']]})
        >>> df
                   A  B          C
        0  [0, 1, 2]  1  [a, b, c]
        1        foo  1        NaN
        2         []  1         []
        3     [3, 4]  1     [d, e]

        Single-column explode.

        >>> df.explode('A')
             A  B          C
        0    0  1  [a, b, c]
        0    1  1  [a, b, c]
        0    2  1  [a, b, c]
        1  foo  1        NaN
        2  NaN  1         []
        3    3  1     [d, e]
        3    4  1     [d, e]

        Multi-column explode.

        >>> df.explode(list('AC'))
             A  B    C
        0    0  1    a
        0    1  1    b
        0    2  1    c
        1  foo  1  NaN
        2  NaN  1  NaN
        3    3  1    d
        3    4  1    e
        """
    def unstack(self, level: IndexLabel = ..., fill_value, sort: bool = ...):
        """
        Pivot a level of the (necessarily hierarchical) index labels.

        Returns a DataFrame having a new level of column labels whose inner-most level
        consists of the pivoted index labels.

        If the index is not a MultiIndex, the output will be a Series
        (the analogue of stack when the columns are not a MultiIndex).

        Parameters
        ----------
        level : int, str, or list of these, default -1 (last level)
            Level(s) of index to unstack, can pass level name.
        fill_value : int, str or dict
            Replace NaN with this value if the unstack produces missing values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        DataFrame.pivot : Pivot a table based on column values.
        DataFrame.stack : Pivot a level of the column labels (inverse operation
            from `unstack`).

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
        ...                                    ('two', 'a'), ('two', 'b')])
        >>> s = pd.Series(np.arange(1.0, 5.0), index=index)
        >>> s
        one  a   1.0
             b   2.0
        two  a   3.0
             b   4.0
        dtype: float64

        >>> s.unstack(level=-1)
             a   b
        one  1.0  2.0
        two  3.0  4.0

        >>> s.unstack(level=0)
           one  two
        a  1.0   3.0
        b  2.0   4.0

        >>> df = s.unstack(level=0)
        >>> df.unstack()
        one  a  1.0
             b  2.0
        two  a  3.0
             b  4.0
        dtype: float64
        """
    def melt(self, id_vars, value_vars, var_name, value_name: Hashable = ..., col_level: Level | None, ignore_index: bool = ...) -> DataFrame:
        '''
        Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

        This function is useful to massage a DataFrame into a format where one
        or more columns are identifier variables (`id_vars`), while all other
        columns, considered measured variables (`value_vars`), are "unpivoted" to
        the row axis, leaving just two non-identifier columns, \'variable\' and
        \'value\'.

        Parameters
        ----------
        id_vars : scalar, tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        value_vars : scalar, tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that
            are not set as `id_vars`.
        var_name : scalar, default None
            Name to use for the \'variable\' column. If None it uses
            ``frame.columns.name`` or \'variable\'.
        value_name : scalar, default \'value\'
            Name to use for the \'value\' column, can\'t be an existing column label.
        col_level : scalar, optional
            If columns are a MultiIndex then use this level to melt.
        ignore_index : bool, default True
            If True, original index is ignored. If False, the original index is retained.
            Index labels will be repeated as necessary.

        Returns
        -------
        DataFrame
            Unpivoted DataFrame.

        See Also
        --------
        melt : Identical method.
        pivot_table : Create a spreadsheet-style pivot table as a DataFrame.
        DataFrame.pivot : Return reshaped DataFrame organized
            by given index / column values.
        DataFrame.explode : Explode a DataFrame from list-like
                columns to long format.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.melt>` for more examples.

        Examples
        --------
        >>> df = pd.DataFrame({\'A\': {0: \'a\', 1: \'b\', 2: \'c\'},
        ...                    \'B\': {0: 1, 1: 3, 2: 5},
        ...                    \'C\': {0: 2, 1: 4, 2: 6}})
        >>> df
           A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> df.melt(id_vars=[\'A\'], value_vars=[\'B\'])
           A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5

        >>> df.melt(id_vars=[\'A\'], value_vars=[\'B\', \'C\'])
           A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
        3  a        C      2
        4  b        C      4
        5  c        C      6

        The names of \'variable\' and \'value\' columns can be customized:

        >>> df.melt(id_vars=[\'A\'], value_vars=[\'B\'],
        ...         var_name=\'myVarname\', value_name=\'myValname\')
           A myVarname  myValname
        0  a         B          1
        1  b         B          3
        2  c         B          5

        Original index values can be kept around:

        >>> df.melt(id_vars=[\'A\'], value_vars=[\'B\', \'C\'], ignore_index=False)
           A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
        0  a        C      2
        1  b        C      4
        2  c        C      6

        If you have multi-index columns:

        >>> df.columns = [list(\'ABC\'), list(\'DEF\')]
        >>> df
           A  B  C
           D  E  F
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> df.melt(col_level=0, id_vars=[\'A\'], value_vars=[\'B\'])
           A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5

        >>> df.melt(id_vars=[(\'A\', \'D\')], value_vars=[(\'B\', \'E\')])
          (A, D) variable_0 variable_1  value
        0      a          B          E      1
        1      b          B          E      3
        2      c          B          E      5
        '''
    def diff(self, periods: int = ..., axis: Axis = ...) -> DataFrame:
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another
        element in the DataFrame (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative
            values.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Take difference over rows (0) or columns (1).

        Returns
        -------
        DataFrame
            First differences of the Series.

        See Also
        --------
        DataFrame.pct_change: Percent change over given number of periods.
        DataFrame.shift: Shift index by desired number of periods with an
            optional time freq.
        Series.diff: First discrete difference of object.

        Notes
        -----
        For boolean dtypes, this uses :meth:`operator.xor` rather than
        :meth:`operator.sub`.
        The result is calculated according to current dtype in DataFrame,
        however dtype of the result is always float64.

        Examples
        --------

        Difference with previous row

        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8],
        ...                    'c': [1, 4, 9, 16, 25, 36]})
        >>> df
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36

        >>> df.diff()
             a    b     c
        0  NaN  NaN   NaN
        1  1.0  0.0   3.0
        2  1.0  1.0   5.0
        3  1.0  1.0   7.0
        4  1.0  2.0   9.0
        5  1.0  3.0  11.0

        Difference with previous column

        >>> df.diff(axis=1)
            a  b   c
        0 NaN  0   0
        1 NaN -1   3
        2 NaN -1   7
        3 NaN -1  13
        4 NaN  0  20
        5 NaN  2  28

        Difference with 3rd previous row

        >>> df.diff(periods=3)
             a    b     c
        0  NaN  NaN   NaN
        1  NaN  NaN   NaN
        2  NaN  NaN   NaN
        3  3.0  2.0  15.0
        4  3.0  4.0  21.0
        5  3.0  6.0  27.0

        Difference with following row

        >>> df.diff(periods=-1)
             a    b     c
        0 -1.0  0.0  -3.0
        1 -1.0 -1.0  -5.0
        2 -1.0 -1.0  -7.0
        3 -1.0 -2.0  -9.0
        4 -1.0 -3.0 -11.0
        5  NaN  NaN   NaN

        Overflow in input dtype

        >>> df = pd.DataFrame({'a': [1, 0]}, dtype=np.uint8)
        >>> df.diff()
               a
        0    NaN
        1  255.0
        """
    def _gotitem(self, key: IndexLabel, ndim: int, subset: DataFrame | Series | None) -> DataFrame | Series:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
    def aggregate(self, func, axis: Axis = ..., *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, \'mean\']``
            - dict of axis labels -> functions, function names or list of such.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
                If 0 or \'index\': apply function to each column.
                If 1 or \'columns\': apply function to each row.
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
        DataFrame.apply : Perform any type of operations.
        DataFrame.transform : Perform transformation type operations.
        pandas.DataFrame.groupby : Perform operations over groups.
        pandas.DataFrame.resample : Perform operations over resampled bins.
        pandas.DataFrame.rolling : Perform operations over rolling window.
        pandas.DataFrame.expanding : Perform operations over expanding window.
        pandas.core.window.ewm.ExponentialMovingWindow : Perform operation over exponential
            weighted window.

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
        >>> df = pd.DataFrame([[1, 2, 3],
        ...                    [4, 5, 6],
        ...                    [7, 8, 9],
        ...                    [np.nan, np.nan, np.nan]],
        ...                   columns=[\'A\', \'B\', \'C\'])

        Aggregate these functions over the rows.

        >>> df.agg([\'sum\', \'min\'])
                A     B     C
        sum  12.0  15.0  18.0
        min   1.0   2.0   3.0

        Different aggregations per column.

        >>> df.agg({\'A\' : [\'sum\', \'min\'], \'B\' : [\'min\', \'max\']})
                A    B
        sum  12.0  NaN
        min   1.0  2.0
        max   NaN  8.0

        Aggregate different functions over the columns and rename the index of the resulting
        DataFrame.

        >>> df.agg(x=(\'A\', \'max\'), y=(\'B\', \'min\'), z=(\'C\', \'mean\'))
             A    B    C
        x  7.0  NaN  NaN
        y  NaN  2.0  NaN
        z  NaN  NaN  6.0

        Aggregate over the columns.

        >>> df.agg("mean", axis="columns")
        0    2.0
        1    5.0
        2    8.0
        3    NaN
        dtype: float64
        '''
    def agg(self, func, axis: Axis = ..., *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, \'mean\']``
            - dict of axis labels -> functions, function names or list of such.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
                If 0 or \'index\': apply function to each column.
                If 1 or \'columns\': apply function to each row.
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
        DataFrame.apply : Perform any type of operations.
        DataFrame.transform : Perform transformation type operations.
        pandas.DataFrame.groupby : Perform operations over groups.
        pandas.DataFrame.resample : Perform operations over resampled bins.
        pandas.DataFrame.rolling : Perform operations over rolling window.
        pandas.DataFrame.expanding : Perform operations over expanding window.
        pandas.core.window.ewm.ExponentialMovingWindow : Perform operation over exponential
            weighted window.

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
        >>> df = pd.DataFrame([[1, 2, 3],
        ...                    [4, 5, 6],
        ...                    [7, 8, 9],
        ...                    [np.nan, np.nan, np.nan]],
        ...                   columns=[\'A\', \'B\', \'C\'])

        Aggregate these functions over the rows.

        >>> df.agg([\'sum\', \'min\'])
                A     B     C
        sum  12.0  15.0  18.0
        min   1.0   2.0   3.0

        Different aggregations per column.

        >>> df.agg({\'A\' : [\'sum\', \'min\'], \'B\' : [\'min\', \'max\']})
                A    B
        sum  12.0  NaN
        min   1.0  2.0
        max   NaN  8.0

        Aggregate different functions over the columns and rename the index of the resulting
        DataFrame.

        >>> df.agg(x=(\'A\', \'max\'), y=(\'B\', \'min\'), z=(\'C\', \'mean\'))
             A    B    C
        x  7.0  NaN  NaN
        y  NaN  2.0  NaN
        z  NaN  NaN  6.0

        Aggregate over the columns.

        >>> df.agg("mean", axis="columns")
        0    2.0
        1    5.0
        2    8.0
        3    NaN
        dtype: float64
        '''
    def transform(self, func: AggFuncType, axis: Axis = ..., *args, **kwargs) -> DataFrame:
        '''
        Call ``func`` on self producing a DataFrame with the same axis shape as self.

        Parameters
        ----------
        func : function, str, list-like or dict-like
            Function to use for transforming the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply. If func
            is both list-like and dict-like, dict-like behavior takes precedence.

            Accepted combinations are:

            - function
            - string function name
            - list-like of functions and/or function names, e.g. ``[np.exp, \'sqrt\']``
            - dict-like of axis labels -> functions, function names or list-like of such.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
                If 0 or \'index\': apply function to each column.
                If 1 or \'columns\': apply function to each row.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        DataFrame
            A DataFrame that must have the same length as self.

        Raises
        ------
        ValueError : If the returned DataFrame has a different length than self.

        See Also
        --------
        DataFrame.agg : Only perform aggregating type operations.
        DataFrame.apply : Invoke function on a DataFrame.

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

        Even though the resulting DataFrame must have the same length as the
        input DataFrame, it is possible to provide several input functions:

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
    def apply(self, func: AggFuncType, axis: Axis = ..., raw: bool = ..., result_type: Literal['expand', 'reduce', 'broadcast'] | None, args: tuple = ..., by_row: Literal[False, 'compat'] = ..., engine: Literal['python', 'numba'] = ..., engine_kwargs: dict[str, bool] | None, **kwargs):
        '''
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is
        either the DataFrame\'s index (``axis=0``) or the DataFrame\'s columns
        (``axis=1``). By default (``result_type=None``), the final return type
        is inferred from the return type of the applied function. Otherwise,
        it depends on the `result_type` argument.

        Parameters
        ----------
        func : function
            Function to apply to each column or row.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            Axis along which the function is applied:

            * 0 or \'index\': apply function to each column.
            * 1 or \'columns\': apply function to each row.

        raw : bool, default False
            Determines if row or column is passed as a Series or ndarray object:

            * ``False`` : passes each row or column as a Series to the
              function.
            * ``True`` : the passed function will receive ndarray objects
              instead.
              If you are just applying a NumPy reduction function this will
              achieve much better performance.

        result_type : {\'expand\', \'reduce\', \'broadcast\', None}, default None
            These only act when ``axis=1`` (columns):

            * \'expand\' : list-like results will be turned into columns.
            * \'reduce\' : returns a Series if possible rather than expanding
              list-like results. This is the opposite of \'expand\'.
            * \'broadcast\' : results will be broadcast to the original shape
              of the DataFrame, the original index and columns will be
              retained.

            The default behaviour (None) depends on the return value of the
            applied function: list-like results will be returned as a Series
            of those. However if the apply function returns a Series these
            are expanded to columns.
        args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        by_row : False or "compat", default "compat"
            Only has an effect when ``func`` is a listlike or dictlike of funcs
            and the func isn\'t a string.
            If "compat", will if possible first translate the func into pandas
            methods (e.g. ``Series().apply(np.sum)`` will be translated to
            ``Series().sum()``). If that doesn\'t work, will try call to apply again with
            ``by_row=True`` and if that fails, will call apply again with
            ``by_row=False`` (backward compatible).
            If False, the funcs will be passed the whole Series at once.

            .. versionadded:: 2.1.0

        engine : {\'python\', \'numba\'}, default \'python\'
            Choose between the python (default) engine or the numba engine in apply.

            The numba engine will attempt to JIT compile the passed function,
            which may result in speedups for large DataFrames.
            It also supports the following engine_kwargs :

            - nopython (compile the function in nopython mode)
            - nogil (release the GIL inside the JIT compiled function)
            - parallel (try to apply the function in parallel over the DataFrame)

              Note: Due to limitations within numba/how pandas interfaces with numba,
              you should only use this if raw=True

            Note: The numba compiler only supports a subset of
            valid Python/numpy operations.

            Please read more about the `supported python features
            <https://numba.pydata.org/numba-doc/dev/reference/pysupported.html>`_
            and `supported numpy features
            <https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html>`_
            in numba to learn what you can or cannot use in the passed function.

            .. versionadded:: 2.2.0

        engine_kwargs : dict
            Pass keyword arguments to the engine.
            This is currently only used by the numba engine,
            see the documentation for the engine argument for more information.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        Series or DataFrame
            Result of applying ``func`` along the given axis of the
            DataFrame.

        See Also
        --------
        DataFrame.map: For elementwise operations.
        DataFrame.aggregate: Only perform aggregating type operations.
        DataFrame.transform: Only perform transforming type operations.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame([[4, 9]] * 3, columns=[\'A\', \'B\'])
        >>> df
           A  B
        0  4  9
        1  4  9
        2  4  9

        Using a numpy universal function (in this case the same as
        ``np.sqrt(df)``):

        >>> df.apply(np.sqrt)
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0

        Using a reducing function on either axis

        >>> df.apply(np.sum, axis=0)
        A    12
        B    27
        dtype: int64

        >>> df.apply(np.sum, axis=1)
        0    13
        1    13
        2    13
        dtype: int64

        Returning a list-like will result in a Series

        >>> df.apply(lambda x: [1, 2], axis=1)
        0    [1, 2]
        1    [1, 2]
        2    [1, 2]
        dtype: object

        Passing ``result_type=\'expand\'`` will expand list-like results
        to columns of a Dataframe

        >>> df.apply(lambda x: [1, 2], axis=1, result_type=\'expand\')
           0  1
        0  1  2
        1  1  2
        2  1  2

        Returning a Series inside the function is similar to passing
        ``result_type=\'expand\'``. The resulting column names
        will be the Series index.

        >>> df.apply(lambda x: pd.Series([1, 2], index=[\'foo\', \'bar\']), axis=1)
           foo  bar
        0    1    2
        1    1    2
        2    1    2

        Passing ``result_type=\'broadcast\'`` will ensure the same shape
        result, whether list-like or scalar is returned by the function,
        and broadcast it along the axis. The resulting column names will
        be the originals.

        >>> df.apply(lambda x: [1, 2], axis=1, result_type=\'broadcast\')
           A  B
        0  1  2
        1  1  2
        2  1  2
        '''
    def map(self, func: PythonFuncType, na_action: str | None, **kwargs) -> DataFrame:
        """
        Apply a function to a Dataframe elementwise.

        .. versionadded:: 2.1.0

           DataFrame.applymap was deprecated and renamed to DataFrame.map.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.replace: Replace values given in `to_replace` with `value`.
        Series.map : Apply a function elementwise on a Series.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> df.map(lambda x: len(str(x)))
           0  1
        0  3  4
        1  5  5

        Like Series.map, NA values can be ignored:

        >>> df_copy = df.copy()
        >>> df_copy.iloc[0, 0] = pd.NA
        >>> df_copy.map(lambda x: len(str(x)), na_action='ignore')
             0  1
        0  NaN  4
        1  5.0  5

        It is also possible to use `map` with functions that are not
        `lambda` functions:

        >>> df.map(round, ndigits=1)
             0    1
        0  1.0  2.1
        1  3.4  4.6

        Note that a vectorized version of `func` often exists, which will
        be much faster. You could square each number elementwise.

        >>> df.map(lambda x: x**2)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489

        But it's better to avoid map in that case.

        >>> df ** 2
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489
        """
    def applymap(self, func: PythonFuncType, na_action: NaAction | None, **kwargs) -> DataFrame:
        """
        Apply a function to a Dataframe elementwise.

        .. deprecated:: 2.1.0

           DataFrame.applymap has been deprecated. Use DataFrame.map instead.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.map : Apply a function along input axis of DataFrame.
        DataFrame.replace: Replace values given in `to_replace` with `value`.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> df.map(lambda x: len(str(x)))
           0  1
        0  3  4
        1  5  5
        """
    def _append(self, other, ignore_index: bool = ..., verify_integrity: bool = ..., sort: bool = ...) -> DataFrame: ...
    def join(self, other: DataFrame | Series | Iterable[DataFrame | Series], on: IndexLabel | None, how: MergeHow = ..., lsuffix: str = ..., rsuffix: str = ..., sort: bool = ..., validate: JoinValidate | None) -> DataFrame:
        '''
        Join columns of another DataFrame.

        Join columns with `other` DataFrame either on index or on a key
        column. Efficiently join multiple DataFrame objects by index at once by
        passing a list.

        Parameters
        ----------
        other : DataFrame, Series, or a list containing any combination of them
            Index should be similar to one of the columns in this one. If a
            Series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined DataFrame.
        on : str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index
            in `other`, otherwise joins index-on-index. If multiple
            values given, the `other` DataFrame must have a MultiIndex. Can
            pass an array as the join key if it is not already contained in
            the calling DataFrame. Like an Excel VLOOKUP operation.
        how : {\'left\', \'right\', \'outer\', \'inner\', \'cross\'}, default \'left\'
            How to handle the operation of the two objects.

            * left: use calling frame\'s index (or column if on is specified)
            * right: use `other`\'s index.
            * outer: form union of calling frame\'s index (or column if on is
              specified) with `other`\'s index, and sort it lexicographically.
            * inner: form intersection of calling frame\'s index (or column if
              on is specified) with `other`\'s index, preserving the order
              of the calling\'s one.
            * cross: creates the cartesian product from both frames, preserves the order
              of the left keys.
        lsuffix : str, default \'\'
            Suffix to use from left frame\'s overlapping columns.
        rsuffix : str, default \'\'
            Suffix to use from right frame\'s overlapping columns.
        sort : bool, default False
            Order result DataFrame lexicographically by the join key. If False,
            the order of the join key depends on the join type (how keyword).
        validate : str, optional
            If specified, checks if join is of specified type.

            * "one_to_one" or "1:1": check if join keys are unique in both left
              and right datasets.
            * "one_to_many" or "1:m": check if join keys are unique in left dataset.
            * "many_to_one" or "m:1": check if join keys are unique in right dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame
            A dataframe containing columns from both the caller and `other`.

        See Also
        --------
        DataFrame.merge : For column(s)-on-column(s) operations.

        Notes
        -----
        Parameters `on`, `lsuffix`, and `rsuffix` are not supported when
        passing a list of `DataFrame` objects.

        Examples
        --------
        >>> df = pd.DataFrame({\'key\': [\'K0\', \'K1\', \'K2\', \'K3\', \'K4\', \'K5\'],
        ...                    \'A\': [\'A0\', \'A1\', \'A2\', \'A3\', \'A4\', \'A5\']})

        >>> df
          key   A
        0  K0  A0
        1  K1  A1
        2  K2  A2
        3  K3  A3
        4  K4  A4
        5  K5  A5

        >>> other = pd.DataFrame({\'key\': [\'K0\', \'K1\', \'K2\'],
        ...                       \'B\': [\'B0\', \'B1\', \'B2\']})

        >>> other
          key   B
        0  K0  B0
        1  K1  B1
        2  K2  B2

        Join DataFrames using their indexes.

        >>> df.join(other, lsuffix=\'_caller\', rsuffix=\'_other\')
          key_caller   A key_other    B
        0         K0  A0        K0   B0
        1         K1  A1        K1   B1
        2         K2  A2        K2   B2
        3         K3  A3       NaN  NaN
        4         K4  A4       NaN  NaN
        5         K5  A5       NaN  NaN

        If we want to join using the key columns, we need to set key to be
        the index in both `df` and `other`. The joined DataFrame will have
        key as its index.

        >>> df.set_index(\'key\').join(other.set_index(\'key\'))
              A    B
        key
        K0   A0   B0
        K1   A1   B1
        K2   A2   B2
        K3   A3  NaN
        K4   A4  NaN
        K5   A5  NaN

        Another option to join using the key columns is to use the `on`
        parameter. DataFrame.join always uses `other`\'s index but we can use
        any column in `df`. This method preserves the original DataFrame\'s
        index in the result.

        >>> df.join(other.set_index(\'key\'), on=\'key\')
          key   A    B
        0  K0  A0   B0
        1  K1  A1   B1
        2  K2  A2   B2
        3  K3  A3  NaN
        4  K4  A4  NaN
        5  K5  A5  NaN

        Using non-unique key values shows how they are matched.

        >>> df = pd.DataFrame({\'key\': [\'K0\', \'K1\', \'K1\', \'K3\', \'K0\', \'K1\'],
        ...                    \'A\': [\'A0\', \'A1\', \'A2\', \'A3\', \'A4\', \'A5\']})

        >>> df
          key   A
        0  K0  A0
        1  K1  A1
        2  K1  A2
        3  K3  A3
        4  K0  A4
        5  K1  A5

        >>> df.join(other.set_index(\'key\'), on=\'key\', validate=\'m:1\')
          key   A    B
        0  K0  A0   B0
        1  K1  A1   B1
        2  K1  A2   B1
        3  K3  A3  NaN
        4  K0  A4   B0
        5  K1  A5   B1
        '''
    def merge(self, right: DataFrame | Series, how: MergeHow = ..., on: IndexLabel | AnyArrayLike | None, left_on: IndexLabel | AnyArrayLike | None, right_on: IndexLabel | AnyArrayLike | None, left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., copy: bool | None, indicator: str | bool = ..., validate: MergeValidate | None) -> DataFrame:
        '''
        Merge DataFrame or named Series objects with a database-style join.

        A named Series object is treated as a DataFrame with a single named column.

        The join is done on columns or indexes. If joining columns on
        columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
        on indexes or indexes on a column or columns, the index will be passed on.
        When performing a cross merge, no column specifications to merge on are
        allowed.

        .. warning::

            If both key columns contain rows where the key is a null value, those
            rows will be matched against each other. This is different from usual SQL
            join behaviour and can lead to unexpected results.

        Parameters
        ----------
        right : DataFrame or named Series
            Object to merge with.
        how : {\'left\', \'right\', \'outer\', \'inner\', \'cross\'}, default \'inner\'
            Type of merge to be performed.

            * left: use only keys from left frame, similar to a SQL left outer join;
              preserve key order.
            * right: use only keys from right frame, similar to a SQL right outer join;
              preserve key order.
            * outer: use union of keys from both frames, similar to a SQL full outer
              join; sort keys lexicographically.
            * inner: use intersection of keys from both frames, similar to a SQL inner
              join; preserve the order of the left keys.
            * cross: creates the cartesian product from both frames, preserves the order
              of the left keys.
        on : label or list
            Column or index level names to join on. These must be found in both
            DataFrames. If `on` is None and not merging on indexes then this defaults
            to the intersection of the columns in both DataFrames.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame. Can also
            be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame. Can also
            be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index : bool, default False
            Use the index from the left DataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other DataFrame (either the index
            or a number of columns) must match the number of levels.
        right_index : bool, default False
            Use the index from the right DataFrame as the join key. Same caveats as
            left_index.
        sort : bool, default False
            Sort the join keys lexicographically in the result DataFrame. If False,
            the order of the join keys depends on the join type (how keyword).
        suffixes : list-like, default is ("_x", "_y")
            A length-2 sequence where each element is optionally a string
            indicating the suffix to add to overlapping column names in
            `left` and `right` respectively. Pass a value of `None` instead
            of a string to indicate that the column name from `left` or
            `right` should be left as-is, with no suffix. At least one of the
            values must not be None.
        copy : bool, default True
            If False, avoid copy if possible.

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
        indicator : bool or str, default False
            If True, adds a column to the output DataFrame called "_merge" with
            information on the source of each row. The column can be given a different
            name by providing a string argument. The column will have a Categorical
            type with the value of "left_only" for observations whose merge key only
            appears in the left DataFrame, "right_only" for observations
            whose merge key only appears in the right DataFrame, and "both"
            if the observation\'s merge key is found in both DataFrames.

        validate : str, optional
            If specified, checks if merge is of specified type.

            * "one_to_one" or "1:1": check if merge keys are unique in both
              left and right datasets.
            * "one_to_many" or "1:m": check if merge keys are unique in left
              dataset.
            * "many_to_one" or "m:1": check if merge keys are unique in right
              dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.

        Returns
        -------
        DataFrame
            A DataFrame of the two merged objects.

        See Also
        --------
        merge_ordered : Merge with optional filling/interpolation.
        merge_asof : Merge on nearest keys.
        DataFrame.join : Similar method using indices.

        Examples
        --------
        >>> df1 = pd.DataFrame({\'lkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],
        ...                     \'value\': [1, 2, 3, 5]})
        >>> df2 = pd.DataFrame({\'rkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],
        ...                     \'value\': [5, 6, 7, 8]})
        >>> df1
            lkey value
        0   foo      1
        1   bar      2
        2   baz      3
        3   foo      5
        >>> df2
            rkey value
        0   foo      5
        1   bar      6
        2   baz      7
        3   foo      8

        Merge df1 and df2 on the lkey and rkey columns. The value columns have
        the default suffixes, _x and _y, appended.

        >>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\')
          lkey  value_x rkey  value_y
        0  foo        1  foo        5
        1  foo        1  foo        8
        2  bar        2  bar        6
        3  baz        3  baz        7
        4  foo        5  foo        5
        5  foo        5  foo        8

        Merge DataFrames df1 and df2 with specified left and right suffixes
        appended to any overlapping columns.

        >>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\',
        ...           suffixes=(\'_left\', \'_right\'))
          lkey  value_left rkey  value_right
        0  foo           1  foo            5
        1  foo           1  foo            8
        2  bar           2  bar            6
        3  baz           3  baz            7
        4  foo           5  foo            5
        5  foo           5  foo            8

        Merge DataFrames df1 and df2, but raise an exception if the DataFrames have
        any overlapping columns.

        >>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\', suffixes=(False, False))
        Traceback (most recent call last):
        ...
        ValueError: columns overlap but no suffix specified:
            Index([\'value\'], dtype=\'object\')

        >>> df1 = pd.DataFrame({\'a\': [\'foo\', \'bar\'], \'b\': [1, 2]})
        >>> df2 = pd.DataFrame({\'a\': [\'foo\', \'baz\'], \'c\': [3, 4]})
        >>> df1
              a  b
        0   foo  1
        1   bar  2
        >>> df2
              a  c
        0   foo  3
        1   baz  4

        >>> df1.merge(df2, how=\'inner\', on=\'a\')
              a  b  c
        0   foo  1  3

        >>> df1.merge(df2, how=\'left\', on=\'a\')
              a  b  c
        0   foo  1  3.0
        1   bar  2  NaN

        >>> df1 = pd.DataFrame({\'left\': [\'foo\', \'bar\']})
        >>> df2 = pd.DataFrame({\'right\': [7, 8]})
        >>> df1
            left
        0   foo
        1   bar
        >>> df2
            right
        0   7
        1   8

        >>> df1.merge(df2, how=\'cross\')
           left  right
        0   foo      7
        1   foo      8
        2   bar      7
        3   bar      8
        '''
    def round(self, decimals: int | dict[IndexLabel, int] | Series = ..., *args, **kwargs) -> DataFrame:
        """
        Round a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. If an int is
            given, round each column to the same number of places.
            Otherwise dict and Series round to variable numbers of places.
            Column names should be in the keys if `decimals` is a
            dict-like, or in the index if `decimals` is a Series. Any
            columns not included in `decimals` will be left as is. Elements
            of `decimals` which are not columns of the input will be
            ignored.
        *args
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.
        **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.

        Returns
        -------
        DataFrame
            A DataFrame with the affected columns rounded to the specified
            number of decimal places.

        See Also
        --------
        numpy.around : Round a numpy array to the given number of decimals.
        Series.round : Round a Series to the given number of decimals.

        Examples
        --------
        >>> df = pd.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
        ...                   columns=['dogs', 'cats'])
        >>> df
            dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places

        >>> df.round(1)
            dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as key and the number of decimal
        places as value

        >>> df.round({'dogs': 1, 'cats': 0})
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as index and the number of
        decimal places as value

        >>> decimals = pd.Series([0, 1], index=['cats', 'dogs'])
        >>> df.round(decimals)
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """
    def corr(self, method: CorrelationMethod = ..., min_periods: int = ..., numeric_only: bool = ...) -> DataFrame:
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float. Note that the returned matrix from corr
                will have 1 along the diagonals and will be symmetric
                regardless of the callable's behavior.
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result. Currently only available for Pearson
            and Spearman correlation.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        DataFrame
            Correlation matrix.

        See Also
        --------
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.
        Series.corr : Compute the correlation between two Series.

        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.

        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> df = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr(method=histogram_intersection)
              dogs  cats
        dogs   1.0   0.3
        cats   0.3   1.0

        >>> df = pd.DataFrame([(1, 1), (2, np.nan), (np.nan, 3), (4, 4)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr(min_periods=3)
              dogs  cats
        dogs   1.0   NaN
        cats   NaN   1.0
        """
    def cov(self, min_periods: int | None, ddof: int | None = ..., numeric_only: bool = ...) -> DataFrame:
        """
        Compute pairwise covariance of columns, excluding NA/null values.

        Compute the pairwise covariance among the series of a DataFrame.
        The returned data frame is the `covariance matrix
        <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
        of the DataFrame.

        Both NA and null values are automatically excluded from the
        calculation. (See the note below about bias from missing values.)
        A threshold can be set for the minimum number of
        observations for each value created. Comparisons with observations
        below this threshold will be returned as ``NaN``.

        This method is generally used for the analysis of time series data to
        understand the relationship between different measures
        across time.

        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            This argument is applicable only when no ``nan`` is in the dataframe.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        DataFrame
            The covariance matrix of the series of the DataFrame.

        See Also
        --------
        Series.cov : Compute covariance with another Series.
        core.window.ewm.ExponentialMovingWindow.cov : Exponential weighted sample
            covariance.
        core.window.expanding.Expanding.cov : Expanding sample covariance.
        core.window.rolling.Rolling.cov : Rolling sample covariance.

        Notes
        -----
        Returns the covariance matrix of the DataFrame's time series.
        The covariance is normalized by N-ddof.

        For DataFrames that have Series that are missing data (assuming that
        data is `missing at random
        <https://en.wikipedia.org/wiki/Missing_data#Missing_at_random>`__)
        the returned covariance matrix will be an unbiased estimate
        of the variance and covariance between the member Series.

        However, for many applications this estimate may not be acceptable
        because the estimate covariance matrix is not guaranteed to be positive
        semi-definite. This could lead to estimate correlations having
        absolute values which are greater than one, and/or a non-invertible
        covariance matrix. See `Estimation of covariance matrices
        <https://en.wikipedia.org/w/index.php?title=Estimation_of_covariance_
        matrices>`__ for more details.

        Examples
        --------
        >>> df = pd.DataFrame([(1, 2), (0, 3), (2, 0), (1, 1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.cov()
                  dogs      cats
        dogs  0.666667 -1.000000
        cats -1.000000  1.666667

        >>> np.random.seed(42)
        >>> df = pd.DataFrame(np.random.randn(1000, 5),
        ...                   columns=['a', 'b', 'c', 'd', 'e'])
        >>> df.cov()
                  a         b         c         d         e
        a  0.998438 -0.020161  0.059277 -0.008943  0.014144
        b -0.020161  1.059352 -0.008543 -0.024738  0.009826
        c  0.059277 -0.008543  1.010670 -0.001486 -0.000271
        d -0.008943 -0.024738 -0.001486  0.921297 -0.013692
        e  0.014144  0.009826 -0.000271 -0.013692  0.977795

        **Minimum number of periods**

        This method also supports an optional ``min_periods`` keyword
        that specifies the required minimum number of non-NA observations for
        each column pair in order to have a valid result:

        >>> np.random.seed(42)
        >>> df = pd.DataFrame(np.random.randn(20, 3),
        ...                   columns=['a', 'b', 'c'])
        >>> df.loc[df.index[:5], 'a'] = np.nan
        >>> df.loc[df.index[5:10], 'b'] = np.nan
        >>> df.cov(min_periods=12)
                  a         b         c
        a  0.316741       NaN -0.150812
        b       NaN  1.248003  0.191417
        c -0.150812  0.191417  0.895202
        """
    def corrwith(self, other: DataFrame | Series, axis: Axis = ..., drop: bool = ..., method: CorrelationMethod = ..., numeric_only: bool = ...) -> Series:
        '''
        Compute pairwise correlation.

        Pairwise correlation is computed between rows or columns of
        DataFrame with rows or columns of Series or DataFrame. DataFrames
        are first aligned along both axes before computing the
        correlations.

        Parameters
        ----------
        other : DataFrame, Series
            Object with which to compute correlations.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            The axis to use. 0 or \'index\' to compute row-wise, 1 or \'columns\' for
            column-wise.
        drop : bool, default False
            Drop missing indices from result.
        method : {\'pearson\', \'kendall\', \'spearman\'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        Series
            Pairwise correlations.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation of columns.

        Examples
        --------
        >>> index = ["a", "b", "c", "d", "e"]
        >>> columns = ["one", "two", "three", "four"]
        >>> df1 = pd.DataFrame(np.arange(20).reshape(5, 4), index=index, columns=columns)
        >>> df2 = pd.DataFrame(np.arange(16).reshape(4, 4), index=index[:4], columns=columns)
        >>> df1.corrwith(df2)
        one      1.0
        two      1.0
        three    1.0
        four     1.0
        dtype: float64

        >>> df2.corrwith(df1, axis=1)
        a    1.0
        b    1.0
        c    1.0
        d    1.0
        e    NaN
        dtype: float64
        '''
    def count(self, axis: Axis = ..., numeric_only: bool = ...):
        '''
        Count non-NA cells for each column or row.

        The values `None`, `NaN`, `NaT`, ``pandas.NA`` are considered NA.

        Parameters
        ----------
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            If 0 or \'index\' counts are generated for each column.
            If 1 or \'columns\' counts are generated for each row.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.value_counts: Count unique combinations of columns.
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = pd.DataFrame({"Person":
        ...                    ["John", "Myla", "Lewis", "John", "Myla"],
        ...                    "Age": [24., np.nan, 21., 33, 26],
        ...                    "Single": [False, True, True, True, False]})
        >>> df
           Person   Age  Single
        0    John  24.0   False
        1    Myla   NaN    True
        2   Lewis  21.0    True
        3    John  33.0    True
        4    Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64

        Counts for each **row**:

        >>> df.count(axis=\'columns\')
        0    3
        1    2
        2    3
        3    3
        4    3
        dtype: int64
        '''
    def _reduce(self, op, name: str, *, axis: Axis = ..., skipna: bool = ..., numeric_only: bool = ..., filter_type, **kwds): ...
    def _reduce_axis1(self, name: str, func, skipna: bool) -> Series:
        """
        Special case for _reduce to try to avoid a potentially-expensive transpose.

        Apply the reduction block-wise along axis=1 and then reduce the resulting
        1D arrays.
        """
    def any(self, *, axis: Axis | None = ..., bool_only: bool = ..., skipna: bool = ..., **kwargs) -> Series | bool:
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
        Series or DataFrame
            If level is specified, then, DataFrame is returned; otherwise, Series
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
    def all(self, axis: Axis | None = ..., bool_only: bool = ..., skipna: bool = ..., **kwargs) -> Series | bool:
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
        Series or DataFrame
            If level is specified, then, DataFrame is returned; otherwise, Series
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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
    def sum(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., min_count: int = ..., **kwargs):
        '''
        Return the sum of the values over the requested axis.

        This is equivalent to the method ``numpy.sum``.

        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        Series or scalar

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
    def prod(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., min_count: int = ..., **kwargs):
        '''
        Return the product of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        Series or scalar

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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
    def sem(self, axis: Axis | None = ..., skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        Series or DataFrame (if level specified) 

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
    def var(self, axis: Axis | None = ..., skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        Series or DataFrame (if level specified) 

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
    def std(self, axis: Axis | None = ..., skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        Series or DataFrame (if level specified) 

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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
        axis : {index (0), columns (1)}
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
        Series or scalar

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
    def product(self, axis: Axis | None = ..., skipna: bool = ..., numeric_only: bool = ..., min_count: int = ..., **kwargs):
        '''
        Return the product of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        Series or scalar

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
        Series or DataFrame
            Return cumulative minimum of Series or DataFrame.

        See Also
        --------
        core.window.expanding.Expanding.min : Similar functionality
            but ignores ``NaN`` values.
        DataFrame.min : Return the minimum over
            DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.

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
        Series or DataFrame
            Return cumulative maximum of Series or DataFrame.

        See Also
        --------
        core.window.expanding.Expanding.max : Similar functionality
            but ignores ``NaN`` values.
        DataFrame.max : Return the maximum over
            DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.

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
        Series or DataFrame
            Return cumulative sum of Series or DataFrame.

        See Also
        --------
        core.window.expanding.Expanding.sum : Similar functionality
            but ignores ``NaN`` values.
        DataFrame.sum : Return the sum over
            DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.

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
        Series or DataFrame
            Return cumulative product of Series or DataFrame.

        See Also
        --------
        core.window.expanding.Expanding.prod : Similar functionality
            but ignores ``NaN`` values.
        DataFrame.prod : Return the product over
            DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.

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
    def nunique(self, axis: Axis = ..., dropna: bool = ...) -> Series:
        """
        Count number of distinct elements in specified axis.

        Return Series with number of distinct elements. Can ignore NaN
        values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
            column-wise.
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series

        See Also
        --------
        Series.nunique: Method nunique for Series.
        DataFrame.count: Count non-NA cells for each column or row.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [4, 5, 6], 'B': [4, 1, 1]})
        >>> df.nunique()
        A    3
        B    2
        dtype: int64

        >>> df.nunique(axis=1)
        0    1
        1    2
        2    2
        dtype: int64
        """
    def idxmin(self, axis: Axis = ..., skipna: bool = ..., numeric_only: bool = ...) -> Series:
        '''
            Return index of first occurrence of minimum over requested axis.

            NA/null values are excluded.

            Parameters
            ----------
            axis : {0 or \'index\', 1 or \'columns\'}, default 0
                The axis to use. 0 or \'index\' for row-wise, 1 or \'columns\' for column-wise.
            skipna : bool, default True
                Exclude NA/null values. If an entire row/column is NA, the result
                will be NA.
            numeric_only : bool, default False
                Include only `float`, `int` or `boolean` data.

                .. versionadded:: 1.5.0

            Returns
            -------
            Series
                Indexes of minima along the specified axis.

            Raises
            ------
            ValueError
                * If the row/column is empty

            See Also
            --------
            Series.idxmin : Return index of the minimum element.

            Notes
            -----
            This method is the DataFrame version of ``ndarray.argmin``.

            Examples
            --------
            Consider a dataset containing food consumption in Argentina.

            >>> df = pd.DataFrame({\'consumption\': [10.51, 103.11, 55.48],
            ...                     \'co2_emissions\': [37.2, 19.66, 1712]},
            ...                   index=[\'Pork\', \'Wheat Products\', \'Beef\'])

            >>> df
                            consumption  co2_emissions
            Pork                  10.51         37.20
            Wheat Products       103.11         19.66
            Beef                  55.48       1712.00

            By default, it returns the index for the minimum value in each column.

            >>> df.idxmin()
            consumption                Pork
            co2_emissions    Wheat Products
            dtype: object

            To return the index for the minimum value in each row, use ``axis="columns"``.

            >>> df.idxmin(axis="columns")
            Pork                consumption
            Wheat Products    co2_emissions
            Beef                consumption
            dtype: object
        '''
    def idxmax(self, axis: Axis = ..., skipna: bool = ..., numeric_only: bool = ...) -> Series:
        '''
            Return index of first occurrence of maximum over requested axis.

            NA/null values are excluded.

            Parameters
            ----------
            axis : {0 or \'index\', 1 or \'columns\'}, default 0
                The axis to use. 0 or \'index\' for row-wise, 1 or \'columns\' for column-wise.
            skipna : bool, default True
                Exclude NA/null values. If an entire row/column is NA, the result
                will be NA.
            numeric_only : bool, default False
                Include only `float`, `int` or `boolean` data.

                .. versionadded:: 1.5.0

            Returns
            -------
            Series
                Indexes of maxima along the specified axis.

            Raises
            ------
            ValueError
                * If the row/column is empty

            See Also
            --------
            Series.idxmax : Return index of the maximum element.

            Notes
            -----
            This method is the DataFrame version of ``ndarray.argmax``.

            Examples
            --------
            Consider a dataset containing food consumption in Argentina.

            >>> df = pd.DataFrame({\'consumption\': [10.51, 103.11, 55.48],
            ...                     \'co2_emissions\': [37.2, 19.66, 1712]},
            ...                   index=[\'Pork\', \'Wheat Products\', \'Beef\'])

            >>> df
                            consumption  co2_emissions
            Pork                  10.51         37.20
            Wheat Products       103.11         19.66
            Beef                  55.48       1712.00

            By default, it returns the index for the maximum value in each column.

            >>> df.idxmax()
            consumption     Wheat Products
            co2_emissions             Beef
            dtype: object

            To return the index for the maximum value in each row, use ``axis="columns"``.

            >>> df.idxmax(axis="columns")
            Pork              co2_emissions
            Wheat Products     consumption
            Beef              co2_emissions
            dtype: object
        '''
    def _get_agg_axis(self, axis_num: int) -> Index:
        """
        Let's be explicit about this.
        """
    def mode(self, axis: Axis = ..., numeric_only: bool = ..., dropna: bool = ...) -> DataFrame:
        """
        Get the mode(s) of each element along the selected axis.

        The mode of a set of values is the value that appears most often.
        It can be multiple values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to iterate over while searching for the mode:

            * 0 or 'index' : get mode of each column
            * 1 or 'columns' : get mode of each row.

        numeric_only : bool, default False
            If True, only apply to numeric columns.
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        DataFrame
            The modes of each column or row.

        See Also
        --------
        Series.mode : Return the highest frequency value in a Series.
        Series.value_counts : Return the counts of values in a Series.

        Examples
        --------
        >>> df = pd.DataFrame([('bird', 2, 2),
        ...                    ('mammal', 4, np.nan),
        ...                    ('arthropod', 8, 0),
        ...                    ('bird', 2, np.nan)],
        ...                   index=('falcon', 'horse', 'spider', 'ostrich'),
        ...                   columns=('species', 'legs', 'wings'))
        >>> df
                   species  legs  wings
        falcon        bird     2    2.0
        horse       mammal     4    NaN
        spider   arthropod     8    0.0
        ostrich       bird     2    NaN

        By default, missing values are not considered, and the mode of wings
        are both 0 and 2. Because the resulting DataFrame has two rows,
        the second row of ``species`` and ``legs`` contains ``NaN``.

        >>> df.mode()
          species  legs  wings
        0    bird   2.0    0.0
        1     NaN   NaN    2.0

        Setting ``dropna=False`` ``NaN`` values are considered and they can be
        the mode (like for wings).

        >>> df.mode(dropna=False)
          species  legs  wings
        0    bird     2    NaN

        Setting ``numeric_only=True``, only the mode of numeric columns is
        computed, and columns of other types are ignored.

        >>> df.mode(numeric_only=True)
           legs  wings
        0   2.0    0.0
        1   NaN    2.0

        To compute the mode over columns and not rows, use the axis parameter:

        >>> df.mode(axis='columns', numeric_only=True)
                   0    1
        falcon   2.0  NaN
        horse    4.0  NaN
        spider   0.0  8.0
        ostrich  2.0  NaN
        """
    def quantile(self, q: float | AnyArrayLike | Sequence[float] = ..., axis: Axis = ..., numeric_only: bool = ..., interpolation: QuantileInterpolation = ..., method: Literal['single', 'table'] = ...) -> Series | DataFrame:
        '''
        Return values at the given quantile over requested axis.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value between 0 <= q <= 1, the quantile(s) to compute.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            Equals 0 or \'index\' for row-wise, 1 or \'columns\' for column-wise.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        interpolation : {\'linear\', \'lower\', \'higher\', \'midpoint\', \'nearest\'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

            * linear: `i + (j - i) * fraction`, where `fraction` is the
              fractional part of the index surrounded by `i` and `j`.
            * lower: `i`.
            * higher: `j`.
            * nearest: `i` or `j` whichever is nearest.
            * midpoint: (`i` + `j`) / 2.
        method : {\'single\', \'table\'}, default \'single\'
            Whether to compute quantiles per-column (\'single\') or over all columns
            (\'table\'). When \'table\', the only allowed interpolation methods are
            \'nearest\', \'lower\', and \'higher\'.

        Returns
        -------
        Series or DataFrame

            If ``q`` is an array, a DataFrame will be returned where the
              index is ``q``, the columns are the columns of self, and the
              values are the quantiles.
            If ``q`` is a float, a Series will be returned where the
              index is the columns of self and the values are the quantiles.

        See Also
        --------
        core.window.rolling.Rolling.quantile: Rolling quantile.
        numpy.percentile: Numpy function to compute the percentile.

        Examples
        --------
        >>> df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
        ...                   columns=[\'a\', \'b\'])
        >>> df.quantile(.1)
        a    1.3
        b    3.7
        Name: 0.1, dtype: float64
        >>> df.quantile([.1, .5])
               a     b
        0.1  1.3   3.7
        0.5  2.5  55.0

        Specifying `method=\'table\'` will compute the quantile over all columns.

        >>> df.quantile(.1, method="table", interpolation="nearest")
        a    1
        b    1
        Name: 0.1, dtype: int64
        >>> df.quantile([.1, .5], method="table", interpolation="nearest")
             a    b
        0.1  1    1
        0.5  3  100

        Specifying `numeric_only=False` will also compute the quantile of
        datetime and timedelta data.

        >>> df = pd.DataFrame({\'A\': [1, 2],
        ...                    \'B\': [pd.Timestamp(\'2010\'),
        ...                          pd.Timestamp(\'2011\')],
        ...                    \'C\': [pd.Timedelta(\'1 days\'),
        ...                          pd.Timedelta(\'2 days\')]})
        >>> df.quantile(0.5, numeric_only=False)
        A                    1.5
        B    2010-07-02 12:00:00
        C        1 days 12:00:00
        Name: 0.5, dtype: object
        '''
    def to_timestamp(self, freq: Frequency | None, how: ToTimestampHow = ..., axis: Axis = ..., copy: bool | None) -> DataFrame:
        """
        Cast to DatetimeIndex of timestamps, at *beginning* of period.

        Parameters
        ----------
        freq : str, default frequency of PeriodIndex
            Desired frequency.
        how : {'s', 'e', 'start', 'end'}
            Convention for converting period to timestamp; start of period
            vs. end.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to convert (the index by default).
        copy : bool, default True
            If False then underlying input data is not copied.

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
        DataFrame
            The DataFrame has a DatetimeIndex.

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2023', '2024'], freq='Y')
        >>> d = {'col1': [1, 2], 'col2': [3, 4]}
        >>> df1 = pd.DataFrame(data=d, index=idx)
        >>> df1
              col1   col2
        2023     1      3
        2024     2      4

        The resulting timestamps will be at the beginning of the year in this case

        >>> df1 = df1.to_timestamp()
        >>> df1
                    col1   col2
        2023-01-01     1      3
        2024-01-01     2      4
        >>> df1.index
        DatetimeIndex(['2023-01-01', '2024-01-01'], dtype='datetime64[ns]', freq=None)

        Using `freq` which is the offset that the Timestamps will have

        >>> df2 = pd.DataFrame(data=d, index=idx)
        >>> df2 = df2.to_timestamp(freq='M')
        >>> df2
                    col1   col2
        2023-01-31     1      3
        2024-01-31     2      4
        >>> df2.index
        DatetimeIndex(['2023-01-31', '2024-01-31'], dtype='datetime64[ns]', freq=None)
        """
    def to_period(self, freq: Frequency | None, axis: Axis = ..., copy: bool | None) -> DataFrame:
        '''
        Convert DataFrame from DatetimeIndex to PeriodIndex.

        Convert DataFrame from DatetimeIndex to PeriodIndex with desired
        frequency (inferred from index if not passed).

        Parameters
        ----------
        freq : str, default
            Frequency of the PeriodIndex.
        axis : {0 or \'index\', 1 or \'columns\'}, default 0
            The axis to convert (the index by default).
        copy : bool, default True
            If False then underlying input data is not copied.

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
        DataFrame
            The DataFrame has a PeriodIndex.

        Examples
        --------
        >>> idx = pd.to_datetime(
        ...     [
        ...         "2001-03-31 00:00:00",
        ...         "2002-05-31 00:00:00",
        ...         "2003-08-31 00:00:00",
        ...     ]
        ... )

        >>> idx
        DatetimeIndex([\'2001-03-31\', \'2002-05-31\', \'2003-08-31\'],
        dtype=\'datetime64[ns]\', freq=None)

        >>> idx.to_period("M")
        PeriodIndex([\'2001-03\', \'2002-05\', \'2003-08\'], dtype=\'period[M]\')

        For the yearly frequency

        >>> idx.to_period("Y")
        PeriodIndex([\'2001\', \'2002\', \'2003\'], dtype=\'period[Y-DEC]\')
        '''
    def isin(self, values: Series | DataFrame | Sequence | Mapping) -> DataFrame:
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all the
            labels match. If `values` is a Series, that's the index. If
            `values` is a dict, the keys must be the column names,
            which must match. If `values` is a DataFrame,
            then both the index and column labels must match.

        Returns
        -------
        DataFrame
            DataFrame of booleans showing whether each element in the DataFrame
            is contained in values.

        See Also
        --------
        DataFrame.eq: Equality test for DataFrame.
        Series.isin: Equivalent method on Series.
        Series.str.contains: Test if pattern or regex is contained within a
            string of a Series or Index.

        Examples
        --------
        >>> df = pd.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
        ...                   index=['falcon', 'dog'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        When ``values`` is a list check whether every value in the DataFrame
        is present in the list (which animals have 0 or 2 legs or wings)

        >>> df.isin([0, 2])
                num_legs  num_wings
        falcon      True       True
        dog        False       True

        To check if ``values`` is *not* in the DataFrame, use the ``~`` operator:

        >>> ~df.isin([0, 2])
                num_legs  num_wings
        falcon     False      False
        dog         True      False

        When ``values`` is a dict, we can pass values to check for each
        column separately:

        >>> df.isin({'num_wings': [0, 3]})
                num_legs  num_wings
        falcon     False      False
        dog        False       True

        When ``values`` is a Series or DataFrame the index and column must
        match. Note that 'falcon' does not match based on the number of legs
        in other.

        >>> other = pd.DataFrame({'num_legs': [8, 3], 'num_wings': [0, 2]},
        ...                      index=['spider', 'falcon'])
        >>> df.isin(other)
                num_legs  num_wings
        falcon     False       True
        dog        False      False
        """
    def hist(self, data: DataFrame, column: IndexLabel | None, by, grid: bool = ..., xlabelsize: int | None, xrot: float | None, ylabelsize: int | None, yrot: float | None, ax, sharex: bool = ..., sharey: bool = ..., figsize: tuple[int, int] | None, layout: tuple[int, int] | None, bins: int | Sequence[int] = ..., backend: str | None, legend: bool = ..., **kwargs):
        """
        Make a histogram of the DataFrame's columns.

        A `histogram`_ is a representation of the distribution of data.
        This function calls :meth:`matplotlib.pyplot.hist`, on each series in
        the DataFrame, resulting in one histogram per column.

        .. _histogram: https://en.wikipedia.org/wiki/Histogram

        Parameters
        ----------
        data : DataFrame
            The pandas object holding the data.
        column : str or sequence, optional
            If passed, will be used to limit data to a subset of columns.
        by : object, optional
            If passed, then used to form histograms for separate groups.
        grid : bool, default True
            Whether to show axis grid lines.
        xlabelsize : int, default None
            If specified changes the x-axis label size.
        xrot : float, default None
            Rotation of x axis labels. For example, a value of 90 displays the
            x labels rotated 90 degrees clockwise.
        ylabelsize : int, default None
            If specified changes the y-axis label size.
        yrot : float, default None
            Rotation of y axis labels. For example, a value of 90 displays the
            y labels rotated 90 degrees clockwise.
        ax : Matplotlib axes object, default None
            The axes to plot the histogram on.
        sharex : bool, default True if ax is None else False
            In case subplots=True, share x axis and set some x axis labels to
            invisible; defaults to True if ax is None otherwise False if an ax
            is passed in.
            Note that passing in both an ax and sharex=True will alter all x axis
            labels for all subplots in a figure.
        sharey : bool, default False
            In case subplots=True, share y axis and set some y axis labels to
            invisible.
        figsize : tuple, optional
            The size in inches of the figure to create. Uses the value in
            `matplotlib.rcParams` by default.
        layout : tuple, optional
            Tuple of (rows, columns) for the layout of the histograms.
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
            All other plotting keyword arguments to be passed to
            :meth:`matplotlib.pyplot.hist`.

        Returns
        -------
        matplotlib.AxesSubplot or numpy.ndarray of them

        See Also
        --------
        matplotlib.pyplot.hist : Plot a histogram using matplotlib.

        Examples
        --------
        This example draws a histogram based on the length and width of
        some animals, displayed in three bins

        .. plot::
            :context: close-figs

            >>> data = {'length': [1.5, 0.5, 1.2, 0.9, 3],
            ...         'width': [0.7, 0.2, 0.15, 0.2, 1.1]}
            >>> index = ['pig', 'rabbit', 'duck', 'chicken', 'horse']
            >>> df = pd.DataFrame(data, index=index)
            >>> hist = df.hist(bins=3)
        """
    def boxplot(self: DataFrame, column, by, ax, fontsize: int | None, rot: int = ..., grid: bool = ..., figsize: tuple[float, float] | None, layout, return_type, backend, **kwargs):
        """
        Make a box plot from DataFrame columns.

        Make a box-and-whisker plot from DataFrame columns, optionally grouped
        by some other columns. A box plot is a method for graphically depicting
        groups of numerical data through their quartiles.
        The box extends from the Q1 to Q3 quartile values of the data,
        with a line at the median (Q2). The whiskers extend from the edges
        of box to show the range of the data. By default, they extend no more than
        `1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
        data point within that interval. Outliers are plotted as separate dots.

        For further details see
        Wikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.

        Parameters
        ----------
        column : str or list of str, optional
            Column name or list of names, or vector.
            Can be any valid input to :meth:`pandas.DataFrame.groupby`.
        by : str or array-like, optional
            Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.
            One box-plot will be done per value of columns in `by`.
        ax : object of class matplotlib.axes.Axes, optional
            The matplotlib axes to be used by boxplot.
        fontsize : float or str
            Tick label font size in points or as a string (e.g., `large`).
        rot : float, default 0
            The rotation angle of labels (in degrees)
            with respect to the screen coordinate system.
        grid : bool, default True
            Setting this to True will show the grid.
        figsize : A tuple (width, height) in inches
            The size of the figure to create in matplotlib.
        layout : tuple (rows, columns), optional
            For example, (3, 5) will display the subplots
            using 3 rows and 5 columns, starting from the top-left.
        return_type : {'axes', 'dict', 'both'} or None, default 'axes'
            The kind of object to return. The default is ``axes``.

            * 'axes' returns the matplotlib axes the boxplot is drawn on.
            * 'dict' returns a dictionary whose values are the matplotlib
              Lines of the boxplot.
            * 'both' returns a namedtuple with the axes and dict.
            * when grouping with ``by``, a Series mapping columns to
              ``return_type`` is returned.

              If ``return_type`` is `None`, a NumPy array
              of axes with the same shape as ``layout`` is returned.
        backend : str, default None
            Backend to use instead of the backend specified in the option
            ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
            specify the ``plotting.backend`` for the whole session, set
            ``pd.options.plotting.backend``.

        **kwargs
            All other plotting keyword arguments to be passed to
            :func:`matplotlib.pyplot.boxplot`.

        Returns
        -------
        result
            See Notes.

        See Also
        --------
        pandas.Series.plot.hist: Make a histogram.
        matplotlib.pyplot.boxplot : Matplotlib equivalent plot.

        Notes
        -----
        The return type depends on the `return_type` parameter:

        * 'axes' : object of class matplotlib.axes.Axes
        * 'dict' : dict of matplotlib.lines.Line2D objects
        * 'both' : a namedtuple with structure (ax, lines)

        For data grouped with ``by``, return a Series of the above or a numpy
        array:

        * :class:`~pandas.Series`
        * :class:`~numpy.array` (for ``return_type = None``)

        Use ``return_type='dict'`` when you want to tweak the appearance
        of the lines after plotting. In this case a dict containing the Lines
        making up the boxes, caps, fliers, medians, and whiskers is returned.

        Examples
        --------

        Boxplots can be created for every column in the dataframe
        by ``df.boxplot()`` or indicating the columns to be used:

        .. plot::
            :context: close-figs

            >>> np.random.seed(1234)
            >>> df = pd.DataFrame(np.random.randn(10, 4),
            ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])
            >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP

        Boxplots of variables distributions grouped by the values of a third
        variable can be created using the option ``by``. For instance:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randn(10, 2),
            ...                   columns=['Col1', 'Col2'])
            >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
            ...                      'B', 'B', 'B', 'B', 'B'])
            >>> boxplot = df.boxplot(by='X')

        A list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot
        in order to group the data by combination of the variables in the x-axis:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randn(10, 3),
            ...                   columns=['Col1', 'Col2', 'Col3'])
            >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
            ...                      'B', 'B', 'B', 'B', 'B'])
            >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',
            ...                      'B', 'A', 'B', 'A', 'B'])
            >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])

        The layout of boxplot can be adjusted giving a tuple to ``layout``:

        .. plot::
            :context: close-figs

            >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
            ...                      layout=(2, 1))

        Additional formatting can be done to the boxplot, like suppressing the grid
        (``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)
        or changing the fontsize (i.e. ``fontsize=15``):

        .. plot::
            :context: close-figs

            >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP

        The parameter ``return_type`` can be used to select the type of element
        returned by `boxplot`.  When ``return_type='axes'`` is selected,
        the matplotlib axes on which the boxplot is drawn are returned:

            >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')
            >>> type(boxplot)
            <class 'matplotlib.axes._axes.Axes'>

        When grouping with ``by``, a Series mapping columns to ``return_type``
        is returned:

            >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
            ...                      return_type='axes')
            >>> type(boxplot)
            <class 'pandas.core.series.Series'>

        If ``return_type`` is `None`, a NumPy array of axes with the same shape
        as ``layout`` is returned:

            >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
            ...                      return_type=None)
            >>> type(boxplot)
            <class 'numpy.ndarray'>
        """
    def _to_dict_of_blocks(self):
        """
        Return a dict of dtype -> Constructor Types that
        each is a homogeneous dtype.

        Internal ONLY - only works for BlockManager
        """
    @property
    def _constructor(self): ...
    @property
    def axes(self): ...
    @property
    def shape(self): ...
    @property
    def _is_homogeneous_type(self): ...
    @property
    def _can_fast_transpose(self): ...
    @property
    def _values(self): ...
    @property
    def style(self): ...
    @property
    def T(self): ...
    @property
    def _series(self): ...
    @property
    def values(self): ...
def _from_nested_dict(data) -> collections.defaultdict: ...
def _reindex_for_setitem(value: DataFrame | Series, index: Index) -> tuple[ArrayLike, BlockValuesRefs | None]: ...

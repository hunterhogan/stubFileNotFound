import lib as lib
import pandas.core.common as com
from collections.abc import Hashable
from pandas._libs.lib import is_list_like as is_list_like, is_scalar as is_scalar
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import maybe_downcast_to_dtype as maybe_downcast_to_dtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.inference import is_nested_list_like as is_nested_list_like
from pandas.core.groupby.grouper import Grouper as Grouper
from pandas.core.indexes.api import get_objs_combined_axis as get_objs_combined_axis
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.reshape.concat import concat as concat
from pandas.core.reshape.util import cartesian_product as cartesian_product
from pandas.core.series import Series as Series
from pandas.util._decorators import Appender as Appender, Substitution as Substitution
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Literal

TYPE_CHECKING: bool
_shared_docs: dict
def pivot_table(data: DataFrame, values, index, columns, aggfunc: AggFuncType = ..., fill_value, margins: bool = ..., dropna: bool = ..., margins_name: Hashable = ..., observed: bool | lib.NoDefault = ..., sort: bool = ...) -> DataFrame:
    '''
    Create a spreadsheet-style pivot table as a DataFrame.

    The levels in the pivot table will be stored in MultiIndex objects
    (hierarchical indexes) on the index and columns of the result DataFrame.

    Parameters
    ----------
    data : DataFrame
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
def __internal_pivot_table(data: DataFrame, values, index, columns, aggfunc: AggFuncTypeBase | AggFuncTypeDict, fill_value, margins: bool, dropna: bool, margins_name: Hashable, observed: bool | lib.NoDefault, sort: bool) -> DataFrame:
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
def _add_margins(table: DataFrame | Series, data: DataFrame, values, rows, cols, aggfunc, observed: bool, margins_name: Hashable = ..., fill_value): ...
def _compute_grand_margin(data: DataFrame, values, aggfunc, margins_name: Hashable = ...): ...
def _generate_marginal_results(table, data: DataFrame, values, rows, cols, aggfunc, observed: bool, margins_name: Hashable = ...): ...
def _generate_marginal_results_without_values(table: DataFrame, data: DataFrame, rows, cols, aggfunc, observed: bool, margins_name: Hashable = ...): ...
def _convert_by(by): ...
def pivot(data: DataFrame, *, columns: IndexLabel, index: IndexLabel | lib.NoDefault = ..., values: IndexLabel | lib.NoDefault = ...) -> DataFrame:
    '''
    Return reshaped DataFrame organized by given index / column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame. This function does not support data
    aggregation, multiple values will result in a MultiIndex in the
    columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

    Parameters
    ----------
    data : DataFrame
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
def crosstab(index, columns, values, rownames, colnames, aggfunc, margins: bool = ..., margins_name: Hashable = ..., dropna: bool = ..., normalize: bool | Literal[0, 1, 'all', 'index', 'columns'] = ...) -> DataFrame:
    '''
    Compute a simple cross tabulation of two (or more) factors.

    By default, computes a frequency table of the factors unless an
    array of values and an aggregation function are passed.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows.
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns.
    values : array-like, optional
        Array of values to aggregate according to the factors.
        Requires `aggfunc` be specified.
    rownames : sequence, default None
        If passed, must match number of row arrays passed.
    colnames : sequence, default None
        If passed, must match number of column arrays passed.
    aggfunc : function, optional
        If specified, requires `values` be specified as well.
    margins : bool, default False
        Add row/column margins (subtotals).
    margins_name : str, default \'All\'
        Name of the row/column that will contain the totals
        when margins is True.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    normalize : bool, {\'all\', \'index\', \'columns\'}, or {0,1}, default False
        Normalize by dividing all values by the sum of values.

        - If passed \'all\' or `True`, will normalize over all values.
        - If passed \'index\' will normalize over each row.
        - If passed \'columns\' will normalize over each column.
        - If margins is `True`, will also normalize margin values.

    Returns
    -------
    DataFrame
        Cross tabulation of the data.

    See Also
    --------
    DataFrame.pivot : Reshape data based on column values.
    pivot_table : Create a pivot table as a DataFrame.

    Notes
    -----
    Any Series passed will have their name attributes used unless row or column
    names for the cross-tabulation are specified.

    Any input passed containing Categorical data will have **all** of its
    categories included in the cross-tabulation, even if the actual data does
    not contain any instances of a particular category.

    In the event that there aren\'t overlapping indexes an empty DataFrame will
    be returned.

    Reference :ref:`the user guide <reshaping.crosstabulations>` for more examples.

    Examples
    --------
    >>> a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
    ...               "bar", "bar", "foo", "foo", "foo"], dtype=object)
    >>> b = np.array(["one", "one", "one", "two", "one", "one",
    ...               "one", "two", "two", "two", "one"], dtype=object)
    >>> c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
    ...               "shiny", "dull", "shiny", "shiny", "shiny"],
    ...              dtype=object)
    >>> pd.crosstab(a, [b, c], rownames=[\'a\'], colnames=[\'b\', \'c\'])
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0
    foo    2     2    1     2

    Here \'c\' and \'f\' are not represented in the data and will not be
    shown in the output because dropna is True by default. Set
    dropna=False to preserve categories with no data.

    >>> foo = pd.Categorical([\'a\', \'b\'], categories=[\'a\', \'b\', \'c\'])
    >>> bar = pd.Categorical([\'d\', \'e\'], categories=[\'d\', \'e\', \'f\'])
    >>> pd.crosstab(foo, bar)
    col_0  d  e
    row_0
    a      1  0
    b      0  1
    >>> pd.crosstab(foo, bar, dropna=False)
    col_0  d  e  f
    row_0
    a      1  0  0
    b      0  1  0
    c      0  0  0
    '''
def _normalize(table: DataFrame, normalize, margins: bool, margins_name: Hashable = ...) -> DataFrame: ...
def _get_names(arrs, names, prefix: str = ...): ...
def _build_names_mapper(rownames: list[str], colnames: list[str]) -> tuple[dict[str, str], list[str], dict[str, str], list[str]]:
    """
    Given the names of a DataFrame's rows and columns, returns a set of unique row
    and column names and mappers that convert to original names.

    A row or column name is replaced if it is duplicate among the rows of the inputs,
    among the columns of the inputs or between the rows and the columns.

    Parameters
    ----------
    rownames: list[str]
    colnames: list[str]

    Returns
    -------
    Tuple(Dict[str, str], List[str], Dict[str, str], List[str])

    rownames_mapper: dict[str, str]
        a dictionary with new row names as keys and original rownames as values
    unique_rownames: list[str]
        a list of rownames with duplicate names replaced by dummy names
    colnames_mapper: dict[str, str]
        a dictionary with new column names as keys and original column names as values
    unique_colnames: list[str]
        a list of column names with duplicate names replaced by dummy names

    """

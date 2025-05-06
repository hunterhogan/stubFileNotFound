from _typeshed import Incomplete
from pandas import Categorical as Categorical, Index as Index, IntervalIndex as IntervalIndex
from pandas._libs import Timedelta as Timedelta, Timestamp as Timestamp, lib as lib
from pandas._typing import DtypeObj as DtypeObj, IntervalLeftRight as IntervalLeftRight
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, is_bool_dtype as is_bool_dtype, is_integer as is_integer, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_scalar as is_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype

def cut(x, bins, right: bool = True, labels: Incomplete | None = None, retbins: bool = False, precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise', ordered: bool = True):
    '''
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4]. This argument is ignored when
        `bins` is an IntervalIndex.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same length as
        the resulting bins. If False, returns only integer indicators of the
        bins. This affects the type of the output container (see below).
        This argument is ignored when `bins` is an IntervalIndex. If True,
        raises an error. When `ordered=False`, labels must be provided.
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {default \'raise\', \'drop\'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.
    ordered : bool, default True
        Whether the labels are ordered or not. Applies to returned types
        Categorical and Series (with Categorical dtype). If True,
        the resulting categorical will be ordered. If False, the resulting
        categorical will be unordered (labels must be provided).

    Returns
    -------
    out : Categorical, Series, or ndarray
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`.

        * None (default) : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are Interval dtype.

        * sequence of scalars : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are whatever the type in the sequence is.

        * False : returns an ndarray of integers.

    bins : numpy.ndarray or IntervalIndex.
        The computed or specified bins. Only returned when `retbins=True`.
        For scalar or sequence `bins`, this is an ndarray with the computed
        bins. If set `duplicates=drop`, `bins` will drop non-unique bin. For
        an IntervalIndex `bins`, this is equal to `bins`.

    See Also
    --------
    qcut : Discretize variable into equal-sized buckets based on rank
        or based on sample quantiles.
    Categorical : Array type for storing data that come from a
        fixed set of values.
    Series : One-dimensional array with axis labels (including time series).
    IntervalIndex : Immutable Index implementing an ordered, sliceable set.

    Notes
    -----
    Any NA values will be NA in the result. Out of bounds values will be NA in
    the resulting Series or Categorical object.

    Reference :ref:`the user guide <reshaping.tile.cut>` for more examples.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
    ... # doctest: +ELLIPSIS
    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] ...

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
    ... # doctest: +ELLIPSIS
    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] ...
    array([0.994, 3.   , 5.   , 7.   ]))

    Discovers the same bins, but assign them specific labels. Notice that
    the returned Categorical\'s categories are `labels` and is ordered.

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]),
    ...        3, labels=["bad", "medium", "good"])
    [\'bad\', \'good\', \'medium\', \'medium\', \'good\', \'bad\']
    Categories (3, object): [\'bad\' < \'medium\' < \'good\']

    ``ordered=False`` will result in unordered categories when labels are passed.
    This parameter can be used to allow non-unique labels:

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,
    ...        labels=["B", "A", "B"], ordered=False)
    [\'B\', \'B\', \'A\', \'A\', \'B\', \'B\']
    Categories (2, object): [\'A\', \'B\']

    ``labels=False`` implies you just want the bins back.

    >>> pd.cut([0, 1, 1, 2], bins=4, labels=False)
    array([0, 1, 1, 3])

    Passing a Series as an input returns a Series with categorical dtype:

    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),
    ...               index=[\'a\', \'b\', \'c\', \'d\', \'e\'])
    >>> pd.cut(s, 3)
    ... # doctest: +ELLIPSIS
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64, right]): [(1.992, 4.667] < (4.667, ...

    Passing a Series as an input returns a Series with mapping value.
    It is used to map numerically to intervals based on bins.

    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),
    ...               index=[\'a\', \'b\', \'c\', \'d\', \'e\'])
    >>> pd.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False)
    ... # doctest: +ELLIPSIS
    (a    1.0
     b    2.0
     c    3.0
     d    4.0
     e    NaN
     dtype: float64,
     array([ 0,  2,  4,  6,  8, 10]))

    Use `drop` optional when bins is not unique

    >>> pd.cut(s, [0, 2, 4, 6, 10, 10], labels=False, retbins=True,
    ...        right=False, duplicates=\'drop\')
    ... # doctest: +ELLIPSIS
    (a    1.0
     b    2.0
     c    3.0
     d    3.0
     e    NaN
     dtype: float64,
     array([ 0,  2,  4,  6, 10]))

    Passing an IntervalIndex for `bins` results in those categories exactly.
    Notice that values not covered by the IntervalIndex are set to NaN. 0
    is to the left of the first bin (which is closed on the right), and 1.5
    falls between two bins.

    >>> bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    >>> pd.cut([0, 0.5, 1.5, 2.5, 4.5], bins)
    [NaN, (0.0, 1.0], NaN, (2.0, 3.0], (4.0, 5.0]]
    Categories (3, interval[int64, right]): [(0, 1] < (2, 3] < (4, 5]]
    '''
def qcut(x, q, labels: Incomplete | None = None, retbins: bool = False, precision: int = 3, duplicates: str = 'raise'):
    '''
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based
    on sample quantiles. For example 1000 values for 10 quantiles would
    produce a Categorical object indicating quantile membership for each data point.

    Parameters
    ----------
    x : 1d ndarray or Series
    q : int or list-like of float
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
    labels : array or False, default None
        Used as labels for the resulting bins. Must be of the same length as
        the resulting bins. If False, return only integer indicators of the
        bins. If True, raises an error.
    retbins : bool, optional
        Whether to return the (bins, labels) or not. Can be useful if bins
        is given as a scalar.
    precision : int, optional
        The precision at which to store and display the bins labels.
    duplicates : {default \'raise\', \'drop\'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns
    -------
    out : Categorical or Series or array of integers if labels is False
        The return type (Categorical or Series) depends on the input: a Series
        of type category if input is a Series else Categorical. Bins are
        represented as categories when categorical data is returned.
    bins : ndarray of floats
        Returned only if `retbins` is True.

    Notes
    -----
    Out of bounds values will be NA in the resulting Categorical object

    Examples
    --------
    >>> pd.qcut(range(5), 4)
    ... # doctest: +ELLIPSIS
    [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
    Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] ...

    >>> pd.qcut(range(5), 3, labels=["good", "medium", "bad"])
    ... # doctest: +SKIP
    [good, good, medium, bad, bad]
    Categories (3, object): [good < medium < bad]

    >>> pd.qcut(range(5), 4, labels=False)
    array([0, 0, 1, 2, 3])
    '''
def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
    """
    If a user passed an integer N for bins, convert this to a sequence of N
    equal(ish)-sized bins.
    """
def _bins_to_cuts(x_idx: Index, bins: Index, right: bool = True, labels: Incomplete | None = None, precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise', ordered: bool = True): ...
def _coerce_to_type(x: Index) -> tuple[Index, DtypeObj | None]:
    """
    if the passed data is of datetime/timedelta, bool or nullable int type,
    this method converts it to numeric so that cut or qcut method can
    handle it
    """
def _is_dt_or_td(dtype: DtypeObj) -> bool: ...
def _format_labels(bins: Index, precision: int, right: bool = True, include_lowest: bool = False):
    """based on the dtype, return our labels"""
def _preprocess_for_cut(x) -> Index:
    """
    handles preprocessing for cut where we convert passed
    input to array, strip the index information and store it
    separately
    """
def _postprocess_for_cut(fac, bins, retbins: bool, original):
    """
    handles post processing for the cut method where
    we combine the index information if the originally passed
    datatype was a series
    """
def _round_frac(x, precision: int):
    """
    Round the fractional part of the given number
    """
def _infer_precision(base_precision: int, bins: Index) -> int:
    """
    Infer an appropriate precision for _round_frac
    """

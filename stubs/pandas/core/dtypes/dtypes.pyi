import numpy as np
import pyarrow as pa
from _typeshed import Incomplete
from collections.abc import MutableMapping
from datetime import tzinfo
from pandas import Categorical as Categorical, CategoricalIndex as CategoricalIndex, DatetimeIndex as DatetimeIndex, Index as Index, IntervalIndex as IntervalIndex, PeriodIndex as PeriodIndex
from pandas._libs import lib as lib, missing as libmissing
from pandas._libs.interval import Interval as Interval
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs import BaseOffset as BaseOffset, NaT as NaT, NaTType as NaTType, Period as Period, Timedelta as Timedelta, Timestamp as Timestamp, timezones as timezones, to_offset as to_offset, tz_compare as tz_compare
from pandas._libs.tslibs.dtypes import PeriodDtypeBase as PeriodDtypeBase, abbrev_to_npy_unit as abbrev_to_npy_unit
from pandas._libs.tslibs.offsets import BDay as BDay
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, IntervalClosedType as IntervalClosedType, Ordered as Ordered, Self as Self, npt as npt, type_t as type_t
from pandas.compat import pa_version_under10p1 as pa_version_under10p1
from pandas.core.arrays import BaseMaskedArray as BaseMaskedArray, DatetimeArray as DatetimeArray, IntervalArray as IntervalArray, NumpyExtensionArray as NumpyExtensionArray, PeriodArray as PeriodArray, SparseArray as SparseArray
from pandas.core.arrays.arrow import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype, StorageExtensionDtype as StorageExtensionDtype, register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.generic import ABCCategoricalIndex as ABCCategoricalIndex, ABCIndex as ABCIndex, ABCRangeIndex as ABCRangeIndex
from pandas.core.dtypes.inference import is_bool as is_bool, is_list_like as is_list_like
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util import capitalize_first_letter as capitalize_first_letter
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any

str_type = str

class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """
    type: Any
    kind: Any
    subdtype: Incomplete
    str: str_type
    num: int
    shape: tuple[int, ...]
    itemsize: int
    base: DtypeObj | None
    isbuiltin: int
    isnative: int
    _cache_dtypes: dict[str_type, PandasExtensionDtype]
    def __repr__(self) -> str_type:
        """
        Return a string representation for a particular object.
        """
    def __hash__(self) -> int: ...
    def __getstate__(self) -> dict[str_type, Any]: ...
    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""

class CategoricalDtypeType(type):
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """

class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    """
    Type for categorical data with the categories and orderedness.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as a ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    None

    See Also
    --------
    Categorical : Represent a categorical variable in classic R / S-plus fashion.

    Notes
    -----
    This class is useful for specifying the type of a ``Categorical``
    independent of the values. See :ref:`categorical.categoricaldtype`
    for more.

    Examples
    --------
    >>> t = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
    >>> pd.Series(['a', 'b', 'a', 'c'], dtype=t)
    0      a
    1      b
    2      a
    3    NaN
    dtype: category
    Categories (2, object): ['b' < 'a']

    An empty CategoricalDtype with a specific dtype can be created
    by providing an empty index. As follows,

    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype
    dtype('<M8[ns]')
    """
    name: str
    type: type[CategoricalDtypeType]
    kind: str_type
    str: str
    base: Incomplete
    _metadata: Incomplete
    _cache_dtypes: dict[str_type, PandasExtensionDtype]
    _supports_2d: bool
    _can_fast_transpose: bool
    def __init__(self, categories: Incomplete | None = None, ordered: Ordered = False) -> None: ...
    @classmethod
    def _from_fastpath(cls, categories: Incomplete | None = None, ordered: bool | None = None) -> CategoricalDtype: ...
    @classmethod
    def _from_categorical_dtype(cls, dtype: CategoricalDtype, categories: Incomplete | None = None, ordered: Ordered | None = None) -> CategoricalDtype: ...
    @classmethod
    def _from_values_or_dtype(cls, values: Incomplete | None = None, categories: Incomplete | None = None, ordered: bool | None = None, dtype: Dtype | None = None) -> CategoricalDtype:
        '''
        Construct dtype from the input parameters used in :class:`Categorical`.

        This constructor method specifically does not do the factorization
        step, if that is needed to find the categories. This constructor may
        therefore return ``CategoricalDtype(categories=None, ordered=None)``,
        which may not be useful. Additional steps may therefore have to be
        taken to create the final dtype.

        The return dtype is specified from the inputs in this prioritized
        order:
        1. if dtype is a CategoricalDtype, return dtype
        2. if dtype is the string \'category\', create a CategoricalDtype from
           the supplied categories and ordered parameters, and return that.
        3. if values is a categorical, use value.dtype, but override it with
           categories and ordered if either/both of those are not None.
        4. if dtype is None and values is not a categorical, construct the
           dtype from categories and ordered, even if either of those is None.

        Parameters
        ----------
        values : list-like, optional
            The list-like must be 1-dimensional.
        categories : list-like, optional
            Categories for the CategoricalDtype.
        ordered : bool, optional
            Designating if the categories are ordered.
        dtype : CategoricalDtype or the string "category", optional
            If ``CategoricalDtype``, cannot be used together with
            `categories` or `ordered`.

        Returns
        -------
        CategoricalDtype

        Examples
        --------
        >>> pd.CategoricalDtype._from_values_or_dtype()
        CategoricalDtype(categories=None, ordered=None, categories_dtype=None)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     categories=[\'a\', \'b\'], ordered=True
        ... )
        CategoricalDtype(categories=[\'a\', \'b\'], ordered=True, categories_dtype=object)
        >>> dtype1 = pd.CategoricalDtype([\'a\', \'b\'], ordered=True)
        >>> dtype2 = pd.CategoricalDtype([\'x\', \'y\'], ordered=False)
        >>> c = pd.Categorical([0, 1], dtype=dtype1)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     c, [\'x\', \'y\'], ordered=True, dtype=dtype2
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot specify `categories` or `ordered` together with
        `dtype`.

        The supplied dtype takes precedence over values\' dtype:

        >>> pd.CategoricalDtype._from_values_or_dtype(c, dtype=dtype2)
        CategoricalDtype(categories=[\'x\', \'y\'], ordered=False, categories_dtype=object)
        '''
    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        '''
        Construct a CategoricalDtype from a string.

        Parameters
        ----------
        string : str
            Must be the string "category" in order to be successfully constructed.

        Returns
        -------
        CategoricalDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a CategoricalDtype cannot be constructed from the input.
        '''
    _categories: Incomplete
    _ordered: Incomplete
    def _finalize(self, categories, ordered: Ordered, fastpath: bool = False) -> None: ...
    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool:
        """
        Rules for CDT equality:
        1) Any CDT is equal to the string 'category'
        2) Any CDT is equal to itself
        3) Any CDT is equal to a CDT with categories=None regardless of ordered
        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
    def __repr__(self) -> str_type: ...
    def _hash_categories(self) -> int: ...
    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        """
        Validates that we have a valid ordered parameter. If
        it is not a boolean, a TypeError will be raised.

        Parameters
        ----------
        ordered : object
            The parameter to be verified.

        Raises
        ------
        TypeError
            If 'ordered' is not a boolean.
        """
    @staticmethod
    def validate_categories(categories, fastpath: bool = False) -> Index:
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype

        Returns
        -------
        new_dtype : CategoricalDtype
        """
    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.categories
        Index(['a', 'b'], dtype='object')
        """
    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.ordered
        True

        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=False)
        >>> cat_type.ordered
        False
        """
    @property
    def _is_boolean(self) -> bool: ...
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def index_class(self) -> type_t[CategoricalIndex]: ...

class DatetimeTZDtype(PandasExtensionDtype):
    '''
    An ExtensionDtype for timezone-aware datetime data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    unit : str, default "ns"
        The precision of the datetime data. Currently limited
        to ``"ns"``.
    tz : str, int, or datetime.tzinfo
        The timezone.

    Attributes
    ----------
    unit
    tz

    Methods
    -------
    None

    Raises
    ------
    ZoneInfoNotFoundError
        When the requested timezone cannot be found.

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> pd.DatetimeTZDtype(tz=ZoneInfo(\'UTC\'))
    datetime64[ns, UTC]

    >>> pd.DatetimeTZDtype(tz=ZoneInfo(\'Europe/Paris\'))
    datetime64[ns, Europe/Paris]
    '''
    type: type[Timestamp]
    kind: str_type
    num: int
    _metadata: Incomplete
    _match: Incomplete
    _cache_dtypes: dict[str_type, PandasExtensionDtype]
    _supports_2d: bool
    _can_fast_transpose: bool
    @property
    def na_value(self) -> NaTType: ...
    def base(self) -> DtypeObj: ...
    def str(self) -> str: ...
    _unit: Incomplete
    _tz: Incomplete
    def __init__(self, unit: str_type | DatetimeTZDtype = 'ns', tz: Incomplete | None = None) -> None: ...
    def _creso(self) -> int:
        """
        The NPY_DATETIMEUNIT corresponding to this dtype's resolution.
        """
    @property
    def unit(self) -> str_type:
        """
        The precision of the datetime data.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.unit
        'ns'
        """
    @property
    def tz(self) -> tzinfo:
        """
        The timezone.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.tz
        zoneinfo.ZoneInfo(key='America/Los_Angeles')
        """
    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        """
        Construct a DatetimeTZDtype from a string.

        Parameters
        ----------
        string : str
            The string alias for this DatetimeTZDtype.
            Should be formatted like ``datetime64[ns, <tz>]``,
            where ``<tz>`` is the timezone name.

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string('datetime64[ns, UTC]')
        datetime64[ns, UTC]
        """
    def __str__(self) -> str_type: ...
    @property
    def name(self) -> str_type:
        """A string representation of the dtype."""
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> DatetimeArray:
        """
        Construct DatetimeArray from pyarrow Array/ChunkedArray.

        Note: If the units in the pyarrow Array are the same as this
        DatetimeDtype, then values corresponding to the integer representation
        of ``NaT`` (e.g. one nanosecond before :attr:`pandas.Timestamp.min`)
        are converted to ``NaT``, regardless of the null indicator in the
        pyarrow array.

        Parameters
        ----------
        array : pyarrow.Array or pyarrow.ChunkedArray
            The Arrow array to convert to DatetimeArray.

        Returns
        -------
        extension array : DatetimeArray
        """
    def __setstate__(self, state) -> None: ...
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def index_class(self) -> type_t[DatetimeIndex]: ...

class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    Examples
    --------
    >>> pd.PeriodDtype(freq='D')
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """
    type: type[Period]
    kind: str_type
    str: str
    base: Incomplete
    num: int
    _metadata: Incomplete
    _match: Incomplete
    _cache_dtypes: dict[BaseOffset, int]
    __hash__: Incomplete
    _freq: BaseOffset
    _supports_2d: bool
    _can_fast_transpose: bool
    def __new__(cls, freq) -> PeriodDtype:
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]: ...
    @property
    def freq(self) -> BaseOffset:
        """
        The frequency object of this PeriodDtype.

        Examples
        --------
        >>> dtype = pd.PeriodDtype(freq='D')
        >>> dtype.freq
        <Day>
        """
    @classmethod
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset: ...
    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
    def __str__(self) -> str_type: ...
    @property
    def name(self) -> str_type: ...
    @property
    def na_value(self) -> NaTType: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
    @classmethod
    def construct_array_type(cls) -> type_t[PeriodArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> PeriodArray:
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
    def index_class(self) -> type_t[PeriodIndex]: ...

class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    Examples
    --------
    >>> pd.IntervalDtype(subtype='int64', closed='both')
    interval[int64, both]
    """
    name: str
    kind: str_type
    str: str
    base: Incomplete
    num: int
    _metadata: Incomplete
    _match: Incomplete
    _cache_dtypes: dict[str_type, PandasExtensionDtype]
    _subtype: None | np.dtype
    _closed: IntervalClosedType | None
    def __init__(self, subtype: Incomplete | None = None, closed: IntervalClosedType | None = None) -> None: ...
    def _can_hold_na(self) -> bool: ...
    @property
    def closed(self) -> IntervalClosedType: ...
    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.

        Examples
        --------
        >>> dtype = pd.IntervalDtype(subtype='int64', closed='both')
        >>> dtype.subtype
        dtype('int64')
        """
    @classmethod
    def construct_array_type(cls) -> type[IntervalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
    @property
    def type(self) -> type[Interval]: ...
    def __str__(self) -> str_type: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __setstate__(self, state) -> None: ...
    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> IntervalArray:
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def index_class(self) -> type_t[IntervalIndex]: ...

class NumpyEADtype(ExtensionDtype):
    """
    A Pandas ExtensionDtype for NumPy dtypes.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    dtype : object
        Object to be converted to a NumPy data type object.

    See Also
    --------
    numpy.dtype
    """
    _metadata: Incomplete
    _supports_2d: bool
    _can_fast_transpose: bool
    _dtype: Incomplete
    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def numpy_dtype(self) -> np.dtype:
        """
        The NumPy dtype this NumpyEADtype wraps.
        """
    @property
    def name(self) -> str:
        """
        A bit-width name for this data-type.
        """
    @property
    def type(self) -> type[np.generic]:
        """
        The type object used to instantiate a scalar of this NumPy data-type.
        """
    @property
    def _is_numeric(self) -> bool: ...
    @property
    def _is_boolean(self) -> bool: ...
    @classmethod
    def construct_from_string(cls, string: str) -> NumpyEADtype: ...
    @classmethod
    def construct_array_type(cls) -> type_t[NumpyExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV') identifying the general kind of data.
        """
    @property
    def itemsize(self) -> int:
        """
        The element size of this data-type object.
        """

class BaseMaskedDtype(ExtensionDtype):
    """
    Base class for dtypes for BaseMaskedArray subclasses.
    """
    base: Incomplete
    type: type
    @property
    def na_value(self) -> libmissing.NAType: ...
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of our numpy dtype"""
    def kind(self) -> str: ...
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
    @classmethod
    def construct_array_type(cls) -> type_t[BaseMaskedArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
        """
        Construct the MaskedDtype corresponding to the given numpy dtype.
        """
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...

class SparseDtype(ExtensionDtype):
    """
    Dtype for data stored in :class:`SparseArray`.

    This dtype implements the pandas ExtensionDtype interface.

    Parameters
    ----------
    dtype : str, ExtensionDtype, numpy.dtype, type, default numpy.float64
        The dtype of the underlying array storing the non-fill value values.
    fill_value : scalar, optional
        The scalar value not stored in the SparseArray. By default, this
        depends on `dtype`.

        =========== ==========
        dtype       na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        ``False``
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The default value may be overridden by specifying a `fill_value`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> ser = pd.Series([1, 0, 0], dtype=pd.SparseDtype(dtype=int, fill_value=0))
    >>> ser
    0    1
    1    0
    2    0
    dtype: Sparse[int64, 0]
    >>> ser.sparse.density
    0.3333333333333333
    """
    _is_immutable: bool
    _metadata: Incomplete
    _dtype: Incomplete
    _fill_value: Incomplete
    def __init__(self, dtype: Dtype = ..., fill_value: Any = None) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def fill_value(self):
        """
        The fill value of the array.

        Converting the SparseArray to a dense ndarray will fill the
        array with this value.

        .. warning::

           It's possible to end up with a SparseArray that has ``fill_value``
           values in ``sp_values``. This can occur, for example, when setting
           ``SparseArray.fill_value`` directly.
        """
    def _check_fill_value(self) -> None: ...
    @property
    def _is_na_fill_value(self) -> bool: ...
    @property
    def _is_numeric(self) -> bool: ...
    @property
    def _is_boolean(self) -> bool: ...
    @property
    def kind(self) -> str:
        """
        The sparse kind. Either 'integer', or 'block'.
        """
    @property
    def type(self): ...
    @property
    def subtype(self): ...
    @property
    def name(self) -> str: ...
    def __repr__(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> type_t[SparseArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def construct_from_string(cls, string: str) -> SparseDtype:
        """
        Construct a SparseDtype from a string form.

        Parameters
        ----------
        string : str
            Can take the following forms.

            string           dtype
            ================ ============================
            'int'            SparseDtype[np.int64, 0]
            'Sparse'         SparseDtype[np.float64, nan]
            'Sparse[int]'    SparseDtype[np.int64, 0]
            'Sparse[int, 0]' SparseDtype[np.int64, 0]
            ================ ============================

            It is not possible to specify non-default fill values
            with a string. An argument like ``'Sparse[int, 1]'``
            will raise a ``TypeError`` because the default fill value
            for integers is 0.

        Returns
        -------
        SparseDtype
        """
    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        """
        Parse a string to get the subtype

        Parameters
        ----------
        dtype : str
            A string like

            * Sparse[subtype]
            * Sparse[subtype, fill_value]

        Returns
        -------
        subtype : str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted.
        """
    @classmethod
    def is_dtype(cls, dtype: object) -> bool: ...
    def update_dtype(self, dtype) -> SparseDtype:
        """
        Convert the SparseDtype to a new dtype.

        This takes care of converting the ``fill_value``.

        Parameters
        ----------
        dtype : Union[str, numpy.dtype, SparseDtype]
            The new dtype to use.

            * For a SparseDtype, it is simply returned
            * For a NumPy dtype (or str), the current fill value
              is converted to the new dtype, and a SparseDtype
              with `dtype` and the new fill value is returned.

        Returns
        -------
        SparseDtype
            A new SparseDtype with the correct `dtype` and fill value
            for that `dtype`.

        Raises
        ------
        ValueError
            When the current fill value cannot be converted to the
            new `dtype` (e.g. trying to convert ``np.nan`` to an
            integer dtype).


        Examples
        --------
        >>> SparseDtype(int, 0).update_dtype(float)
        Sparse[float64, 0.0]

        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))
        Sparse[float64, nan]
        """
    @property
    def _subtype_with_str(self):
        """
        Whether the SparseDtype's subtype should be considered ``str``.

        Typically, pandas will store string data in an object-dtype array.
        When converting values to a dtype, e.g. in ``.astype``, we need to
        be more specific, we need the actual underlying type.

        Returns
        -------
        >>> SparseDtype(int, 1)._subtype_with_str
        dtype('int64')

        >>> SparseDtype(object, 1)._subtype_with_str
        dtype('O')

        >>> dtype = SparseDtype(str, '')
        >>> dtype.subtype
        dtype('O')

        >>> dtype._subtype_with_str
        <class 'str'>
        """
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...

class ArrowDtype(StorageExtensionDtype):
    '''
    An ExtensionDtype for PyArrow data types.

    .. warning::

       ArrowDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    While most ``dtype`` arguments can accept the "string"
    constructor, e.g. ``"int64[pyarrow]"``, ArrowDtype is useful
    if the data type contains parameters like ``pyarrow.timestamp``.

    Parameters
    ----------
    pyarrow_dtype : pa.DataType
        An instance of a `pyarrow.DataType <https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions>`__.

    Attributes
    ----------
    pyarrow_dtype

    Methods
    -------
    None

    Returns
    -------
    ArrowDtype

    Examples
    --------
    >>> import pyarrow as pa
    >>> pd.ArrowDtype(pa.int64())
    int64[pyarrow]

    Types with parameters must be constructed with ArrowDtype.

    >>> pd.ArrowDtype(pa.timestamp("s", tz="America/New_York"))
    timestamp[s, tz=America/New_York][pyarrow]
    >>> pd.ArrowDtype(pa.list_(pa.int64()))
    list<item: int64>[pyarrow]
    '''
    _metadata: Incomplete
    pyarrow_dtype: Incomplete
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def type(self):
        """
        Returns associated scalar type.
        """
    @property
    def name(self) -> str:
        """
        A string identifying the data type.
        """
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of the related numpy dtype"""
    def kind(self) -> str: ...
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
    @classmethod
    def construct_array_type(cls) -> type_t[ArrowExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def construct_from_string(cls, string: str) -> ArrowDtype:
        '''
        Construct this type from a string.

        Parameters
        ----------
        string : str
            string should follow the format f"{pyarrow_type}[pyarrow]"
            e.g. int64[pyarrow]
        '''
    @classmethod
    def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
        """
        Construct a temporal ArrowDtype from string.
        """
    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.
        """
    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.
        """
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray):
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """

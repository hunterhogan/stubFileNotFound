import np
import npt
import numpy.dtypes
import pa
import pandas._libs.lib as lib
import pandas._libs.missing as libmissing
import pandas._libs.tslibs.dtypes
import pandas._libs.tslibs.period
import pandas._libs.tslibs.timestamps
import pandas._libs.tslibs.timezones as timezones
import pandas.core.dtypes.base
import re
from _typeshed import Incomplete
from builtins import str_type
from pandas._libs.interval import Interval as Interval
from pandas._libs.lib import is_bool as is_bool, is_list_like as is_list_like
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.dtypes import PeriodDtypeBase as PeriodDtypeBase, abbrev_to_npy_unit as abbrev_to_npy_unit
from pandas._libs.tslibs.nattype import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.offsets import BDay as BDay, BaseOffset as BaseOffset, to_offset as to_offset
from pandas._libs.tslibs.period import Period as Period
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas._libs.tslibs.timezones import tz_compare as tz_compare
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype, StorageExtensionDtype as StorageExtensionDtype, register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.generic import ABCCategoricalIndex as ABCCategoricalIndex, ABCIndex as ABCIndex, ABCRangeIndex as ABCRangeIndex
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util import capitalize_first_letter as capitalize_first_letter
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, ClassVar

TYPE_CHECKING: bool
pa_version_under10p1: bool

class PandasExtensionDtype(pandas.core.dtypes.base.ExtensionDtype):
    subdtype: ClassVar[None] = ...
    num: ClassVar[int] = ...
    shape: ClassVar[tuple] = ...
    itemsize: ClassVar[int] = ...
    base: ClassVar[None] = ...
    isbuiltin: ClassVar[int] = ...
    isnative: ClassVar[int] = ...
    _cache_dtypes: ClassVar[dict] = ...
    def __hash__(self) -> int: ...
    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""

class CategoricalDtypeType(type): ...

class CategoricalDtype(PandasExtensionDtype):
    class type(type): ...
    name: ClassVar[str] = ...
    kind: ClassVar[str] = ...
    str: ClassVar[str] = ...
    base: ClassVar[numpy.dtypes.ObjectDType] = ...
    _metadata: ClassVar[tuple] = ...
    _cache_dtypes: ClassVar[dict] = ...
    _supports_2d: ClassVar[bool] = ...
    _can_fast_transpose: ClassVar[bool] = ...
    _hash_categories: Incomplete
    index_class: Incomplete
    def __init__(self, categories, ordered: Ordered = ...) -> None: ...
    @classmethod
    def _from_fastpath(cls, categories, ordered: bool | None) -> CategoricalDtype: ...
    @classmethod
    def _from_categorical_dtype(cls, dtype: CategoricalDtype, categories, ordered: Ordered | None) -> CategoricalDtype: ...
    @classmethod
    def _from_values_or_dtype(cls, values, categories, ordered: bool | None, dtype: Dtype | None) -> CategoricalDtype:
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
    def _finalize(self, categories, ordered: Ordered, fastpath: bool = ...) -> None: ...
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
    def validate_categories(categories, fastpath: bool = ...) -> Index:
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
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    @property
    def categories(self): ...
    @property
    def ordered(self): ...
    @property
    def _is_boolean(self): ...

class DatetimeTZDtype(PandasExtensionDtype):
    type: ClassVar[type[pandas._libs.tslibs.timestamps.Timestamp]] = ...
    kind: ClassVar[str] = ...
    num: ClassVar[int] = ...
    _metadata: ClassVar[tuple] = ...
    _match: ClassVar[re.Pattern] = ...
    _cache_dtypes: ClassVar[dict] = ...
    _supports_2d: ClassVar[bool] = ...
    _can_fast_transpose: ClassVar[bool] = ...
    base: Incomplete
    str: Incomplete
    _creso: Incomplete
    index_class: Incomplete
    def __init__(self, unit: str_type | DatetimeTZDtype = ..., tz) -> None: ...
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
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    @property
    def na_value(self): ...
    @property
    def unit(self): ...
    @property
    def tz(self): ...
    @property
    def name(self): ...

class PeriodDtype(pandas._libs.tslibs.dtypes.PeriodDtypeBase, PandasExtensionDtype):
    type: ClassVar[type[pandas._libs.tslibs.period.Period]] = ...
    kind: ClassVar[str] = ...
    str: ClassVar[str] = ...
    base: ClassVar[numpy.dtypes.ObjectDType] = ...
    num: ClassVar[int] = ...
    _metadata: ClassVar[tuple] = ...
    _match: ClassVar[re.Pattern] = ...
    _cache_dtypes: ClassVar[dict] = ...
    __hash__: ClassVar[wrapper_descriptor] = ...
    _supports_2d: ClassVar[bool] = ...
    _can_fast_transpose: ClassVar[bool] = ...
    index_class: Incomplete
    @classmethod
    def __init__(cls, freq) -> PeriodDtype:
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]: ...
    @classmethod
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset: ...
    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
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
    @property
    def freq(self): ...
    @property
    def name(self): ...
    @property
    def na_value(self): ...

class IntervalDtype(PandasExtensionDtype):
    name: ClassVar[str] = ...
    kind: ClassVar[str] = ...
    str: ClassVar[str] = ...
    base: ClassVar[numpy.dtypes.ObjectDType] = ...
    num: ClassVar[int] = ...
    _metadata: ClassVar[tuple] = ...
    _match: ClassVar[re.Pattern] = ...
    _cache_dtypes: ClassVar[dict] = ...
    _can_hold_na: Incomplete
    index_class: Incomplete
    def __init__(self, subtype, closed: IntervalClosedType | None) -> None: ...
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
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
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
    @property
    def closed(self): ...
    @property
    def subtype(self): ...
    @property
    def type(self): ...

class NumpyEADtype(pandas.core.dtypes.base.ExtensionDtype):
    _metadata: ClassVar[tuple] = ...
    _supports_2d: ClassVar[bool] = ...
    _can_fast_transpose: ClassVar[bool] = ...
    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None: ...
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
    def numpy_dtype(self): ...
    @property
    def name(self): ...
    @property
    def type(self): ...
    @property
    def _is_numeric(self): ...
    @property
    def _is_boolean(self): ...
    @property
    def kind(self): ...
    @property
    def itemsize(self): ...

class BaseMaskedDtype(pandas.core.dtypes.base.ExtensionDtype):
    base: ClassVar[None] = ...
    numpy_dtype: Incomplete
    kind: Incomplete
    itemsize: Incomplete
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
    @property
    def na_value(self): ...

class SparseDtype(pandas.core.dtypes.base.ExtensionDtype):
    _is_immutable: ClassVar[bool] = ...
    _metadata: ClassVar[tuple] = ...
    def __init__(self, dtype: Dtype = ..., fill_value: Any) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def _check_fill_value(self) -> None: ...
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
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    @property
    def fill_value(self): ...
    @property
    def _is_na_fill_value(self): ...
    @property
    def _is_numeric(self): ...
    @property
    def _is_boolean(self): ...
    @property
    def kind(self): ...
    @property
    def type(self): ...
    @property
    def subtype(self): ...
    @property
    def name(self): ...
    @property
    def _subtype_with_str(self): ...

class ArrowDtype(pandas.core.dtypes.base.StorageExtensionDtype):
    _metadata: ClassVar[tuple] = ...
    numpy_dtype: Incomplete
    kind: Incomplete
    itemsize: Incomplete
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
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
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray):
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
    @property
    def type(self): ...
    @property
    def name(self): ...
    @property
    def _is_numeric(self): ...
    @property
    def _is_boolean(self): ...

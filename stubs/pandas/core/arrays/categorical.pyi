import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator, Sequence
from pandas import DataFrame as DataFrame, Index as Index, Series as Series
from pandas._config import get_option as get_option
from pandas._libs import NaT as NaT, lib as lib
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._typing import ArrayLike as ArrayLike, AstypeArg as AstypeArg, AxisInt as AxisInt, Dtype as Dtype, DtypeObj as DtypeObj, NpDtype as NpDtype, Ordered as Ordered, Self as Self, Shape as Shape, SortKind as SortKind, npt as npt
from pandas.core import algorithms as algorithms, arraylike as arraylike, ops as ops
from pandas.core.accessor import PandasDelegate as PandasDelegate, delegate_names as delegate_names
from pandas.core.algorithms import factorize as factorize, take_nd as take_nd
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray, ravel_compat as ravel_compat
from pandas.core.base import ExtensionArray as ExtensionArray, NoNewAttributesMixin as NoNewAttributesMixin, PandasObject as PandasObject
from pandas.core.construction import extract_array as extract_array, sanitize_array as sanitize_array
from pandas.core.dtypes.cast import coerce_indexer_dtype as coerce_indexer_dtype, find_common_type as find_common_type
from pandas.core.dtypes.common import ensure_int64 as ensure_int64, ensure_platform_int as ensure_platform_int, is_any_real_numeric_dtype as is_any_real_numeric_dtype, is_bool_dtype as is_bool_dtype, is_dict_like as is_dict_like, is_hashable as is_hashable, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_scalar as is_scalar, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, CategoricalDtype as CategoricalDtype, CategoricalDtypeType as CategoricalDtypeType, ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna
from pandas.core.ops.common import unpack_zerodim_and_defer as unpack_zerodim_and_defer
from pandas.core.sorting import nargsort as nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin as ObjectStringArrayMixin
from pandas.io.formats import console as console
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg
from typing import Literal, overload

def _cat_compare_op(op): ...
def contains(cat, key, container) -> bool:
    """
    Helper for membership check for ``key`` in ``cat``.

    This is a helper method for :method:`__contains__`
    and :class:`CategoricalIndex.__contains__`.

    Returns True if ``key`` is in ``cat.categories`` and the
    location of ``key`` in ``categories`` is in ``container``.

    Parameters
    ----------
    cat : :class:`Categorical`or :class:`categoricalIndex`
    key : a hashable object
        The key to check membership for.
    container : Container (e.g. list-like or mapping)
        The container to check for membership in.

    Returns
    -------
    is_in : bool
        True if ``key`` is in ``self.categories`` and location of
        ``key`` in ``categories`` is in ``container``, else False.

    Notes
    -----
    This method does not check for NaN values. Do that separately
    before calling this method.
    """

class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):
    """
    Represent a categorical variable in classic R / S-plus fashion.

    `Categoricals` can only take on a limited, and usually fixed, number
    of possible values (`categories`). In contrast to statistical categorical
    variables, a `Categorical` might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    All values of the `Categorical` are either in `categories` or `np.nan`.
    Assigning values outside of `categories` will raise a `ValueError`. Order
    is defined by the order of the `categories`, not lexical order of the
    values.

    Parameters
    ----------
    values : list-like
        The values of the categorical. If categories are given, values not in
        categories will be replaced with NaN.
    categories : Index-like (unique), optional
        The unique categories for this categorical. If not given, the
        categories are assumed to be the unique values of `values` (sorted, if
        possible, otherwise in the order in which they appear).
    ordered : bool, default False
        Whether or not this categorical is treated as a ordered categorical.
        If True, the resulting categorical will be ordered.
        An ordered categorical respects, when sorted, the order of its
        `categories` attribute (which in turn is the `categories` argument, if
        provided).
    dtype : CategoricalDtype
        An instance of ``CategoricalDtype`` to use for this categorical.

    Attributes
    ----------
    categories : Index
        The categories of this categorical.
    codes : ndarray
        The codes (integer positions, which point to the categories) of this
        categorical, read only.
    ordered : bool
        Whether or not this Categorical is ordered.
    dtype : CategoricalDtype
        The instance of ``CategoricalDtype`` storing the ``categories``
        and ``ordered``.

    Methods
    -------
    from_codes
    __array__

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    CategoricalDtype : Type for categorical data.
    CategoricalIndex : An Index with an underlying ``Categorical``.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`__
    for more.

    Examples
    --------
    >>> pd.Categorical([1, 2, 3, 1, 2, 3])
    [1, 2, 3, 1, 2, 3]
    Categories (3, int64): [1, 2, 3]

    >>> pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Missing values are not included as a category.

    >>> c = pd.Categorical([1, 2, 3, 1, 2, 3, np.nan])
    >>> c
    [1, 2, 3, 1, 2, 3, NaN]
    Categories (3, int64): [1, 2, 3]

    However, their presence is indicated in the `codes` attribute
    by code `-1`.

    >>> c.codes
    array([ 0,  1,  2,  0,  1,  2, -1], dtype=int8)

    Ordered `Categoricals` can be sorted according to the custom order
    of the categories and can have a min and max value.

    >>> c = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,
    ...                    categories=['c', 'b', 'a'])
    >>> c
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['c' < 'b' < 'a']
    >>> c.min()
    'c'
    """
    __array_priority__: int
    _hidden_attrs: Incomplete
    _typ: str
    _dtype: CategoricalDtype
    @classmethod
    def _simple_new(cls, codes: np.ndarray, dtype: CategoricalDtype) -> Self: ...
    def __init__(self, values, categories: Incomplete | None = None, ordered: Incomplete | None = None, dtype: Dtype | None = None, fastpath: bool | lib.NoDefault = ..., copy: bool = True) -> None: ...
    @property
    def dtype(self) -> CategoricalDtype:
        """
        The :class:`~pandas.api.types.CategoricalDtype` for this instance.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat
        ['a', 'b']
        Categories (2, object): ['a' < 'b']
        >>> cat.dtype
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=object)
        """
    @property
    def _internal_fill_value(self) -> int: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False) -> Self: ...
    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self: ...
    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...
    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    def to_list(self):
        """
        Alias for tolist.
        """
    @classmethod
    def _from_inferred_categories(cls, inferred_categories, inferred_codes, dtype, true_values: Incomplete | None = None) -> Self:
        '''
        Construct a Categorical from inferred values.

        For inferred categories (`dtype` is None) the categories are sorted.
        For explicit `dtype`, the `inferred_categories` are cast to the
        appropriate type.

        Parameters
        ----------
        inferred_categories : Index
        inferred_codes : Index
        dtype : CategoricalDtype or \'category\'
        true_values : list, optional
            If none are provided, the default ones are
            "True", "TRUE", and "true."

        Returns
        -------
        Categorical
        '''
    @classmethod
    def from_codes(cls, codes, categories: Incomplete | None = None, ordered: Incomplete | None = None, dtype: Dtype | None = None, validate: bool = True) -> Self:
        '''
        Make a Categorical type from codes and categories or dtype.

        This constructor is useful if you already have codes and
        categories/dtype and so do not need the (computation intensive)
        factorization step, which is usually done on the constructor.

        If your data does not follow this convention, please use the normal
        constructor.

        Parameters
        ----------
        codes : array-like of int
            An integer array, where each integer points to a category in
            categories or dtype.categories, or else is -1 for NaN.
        categories : index-like, optional
            The categories for the categorical. Items need to be unique.
            If the categories are not given here, then they must be provided
            in `dtype`.
        ordered : bool, optional
            Whether or not this categorical is treated as an ordered
            categorical. If not given here or in `dtype`, the resulting
            categorical will be unordered.
        dtype : CategoricalDtype or "category", optional
            If :class:`CategoricalDtype`, cannot be used together with
            `categories` or `ordered`.
        validate : bool, default True
            If True, validate that the codes are valid for the dtype.
            If False, don\'t validate that the codes are valid. Be careful about skipping
            validation, as invalid codes can lead to severe problems, such as segfaults.

            .. versionadded:: 2.1.0

        Returns
        -------
        Categorical

        Examples
        --------
        >>> dtype = pd.CategoricalDtype([\'a\', \'b\'], ordered=True)
        >>> pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
        [\'a\', \'b\', \'a\', \'b\']
        Categories (2, object): [\'a\' < \'b\']
        '''
    @property
    def categories(self) -> Index:
        """
        The categories of this categorical.

        Setting assigns new values to each category (effectively a rename of
        each individual category).

        The assigned value has to be a list-like object. All items must be
        unique and the number of items in the new categories must be the same
        as the number of items in the old categories.

        Raises
        ------
        ValueError
            If the new categories do not validate as categories or if the
            number of new categories is unequal the number of old categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c', 'd'])
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.categories
        Index(['b', 'c', 'd'], dtype='object')

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.categories
        Index(['a', 'b'], dtype='object')

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'c', 'b', 'a', 'c', 'b'])
        >>> ci.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.categories
        Index(['c', 'b', 'a'], dtype='object')
        """
    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.ordered
        False

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.ordered
        True

        >>> cat = pd.Categorical(['a', 'b'], ordered=False)
        >>> cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b'], ordered=True)
        >>> ci.ordered
        True

        >>> ci = pd.CategoricalIndex(['a', 'b'], ordered=False)
        >>> ci.ordered
        False
        """
    @property
    def codes(self) -> np.ndarray:
        """
        The category codes of this categorical index.

        Codes are an array of integers which are the positions of the actual
        values in the categories array.

        There is no setter, use the other categorical methods and the normal item
        setter to change values in the categorical.

        Returns
        -------
        ndarray[int]
            A non-writable view of the ``codes`` array.

        Examples
        --------
        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.codes
        array([0, 1], dtype=int8)

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'])
        >>> ci.codes
        array([0, 1, 2, 0, 1, 2], dtype=int8)

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.codes
        array([2, 0], dtype=int8)
        """
    def _set_categories(self, categories, fastpath: bool = False) -> None:
        """
        Sets new categories inplace

        Parameters
        ----------
        fastpath : bool, default False
           Don't perform validation of the categories for uniqueness or nulls

        Examples
        --------
        >>> c = pd.Categorical(['a', 'b'])
        >>> c
        ['a', 'b']
        Categories (2, object): ['a', 'b']

        >>> c._set_categories(pd.Index(['a', 'c']))
        >>> c
        ['a', 'c']
        Categories (2, object): ['a', 'c']
        """
    def _set_dtype(self, dtype: CategoricalDtype) -> Self:
        """
        Internal method for directly updating the CategoricalDtype

        Parameters
        ----------
        dtype : CategoricalDtype

        Notes
        -----
        We don't do any validation here. It's assumed that the dtype is
        a (valid) instance of `CategoricalDtype`.
        """
    def set_ordered(self, value: bool) -> Self:
        """
        Set the ordered attribute to the boolean value.

        Parameters
        ----------
        value : bool
           Set whether this categorical is ordered (True) or not (False).
        """
    def as_ordered(self) -> Self:
        """
        Set the Categorical to be ordered.

        Returns
        -------
        Categorical
            Ordered Categorical.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.ordered
        False
        >>> ser = ser.cat.as_ordered()
        >>> ser.cat.ordered
        True

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'])
        >>> ci.ordered
        False
        >>> ci = ci.as_ordered()
        >>> ci.ordered
        True
        """
    def as_unordered(self) -> Self:
        """
        Set the Categorical to be unordered.

        Returns
        -------
        Categorical
            Unordered Categorical.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True
        >>> ser = ser.cat.as_unordered()
        >>> ser.cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'], ordered=True)
        >>> ci.ordered
        True
        >>> ci = ci.as_unordered()
        >>> ci.ordered
        False
        """
    def set_categories(self, new_categories, ordered: Incomplete | None = None, rename: bool = False):
        """
        Set the categories to the specified new categories.

        ``new_categories`` can include new categories (which will result in
        unused categories) or remove old categories (which results in values
        set to ``NaN``). If ``rename=True``, the categories will simply be renamed
        (less or more items than in old categories will result in values set to
        ``NaN`` or in unused categories respectively).

        This method can be used to perform more than one action of adding,
        removing, and reordering simultaneously and is therefore faster than
        performing the individual steps via the more specialised methods.

        On the other hand this methods does not do checks (e.g., whether the
        old categories are included in the new categories on a reorder), which
        can result in surprising changes, for example when using special string
        dtypes, which does not considers a S1 string equal to a single char
        python string.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, default False
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.
        rename : bool, default False
           Whether or not the new_categories should be considered as a rename
           of the old categories or as reordered categories.

        Returns
        -------
        Categorical with reordered categories.

        Raises
        ------
        ValueError
            If new_categories does not validate as categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'A'],
        ...                           categories=['a', 'b', 'c'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser
        0   a
        1   b
        2   c
        3   NaN
        dtype: category
        Categories (3, object): ['a' < 'b' < 'c']

        >>> ser.cat.set_categories(['A', 'B', 'C'], rename=True)
        0   A
        1   B
        2   C
        3   NaN
        dtype: category
        Categories (3, object): ['A' < 'B' < 'C']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'A'],
        ...                          categories=['a', 'b', 'c'], ordered=True)
        >>> ci
        CategoricalIndex(['a', 'b', 'c', nan], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')

        >>> ci.set_categories(['A', 'b', 'c'])
        CategoricalIndex([nan, 'b', 'c', nan], categories=['A', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> ci.set_categories(['A', 'b', 'c'], rename=True)
        CategoricalIndex(['A', 'b', 'c', nan], categories=['A', 'b', 'c'],
                         ordered=True, dtype='category')
        """
    def rename_categories(self, new_categories) -> Self:
        """
        Rename categories.

        Parameters
        ----------
        new_categories : list-like, dict-like or callable

            New categories which will replace old categories.

            * list-like: all items must be unique and the number of items in
              the new categories must match the existing number of categories.

            * dict-like: specifies a mapping from
              old categories to new. Categories not contained in the mapping
              are passed through and extra categories in the mapping are
              ignored.

            * callable : a callable that is called on all items in the old
              categories and whose return values comprise the new categories.

        Returns
        -------
        Categorical
            Categorical with renamed categories.

        Raises
        ------
        ValueError
            If new categories are list-like and do not have the same number of
            items than the current categories or do not validate as categories

        See Also
        --------
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'a', 'b'])
        >>> c.rename_categories([0, 1])
        [0, 0, 1]
        Categories (2, int64): [0, 1]

        For dict-like ``new_categories``, extra keys are ignored and
        categories not in the dictionary are passed through

        >>> c.rename_categories({'a': 'A', 'c': 'C'})
        ['A', 'A', 'b']
        Categories (2, object): ['A', 'b']

        You may also provide a callable to create the new categories

        >>> c.rename_categories(lambda x: x.upper())
        ['A', 'A', 'B']
        Categories (2, object): ['A', 'B']
        """
    def reorder_categories(self, new_categories, ordered: Incomplete | None = None) -> Self:
        """
        Reorder categories as specified in new_categories.

        ``new_categories`` need to include all old categories and no new category
        items.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, optional
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.

        Returns
        -------
        Categorical
            Categorical with reordered categories.

        Raises
        ------
        ValueError
            If the new categories do not contain all old category items or any
            new ones

        See Also
        --------
        rename_categories : Rename categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser = ser.cat.reorder_categories(['c', 'b', 'a'], ordered=True)
        >>> ser
        0   a
        1   b
        2   c
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        >>> ser.sort_values()
        2   c
        1   b
        0   a
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'])
        >>> ci
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'],
                         ordered=False, dtype='category')
        >>> ci.reorder_categories(['c', 'b', 'a'], ordered=True)
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'],
                         ordered=True, dtype='category')
        """
    def add_categories(self, new_categories) -> Self:
        """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
           The new categories to be included.

        Returns
        -------
        Categorical
            Categorical with new categories added.

        Raises
        ------
        ValueError
            If the new categories include old categories or do not validate as
            categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['c', 'b', 'c'])
        >>> c
        ['c', 'b', 'c']
        Categories (2, object): ['b', 'c']

        >>> c.add_categories(['d', 'a'])
        ['c', 'b', 'c']
        Categories (4, object): ['b', 'c', 'd', 'a']
        """
    def remove_categories(self, removals) -> Self:
        """
        Remove the specified categories.

        `removals` must be included in the old categories. Values which were in
        the removed categories will be set to NaN

        Parameters
        ----------
        removals : category or list of categories
           The categories which should be removed.

        Returns
        -------
        Categorical
            Categorical with removed categories.

        Raises
        ------
        ValueError
            If the removals are not contained in the categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'c', 'b', 'c', 'd'])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_categories(['d', 'a'])
        [NaN, 'c', 'b', 'c', NaN]
        Categories (2, object): ['b', 'c']
        """
    def remove_unused_categories(self) -> Self:
        """
        Remove categories which are not used.

        Returns
        -------
        Categorical
            Categorical with unused categories dropped.

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'c', 'b', 'c', 'd'])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c[2] = 'a'
        >>> c[4] = 'c'
        >>> c
        ['a', 'c', 'a', 'c', 'c']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_unused_categories()
        ['a', 'c', 'a', 'c', 'c']
        Categories (2, object): ['a', 'c']
        """
    def map(self, mapper, na_action: Literal['ignore'] | None | lib.NoDefault = ...):
        """
        Map categories using an input mapping or function.

        Maps the categories to new categories. If the mapping correspondence is
        one-to-one the result is a :class:`~pandas.Categorical` which has the
        same order property as the original, otherwise a :class:`~pandas.Index`
        is returned. NaN values are unaffected.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default 'ignore'
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

            .. deprecated:: 2.1.0

               The default value of 'ignore' has been deprecated and will be changed to
               None in the future.

        Returns
        -------
        pandas.Categorical or pandas.Index
            Mapped categorical.

        See Also
        --------
        CategoricalIndex.map : Apply a mapping correspondence on a
            :class:`~pandas.CategoricalIndex`.
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.map(lambda x: x.upper(), na_action=None)
        ['A', 'B', 'C']
        Categories (3, object): ['A', 'B', 'C']
        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'third'}, na_action=None)
        ['first', 'second', 'third']
        Categories (3, object): ['first', 'second', 'third']

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> cat = pd.Categorical(['a', 'b', 'c'], ordered=True)
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        >>> cat.map({'a': 3, 'b': 2, 'c': 1}, na_action=None)
        [3, 2, 1]
        Categories (3, int64): [3 < 2 < 1]

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'first'}, na_action=None)
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> cat.map({'a': 'first', 'b': 'second'}, na_action=None)
        Index(['first', 'second', nan], dtype='object')
        """
    __eq__: Incomplete
    __ne__: Incomplete
    __lt__: Incomplete
    __gt__: Incomplete
    __le__: Incomplete
    __ge__: Incomplete
    def _validate_setitem_value(self, value): ...
    def _validate_scalar(self, fill_value):
        """
        Convert a user-facing fill_value to a representation to use with our
        underlying ndarray, raising TypeError if this is not possible.

        Parameters
        ----------
        fill_value : object

        Returns
        -------
        fill_value : int

        Raises
        ------
        TypeError
        """
    @classmethod
    def _validate_codes_for_dtype(cls, codes, *, dtype: CategoricalDtype) -> np.ndarray: ...
    def __array__(self, dtype: NpDtype | None = None, copy: bool | None = None) -> np.ndarray:
        """
        The numpy array interface.

        Returns
        -------
        numpy.array
            A numpy array of either the specified dtype or,
            if dtype==None (default), the same dtype as
            categorical.categories.dtype.

        Examples
        --------

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)

        The following calls ``cat.__array__``

        >>> np.asarray(cat)
        array(['a', 'b'], dtype=object)
        """
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
    @property
    def nbytes(self) -> int: ...
    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Detect missing values

        Missing values (-1 in .codes) are detected.

        Returns
        -------
        np.ndarray[bool] of whether my values are null

        See Also
        --------
        isna : Top-level isna.
        isnull : Alias of isna.
        Categorical.notna : Boolean inverse of Categorical.isna.

        """
    isnull = isna
    def notna(self) -> npt.NDArray[np.bool_]:
        """
        Inverse of isna

        Both missing values (-1 in .codes) and NA as a category are detected as
        null.

        Returns
        -------
        np.ndarray[bool] of whether my values are not null

        See Also
        --------
        notna : Top-level notna.
        notnull : Alias of notna.
        Categorical.isna : Boolean inverse of Categorical.notna.

        """
    notnull = notna
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
    @classmethod
    def _empty(cls, shape: Shape, dtype: CategoricalDtype) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : CategoricalDtype
        """
    def _internal_get_values(self) -> ArrayLike:
        """
        Return the values.

        For internal compatibility with pandas formatting.

        Returns
        -------
        np.ndarray or ExtensionArray
            A numpy array or ExtensionArray of the same dtype as
            categorical.categories.dtype.
        """
    def check_for_ordered(self, op) -> None:
        """assert that we are ordered"""
    def argsort(self, *, ascending: bool = True, kind: SortKind = 'quicksort', **kwargs):
        """
        Return the indices that would sort the Categorical.

        Missing values are sorted at the end.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        **kwargs:
            passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort

        Notes
        -----
        While an ordering is applied to the category values, arg-sorting
        in this context refers more to organizing and grouping together
        based on matching category values. Thus, this function can be
        called on an unordered Categorical instance unlike the functions
        'Categorical.min' and 'Categorical.max'.

        Examples
        --------
        >>> pd.Categorical(['b', 'b', 'a', 'c']).argsort()
        array([2, 0, 1, 3])

        >>> cat = pd.Categorical(['b', 'b', 'a', 'c'],
        ...                      categories=['c', 'b', 'a'],
        ...                      ordered=True)
        >>> cat.argsort()
        array([3, 0, 1, 2])

        Missing values are placed at the end

        >>> cat = pd.Categorical([2, None, 1])
        >>> cat.argsort()
        array([2, 0, 1])
        """
    @overload
    def sort_values(self, *, inplace: Literal[False] = ..., ascending: bool = ..., na_position: str = ...) -> Self: ...
    @overload
    def sort_values(self, *, inplace: Literal[True], ascending: bool = ..., na_position: str = ...) -> None: ...
    def _rank(self, *, axis: AxisInt = 0, method: str = 'average', na_option: str = 'keep', ascending: bool = True, pct: bool = False):
        """
        See Series.rank.__doc__.
        """
    def _values_for_rank(self) -> np.ndarray:
        """
        For correctly ranking ordered categorical data. See GH#15420

        Ordered categorical data should be ranked on the basis of
        codes with -1 translated to NaN.

        Returns
        -------
        numpy.array

        """
    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> npt.NDArray[np.uint64]:
        """
        Hash a Categorical by hashing its categories, and then mapping the codes
        to the hashes.

        Parameters
        ----------
        encoding : str
        hash_key : str
        categorize : bool
            Ignored for Categorical.

        Returns
        -------
        np.ndarray[uint64]
        """
    @property
    def _codes(self) -> np.ndarray: ...
    def _box_func(self, i: int): ...
    def _unbox_scalar(self, key) -> int: ...
    def __iter__(self) -> Iterator:
        """
        Returns an Iterator over the values of this Categorical.
        """
    def __contains__(self, key) -> bool:
        """
        Returns True if `key` is in this Categorical.
        """
    def _formatter(self, boxed: bool = False): ...
    def _repr_categories(self) -> list[str]:
        """
        return the base repr for the categories
        """
    def _get_repr_footer(self) -> str:
        """
        Returns a string representation of the footer.
        """
    def _get_values_repr(self) -> str: ...
    def __repr__(self) -> str:
        """
        String representation.
        """
    def _validate_listlike(self, value): ...
    def _reverse_indexer(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Compute the inverse of a categorical, returning
        a dict of categories -> indexers.

        *This is an internal function*

        Returns
        -------
        Dict[Hashable, np.ndarray[np.intp]]
            dict of categories -> indexers

        Examples
        --------
        >>> c = pd.Categorical(list('aabca'))
        >>> c
        ['a', 'a', 'b', 'c', 'a']
        Categories (3, object): ['a', 'b', 'c']
        >>> c.categories
        Index(['a', 'b', 'c'], dtype='object')
        >>> c.codes
        array([0, 0, 1, 2, 0], dtype=int8)
        >>> c._reverse_indexer()
        {'a': array([0, 1, 4]), 'b': array([2]), 'c': array([3])}

        """
    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs): ...
    def min(self, *, skipna: bool = True, **kwargs):
        """
        The minimum value of the object.

        Only ordered `Categoricals` have a minimum!

        Raises
        ------
        TypeError
            If the `Categorical` is not `ordered`.

        Returns
        -------
        min : the minimum of this `Categorical`, NA value if empty
        """
    def max(self, *, skipna: bool = True, **kwargs):
        """
        The maximum value of the object.

        Only ordered `Categoricals` have a maximum!

        Raises
        ------
        TypeError
            If the `Categorical` is not `ordered`.

        Returns
        -------
        max : the maximum of this `Categorical`, NA if array is empty
        """
    def _mode(self, dropna: bool = True) -> Categorical: ...
    def unique(self) -> Self:
        '''
        Return the ``Categorical`` which ``categories`` and ``codes`` are
        unique.

        .. versionchanged:: 1.3.0

            Previously, unused categories were dropped from the new categories.

        Returns
        -------
        Categorical

        See Also
        --------
        pandas.unique
        CategoricalIndex.unique
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> pd.Categorical(list("baabc")).unique()
        [\'b\', \'a\', \'c\']
        Categories (3, object): [\'a\', \'b\', \'c\']
        >>> pd.Categorical(list("baab"), categories=list("abc"), ordered=True).unique()
        [\'b\', \'a\']
        Categories (3, object): [\'a\' < \'b\' < \'c\']
        '''
    def _cast_quantile_result(self, res_values: np.ndarray) -> np.ndarray: ...
    def equals(self, other: object) -> bool:
        """
        Returns True if categorical arrays are equal.

        Parameters
        ----------
        other : `Categorical`

        Returns
        -------
        bool
        """
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = 0) -> Self: ...
    def _encode_with_my_categories(self, other: Categorical) -> Categorical:
        """
        Re-encode another categorical using this Categorical's categories.

        Notes
        -----
        This assumes we have already checked
        self._categories_match_up_to_permutation(other).
        """
    def _categories_match_up_to_permutation(self, other: Categorical) -> bool:
        """
        Returns True if categoricals are the same dtype
          same categories, and same ordered

        Parameters
        ----------
        other : Categorical

        Returns
        -------
        bool
        """
    def describe(self) -> DataFrame:
        """
        Describes this Categorical

        Returns
        -------
        description: `DataFrame`
            A dataframe with frequency and counts by category.
        """
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Check whether `values` are contained in Categorical.

        Return a boolean NumPy Array showing whether each element in
        the Categorical matches an element in the passed sequence of
        `values` exactly.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        np.ndarray[bool]

        Raises
        ------
        TypeError
          * If `values` is not a set or list-like

        See Also
        --------
        pandas.Series.isin : Equivalent method on Series.

        Examples
        --------
        >>> s = pd.Categorical(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'])
        >>> s.isin(['cow', 'lama'])
        array([ True,  True,  True, False,  True, False])

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        array([ True, False,  True, False,  True, False])
        """
    def _replace(self, *, to_replace, value, inplace: bool = False): ...
    def _str_map(self, f, na_value=..., dtype=..., convert: bool = True): ...
    def _str_get_dummies(self, sep: str = '|'): ...
    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs): ...

class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    '''
    Accessor object for categorical properties of the Series values.

    Parameters
    ----------
    data : Series or CategoricalIndex

    Examples
    --------
    >>> s = pd.Series(list("abbccc")).astype("category")
    >>> s
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): [\'a\', \'b\', \'c\']

    >>> s.cat.categories
    Index([\'a\', \'b\', \'c\'], dtype=\'object\')

    >>> s.cat.rename_categories(list("cba"))
    0    c
    1    b
    2    b
    3    a
    4    a
    5    a
    dtype: category
    Categories (3, object): [\'c\', \'b\', \'a\']

    >>> s.cat.reorder_categories(list("cba"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): [\'c\', \'b\', \'a\']

    >>> s.cat.add_categories(["d", "e"])
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): [\'a\', \'b\', \'c\', \'d\', \'e\']

    >>> s.cat.remove_categories(["a", "c"])
    0    NaN
    1      b
    2      b
    3    NaN
    4    NaN
    5    NaN
    dtype: category
    Categories (1, object): [\'b\']

    >>> s1 = s.cat.add_categories(["d", "e"])
    >>> s1.cat.remove_unused_categories()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): [\'a\', \'b\', \'c\']

    >>> s.cat.set_categories(list("abcde"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): [\'a\', \'b\', \'c\', \'d\', \'e\']

    >>> s.cat.as_ordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): [\'a\' < \'b\' < \'c\']

    >>> s.cat.as_unordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): [\'a\', \'b\', \'c\']
    '''
    _parent: Incomplete
    _index: Incomplete
    _name: Incomplete
    def __init__(self, data) -> None: ...
    @staticmethod
    def _validate(data) -> None: ...
    def _delegate_property_get(self, name: str): ...
    def _delegate_property_set(self, name: str, new_values): ...
    @property
    def codes(self) -> Series:
        '''
        Return Series of codes as well as the index.

        Examples
        --------
        >>> raw_cate = pd.Categorical(["a", "b", "c", "a"], categories=["a", "b"])
        >>> ser = pd.Series(raw_cate)
        >>> ser.cat.codes
        0   0
        1   1
        2  -1
        3   0
        dtype: int8
        '''
    def _delegate_method(self, name: str, *args, **kwargs): ...

def _get_codes_for_values(values: Index | Series | ExtensionArray | np.ndarray, categories: Index) -> np.ndarray:
    """
    utility routine to turn values into codes given the specified categories

    If `values` is known to be a Categorical, use recode_for_categories instead.
    """
def recode_for_categories(codes: np.ndarray, old_categories, new_categories, copy: bool = True) -> np.ndarray:
    """
    Convert a set of codes for to a new set of categories

    Parameters
    ----------
    codes : np.ndarray
    old_categories, new_categories : Index
    copy: bool, default True
        Whether to copy if the codes are unchanged.

    Returns
    -------
    new_codes : np.ndarray[np.int64]

    Examples
    --------
    >>> old_cat = pd.Index(['b', 'a', 'c'])
    >>> new_cat = pd.Index(['a', 'b'])
    >>> codes = np.array([0, 1, 1, 2])
    >>> recode_for_categories(codes, old_cat, new_cat)
    array([ 1,  0,  0, -1], dtype=int8)
    """
def factorize_from_iterable(values) -> tuple[np.ndarray, Index]:
    """
    Factorize an input `values` into `categories` and `codes`. Preserves
    categorical dtype in `categories`.

    Parameters
    ----------
    values : list-like

    Returns
    -------
    codes : ndarray
    categories : Index
        If `values` has a categorical dtype, then `categories` is
        a CategoricalIndex keeping the categories and order of `values`.
    """
def factorize_from_iterables(iterables) -> tuple[list[np.ndarray], list[Index]]:
    """
    A higher-level wrapper over `factorize_from_iterable`.

    Parameters
    ----------
    iterables : list-like of list-likes

    Returns
    -------
    codes : list of ndarrays
    categories : list of Indexes

    Notes
    -----
    See `factorize_from_iterable` for more info.
    """

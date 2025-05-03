import _cython_3_0_11
from _typeshed import Incomplete
from typing import ClassVar

SIZE_HINT_LIMIT: int
__pyx_unpickle_HashTable: _cython_3_0_11.cython_function_or_method
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
_unique_label_indices_int32: _cython_3_0_11.cython_function_or_method
_unique_label_indices_int64: _cython_3_0_11.cython_function_or_method
duplicated: _cython_3_0_11.fused_cython_function
get_hashtable_trace_domain: _cython_3_0_11.cython_function_or_method
ismember: _cython_3_0_11.fused_cython_function
mode: _cython_3_0_11.fused_cython_function
object_hash: _cython_3_0_11.cython_function_or_method
objects_are_equal: _cython_3_0_11.cython_function_or_method
unique_label_indices: _cython_3_0_11.cython_function_or_method
value_count: _cython_3_0_11.fused_cython_function

class Complex128Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Complex128Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="complex128"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Complex128HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[complex128]
            Array of values of which unique will be calculated
        uniques : Complex128Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[complex128]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[complex128]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[complex128]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[complex128]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[complex128]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Complex128Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Complex64Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Complex64Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="complex64"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Complex64HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[complex64]
            Array of values of which unique will be calculated
        uniques : Complex64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[complex64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[complex64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[complex64]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[complex64]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[complex64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Complex64Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Factorizer:
    count: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs): ...
    def get_count(self, *args, **kwargs): ...
    def __reduce__(self): ...

class Float32Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Float32Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="float32"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Float32HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[float32]
            Array of values of which unique will be calculated
        uniques : Float32Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[float32]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[float32]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[float32]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[float32]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[float32]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Float32Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Float64Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Float64Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="float64"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Float64HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[float64]
            Array of values of which unique will be calculated
        uniques : Float64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[float64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[float64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[float64]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[float64]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[float64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Float64Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class HashTable:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class Int16Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Int16Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="int16"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Int16HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int16]
            Array of values of which unique will be calculated
        uniques : Int16Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[int16]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[int16]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[int16]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int16]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[int16]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int16Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int32Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Int32Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="int32"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Int32HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int32]
            Array of values of which unique will be calculated
        uniques : Int32Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[int32]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[int32]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[int32]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int32]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[int32]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int32Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int64Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Int64Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="int64"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Int64HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        uniques : Int64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_labels_groupby(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_keys_to_values(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int64Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int8Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = Int8Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="int8"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class Int8HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int8]
            Array of values of which unique will be calculated
        uniques : Int8Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[int8]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[int8]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[int8]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int8]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[int8]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Int8Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class IntpHashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        uniques : Int64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_labels_groupby(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_keys_to_values(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class ObjectFactorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        """

        Returns
        -------
        np.ndarray[np.intp]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = ObjectFactorizer(3)
        >>> fac.factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
    def __reduce__(self): ...

class ObjectVector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class PyObjectHashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        uniques : ObjectVector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then None _plus_
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then None _plus_
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            Not yet implemented for PyObjectHashTable.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs): ...
    def get_labels(self, *args, **kwargs): ...
    def get_state(self, *args, **kwargs):
        """
        returns infos about the current state of the hashtable like size,
        number of buckets and so on.
        """
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            Not yet implemented for PyObjectHashTable

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class StringHashTable(HashTable):
    na_string_sentinel: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        uniques : ObjectVector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then any value
            that is not a string is considered missing. If na_value is
            not None, then _additionally_ any value "val" satisfying
            val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then any value
            that is not a string is considered missing. If na_value is
            not None, then _additionally_ any value "val" satisfying
            val == na_value is considered missing.
        mask : ndarray[bool], optional
            Not yet implemented for StringHashTable.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[intp]
            The labels from values to uniques
        '''
    def get_indexer(self, *args, **kwargs): ...
    def get_item(self, *args, **kwargs): ...
    def get_labels(self, *args, **kwargs): ...
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs): ...
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            Not yet implemented for StringHashTable

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        """
    def __reduce__(self): ...

class StringVector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt16Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = UInt16Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="uint16"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class UInt16HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint16]
            Array of values of which unique will be calculated
        uniques : UInt16Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[uint16]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[uint16]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[uint16]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint16]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[uint16]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt16Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt32Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = UInt32Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="uint32"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class UInt32HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint32]
            Array of values of which unique will be calculated
        uniques : UInt32Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[uint32]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[uint32]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[uint32]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint32]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[uint32]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt32Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt64Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = UInt64Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="uint64"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class UInt64HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint64]
            Array of values of which unique will be calculated
        uniques : UInt64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[uint64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[uint64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[uint64]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint64]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[uint64]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt64Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt8Factorizer(Factorizer):
    table: Incomplete
    uniques: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def factorize(self, *args, **kwargs):
        '''
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = UInt8Factorizer(3)
        >>> fac.factorize(np.array([1,2,3], dtype="uint8"), na_sentinel=20)
        array([0, 1, 2])
        '''
    def __reduce__(self): ...

class UInt8HashTable(HashTable):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _unique(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint8]
            Array of values of which unique will be calculated
        uniques : UInt8Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : bool, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        use_result_mask: bool, default False
            Whether to create a result mask for the unique values. Not supported
            with return_inverse=True.

        Returns
        -------
        uniques : ndarray[uint8]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse=True)
            The labels from values to uniques
        result_mask: ndarray[bool], if use_result_mask is true
            The mask for the result values.
        '''
    def factorize(self, *args, **kwargs):
        '''
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[uint8]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or
            condition "val != val".

        Returns
        -------
        uniques : ndarray[uint8]
            Unique values of input, not sorted
        labels : ndarray[intp_t]
            The labels from values to uniques
        '''
    def get_item(self, *args, **kwargs):
        """Extracts the position of val from the hashtable.

                Parameters
                ----------
                val : Scalar
                    The value that is looked up in the hashtable

                Returns
                -------
                The position of the requested integer.
        """
    def get_labels(self, *args, **kwargs): ...
    def get_na(self, *args, **kwargs):
        """Extracts the position of na_value from the hashtable.

                Returns
                -------
                The position of the last na value.
        """
    def get_state(self, *args, **kwargs):
        """ returns infos about the state of the hashtable"""
    def lookup(self, *args, **kwargs): ...
    def map_locations(self, *args, **kwargs): ...
    def set_item(self, *args, **kwargs): ...
    def set_na(self, *args, **kwargs): ...
    def sizeof(self, *args, **kwargs):
        """ return the size of my table in bytes """
    def unique(self, *args, **kwargs):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint8]
            Array of values of which unique will be calculated
        return_inverse : bool, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.
        mask : ndarray[bool], optional
            If not None, the mask is used as indicator for missing values
            (True = missing, False = valid) instead of `na_value` or

        Returns
        -------
        uniques : ndarray[uint8]
            Unique values of input, not sorted
        labels : ndarray[intp_t] (if return_inverse)
            The labels from values to uniques
        result_mask: ndarray[bool], if mask is given as input
            The mask for the result values.
        """
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class UInt8Vector(Vector):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def to_array(self, *args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class Vector:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

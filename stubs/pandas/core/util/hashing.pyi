import np
import npt
from pandas._libs.hashing import hash_object_array as hash_object_array
from pandas._libs.lib import is_list_like as is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCExtensionArray as ABCExtensionArray, ABCIndex as ABCIndex, ABCMultiIndex as ABCMultiIndex, ABCSeries as ABCSeries

TYPE_CHECKING: bool
_default_hash_key: str
def combine_hash_arrays(arrays: Iterator[np.ndarray], num_items: int) -> npt.NDArray[np.uint64]:
    """
    Parameters
    ----------
    arrays : Iterator[np.ndarray]
    num_items : int

    Returns
    -------
    np.ndarray[uint64]

    Should be the same as CPython's tupleobject.c
    """
def hash_pandas_object(obj: Index | DataFrame | Series, index: bool = ..., encoding: str = ..., hash_key: str | None = ..., categorize: bool = ...) -> Series:
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    obj : Index, Series, or DataFrame
    index : bool, default True
        Include the index in the hash (if Series/DataFrame).
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    Series of uint64, same length as the object

    Examples
    --------
    >>> pd.util.hash_pandas_object(pd.Series([1, 2, 3]))
    0    14639053686158035780
    1     3869563279212530728
    2      393322362522515241
    dtype: uint64
    """
def hash_tuples(vals: MultiIndex | Iterable[tuple[Hashable, ...]], encoding: str = ..., hash_key: str = ...) -> npt.NDArray[np.uint64]:
    """
    Hash an MultiIndex / listlike-of-tuples efficiently.

    Parameters
    ----------
    vals : MultiIndex or listlike-of-tuples
    encoding : str, default 'utf8'
    hash_key : str, default _default_hash_key

    Returns
    -------
    ndarray[np.uint64] of hashed values
    """
def hash_array(vals: ArrayLike, encoding: str = ..., hash_key: str = ..., categorize: bool = ...) -> npt.NDArray[np.uint64]:
    """
    Given a 1d array, return an array of deterministic integers.

    Parameters
    ----------
    vals : ndarray or ExtensionArray
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    ndarray[np.uint64, ndim=1]
        Hashed values, same length as the vals.

    Examples
    --------
    >>> pd.util.hash_array(np.array([1, 2, 3]))
    array([ 6238072747940578789, 15839785061582574730,  2185194620014831856],
      dtype=uint64)
    """
def _hash_ndarray(vals: np.ndarray, encoding: str = ..., hash_key: str = ..., categorize: bool = ...) -> npt.NDArray[np.uint64]:
    """
    See hash_array.__doc__.
    """

from pandas._typing import ArrayLike as ArrayLike
from typing import Any

def should_extension_dispatch(left: ArrayLike, right: Any) -> bool:
    """
    Identify cases where Series operation should dispatch to ExtensionArray method.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object

    Returns
    -------
    bool
    """

from pandas.core.dtypes.generic import ABCExtensionArray as ABCExtensionArray
from typing import Any

TYPE_CHECKING: bool
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

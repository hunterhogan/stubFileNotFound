from _typeshed import Incomplete
from typing import Any

_address_pattern: Incomplete
_contained_address_pattern: Incomplete

def _get_parent_and_dict_from_namespace(namespace: Any) -> Any:
    """
    Extract the parent object and `dict` from `namespace`.

    For the `namespace`, the user can give either a parent object
    (`getattr(namespace, address) is obj`) or a `dict`-like namespace
    (`namespace[address] is obj`).

    Returns `(parent_object, namespace_dict)`.
    """
def is_address(string: Any) -> Any: ...




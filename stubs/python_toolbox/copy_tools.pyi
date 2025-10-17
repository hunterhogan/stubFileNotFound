from typing import Any

def deepcopy_as_simple_object(thing: Any, memo: Any=None) -> Any:
    """Deepcopy an object as a simple `object`, ignoring any __deepcopy__ method."""




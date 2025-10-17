
from typing import Any

class FunctionAnchoringType(type):
    r"""
    Metaclass for working around Python\'s problems with pickling functions.

    Python has a hard time pickling functions that are not at module level,
    because when unpickling them, Python looks for them only on the module
    level.

    What we do in this function is create a reference to each of the class\'s
    functions on the module level. We call this "anchoring." Note that we\'re
    only anchoring the *functions*, not the *methods*. Methods *can* be pickled
    by Python, but plain functions, like those created by `staticmethod`,
    cannot.

    This workaround is hacky, yes, but it seems like the best solution until
    Python learns how to pickle non-module-level functions.
    """

    def __new__(mcls: Any, name: Any, bases: Any, namespace_dict: Any) -> Any: ...




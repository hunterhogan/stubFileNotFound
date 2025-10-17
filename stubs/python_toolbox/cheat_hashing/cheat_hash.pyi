from .cheat_hash_functions import (
	cheat_hash_dict as cheat_hash_dict, cheat_hash_object as cheat_hash_object, cheat_hash_sequence as cheat_hash_sequence,
	cheat_hash_set as cheat_hash_set)
from _typeshed import Incomplete
from typing import Any

infinity: Incomplete
dispatch_map: Incomplete

def cheat_hash(thing: Any) -> Any:
    """
    Cheat-hash an object. Works on mutable objects.

    This is a replacement for `hash` which generates something like an hash for
    an object, even if it is mutable, unhashable and/or refers to
    mutable/unhashable objects.

    This is intended for situtations where you have mutable objects that you
    never modify, and you want to be able to hash them despite Python not
    letting you.
    """




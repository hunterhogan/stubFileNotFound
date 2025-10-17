from _typeshed import Incomplete
from python_toolbox import binary_search as binary_search, decorator_tools as decorator_tools, misc_tools as misc_tools
from typing import Any

infinity: Incomplete

class CLEAR_ENTIRE_CACHE(misc_tools.NonInstantiable):
    """Sentinel object for clearing the entire cache."""

def _get_now() -> Any:
    """
    Get the current datetime.

    This is specified as a function to make testing easier.
    """
@decorator_tools.helpful_decorator_builder
def cache(max_size: Any=..., time_to_keep: Any=None) -> Any:
    """
    Cache a function, saving results so they won't have to be computed again.

    This decorator understands function arguments. For example, it understands
    that for a function like this:

        @cache()
        def f(a, b=2):
            return whatever

    The calls `f(1)` or `f(1, 2)` or `f(b=2, a=1)` are all identical, and a
    cached result saved for one of these calls will be used for the others.

    All the arguments are sleekreffed to prevent memory leaks. Sleekref is a
    variation of weakref. Sleekref is when you try to weakref an object, but if
    it's non-weakreffable, like a `list` or a `dict`, you maintain a normal,
    strong reference to it. (See documentation of
    `python_toolbox.sleek_reffing` for more details.) Thanks to sleekreffing
    you can avoid memory leaks when using weakreffable arguments, but if you
    ever want to use non-weakreffable arguments you are still able to.
    (Assuming you don't mind the memory leaks.)

    You may optionally specify a `max_size` for maximum number of cached
    results to store; old entries are thrown away according to a
    least-recently-used alogrithm. (Often abbreivated LRU.)

    You may optionally specific a `time_to_keep`, which is a time period after
    which a cache entry will expire. (Pass in either a `timedelta` object or
    keyword arguments to create one.)
    """




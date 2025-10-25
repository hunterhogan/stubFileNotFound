from typing import Any
__all__ = ['re_iter', 're_all', 're_find', 're_finder', 're_test', 're_tester', 'str_join', 'cut_prefix', 'cut_suffix']

def re_iter(regex: Any, s: Any, flags: int = 0) -> Any:
    """Iterates over matches of regex in s, presents them in simplest possible form"""
def re_all(regex: Any, s: Any, flags: int = 0) -> Any:
    """Lists all matches of regex in s, presents them in simplest possible form"""
def re_find(regex: Any, s: Any, flags: int = 0) -> Any:
    """Matches regex against the given string,
       returns the match in the simplest possible form."""
def re_test(regex: Any, s: Any, flags: int = 0) -> Any:
    """Tests whether regex matches against s."""
def re_finder(regex: Any, flags: int = 0) -> Any:
    """Creates a function finding regex in passed string."""
def re_tester(regex: Any, flags: int = 0) -> Any:
    """Creates a predicate testing passed string with regex."""
def str_join(sep: Any, seq: Any=...) -> Any:
    """Joins the given sequence with sep.
       Forces stringification of seq items."""
def cut_prefix(s: Any, prefix: Any) -> Any:
    """Cuts prefix from given string if it's present."""
def cut_suffix(s: Any, suffix: Any) -> Any:
    """Cuts suffix from given string if it's present."""

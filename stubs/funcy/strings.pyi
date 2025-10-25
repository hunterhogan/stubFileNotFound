__all__ = ['re_iter', 're_all', 're_find', 're_finder', 're_test', 're_tester', 'str_join', 'cut_prefix', 'cut_suffix']

def re_iter(regex, s, flags: int = 0):
    """Iterates over matches of regex in s, presents them in simplest possible form"""
def re_all(regex, s, flags: int = 0):
    """Lists all matches of regex in s, presents them in simplest possible form"""
def re_find(regex, s, flags: int = 0):
    """Matches regex against the given string,
       returns the match in the simplest possible form."""
def re_test(regex, s, flags: int = 0):
    """Tests whether regex matches against s."""
def re_finder(regex, flags: int = 0):
    """Creates a function finding regex in passed string."""
def re_tester(regex, flags: int = 0):
    """Creates a predicate testing passed string with regex."""
def str_join(sep, seq=...):
    """Joins the given sequence with sep.
       Forces stringification of seq items."""
def cut_prefix(s, prefix):
    """Cuts prefix from given string if it's present."""
def cut_suffix(s, suffix):
    """Cuts suffix from given string if it's present."""

from _typeshed import Incomplete
from collections.abc import Iterable, Mapping, Sequence
from pandas._config import get_option as get_option
from pandas.core.dtypes.inference import is_sequence as is_sequence
from pandas.io.formats.console import get_console_size as get_console_size
from typing import Any, TypeVar

from collections.abc import Callable

EscapeChars = Mapping[str, str] | Iterable[str]
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

def adjoin(space: int, *lists: list[str], **kwargs) -> str:
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    ----------
    space : int
        number of spaces for padding
    lists : str
        list of str which being joined
    strlen : callable
        function used to calculate the length of each str. Needed for unicode
        handling.
    justfunc : callable
        function used to justify str. Needed for unicode handling.
    """
def _adj_justify(texts: Iterable[str], max_len: int, mode: str = 'right') -> list[str]:
    """
    Perform ljust, center, rjust against string or list-like
    """
def _pprint_seq(seq: Sequence, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.

    bounds length of printed sequence, depending on options
    """
def _pprint_dict(seq: Mapping, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.
    """
def pprint_thing(thing: Any, _nest_lvl: int = 0, escape_chars: EscapeChars | None = None, default_escapes: bool = False, quote_strings: bool = False, max_seq_items: int | None = None) -> str:
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list or dict, optional
        Characters to escape. If a dict is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """
def pprint_thing_encoded(object, encoding: str = 'utf-8', errors: str = 'replace') -> bytes: ...
def enable_data_resource_formatter(enable: bool) -> None: ...
def default_pprint(thing: Any, max_seq_items: int | None = None) -> str: ...
def format_object_summary(obj, formatter: Callable, is_justify: bool = True, name: str | None = None, indent_for_name: bool = True, line_break_each_value: bool = False) -> str:
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

    Returns
    -------
    summary string
    """
def _justify(head: list[Sequence[str]], tail: list[Sequence[str]]) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
    tail : list-like of list-likes of strings

    Returns
    -------
    tuple of list of tuples of strings
        Same as head and tail, but items are right aligned when stacked
        vertically.

    Examples
    --------
    >>> _justify([['a', 'b']], [['abc', 'abcd']])
    ([('  a', '   b')], [('abc', 'abcd')])
    """

class PrettyDict(dict[_KT, _VT]):
    """Dict extension to support abbreviated __repr__"""
    def __repr__(self) -> str: ...

class _TextAdjustment:
    encoding: Incomplete
    def __init__(self) -> None: ...
    def len(self, text: str) -> int: ...
    def justify(self, texts: Any, max_len: int, mode: str = 'right') -> list[str]:
        """
        Perform ljust, center, rjust against string or list-like
        """
    def adjoin(self, space: int, *lists, **kwargs) -> str: ...

class _EastAsianTextAdjustment(_TextAdjustment):
    ambiguous_width: int
    _EAW_MAP: Incomplete
    def __init__(self) -> None: ...
    def len(self, text: str) -> int:
        """
        Calculate display width considering unicode East Asian Width
        """
    def justify(self, texts: Iterable[str], max_len: int, mode: str = 'right') -> list[str]: ...

def get_adjustment() -> _TextAdjustment: ...

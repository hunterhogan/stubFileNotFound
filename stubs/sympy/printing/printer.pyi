import inspect
from _typeshed import Incomplete
from collections.abc import Generator
from sympy.core.function import AppliedUndef as AppliedUndef, Function as Function, UndefinedFunction as UndefinedFunction
from typing import Any

def printer_context(printer, **kwargs) -> Generator[None]: ...

class Printer:
    """ Generic printer

    Its job is to provide infrastructure for implementing new printers easily.

    If you want to define your custom Printer or your custom printing method
    for your custom class then see the example above: printer_example_ .
    """
    _global_settings: dict[str, Any]
    _default_settings: dict[str, Any]
    printmethod: str
    @classmethod
    def _get_initial_settings(cls): ...
    _str: Incomplete
    _settings: Incomplete
    _context: Incomplete
    _print_level: int
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    @classmethod
    def set_global_settings(cls, **settings) -> None:
        """Set system-wide printing settings. """
    @property
    def order(self): ...
    def doprint(self, expr):
        """Returns printer's representation for expr (as a string)"""
    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
    def emptyPrinter(self, expr): ...
    def _as_ordered_terms(self, expr, order: Incomplete | None = None):
        """A compatibility function for ordering terms in Add. """

class _PrintFunction:
    """
    Function wrapper to replace ``**settings`` in the signature with printer defaults
    """
    __other_params: Incomplete
    __print_cls: Incomplete
    def __init__(self, f, print_cls: type[Printer]) -> None: ...
    def __reduce__(self): ...
    def __call__(self, *args, **kwargs): ...
    @property
    def __signature__(self) -> inspect.Signature: ...

def print_function(print_cls):
    """ A decorator to replace kwargs with the printer settings in __signature__ """

from .c import C89CodePrinter as C89CodePrinter, C99CodePrinter as C99CodePrinter
from .codeprinter import requires as requires
from _typeshed import Incomplete
from sympy.codegen.ast import Type as Type, none as none
from sympy.printing.codeprinter import cxxcode as cxxcode

reserved: Incomplete
_math_functions: Incomplete

def _attach_print_method(cls, sympy_name, func_name): ...
def _attach_print_methods(cls, cont) -> None: ...

class _CXXCodePrinterBase:
    printmethod: str
    language: str
    _ns: str
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    def _print_Max(self, expr): ...
    def _print_Min(self, expr): ...
    def _print_using(self, expr): ...
    def _print_Raise(self, rs): ...
    def _print_RuntimeError_(self, re): ...

class CXX98CodePrinter(_CXXCodePrinterBase, C89CodePrinter):
    standard: str
    reserved_words: Incomplete

class CXX11CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard: str
    reserved_words: Incomplete
    type_mappings: Incomplete
    def _print_using(self, expr): ...

class CXX17CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard: str
    reserved_words: Incomplete
    _kf: Incomplete
    def _print_beta(self, expr): ...
    def _print_Ei(self, expr): ...
    def _print_zeta(self, expr): ...

cxx_code_printers: Incomplete

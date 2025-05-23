from _typeshed import Incomplete
from sympy.codegen.ast import Assignment as Assignment, Declaration as Declaration, Pointer as Pointer, bool_ as bool_, complex128 as complex128, complex64 as complex64, complex_ as complex_, float32 as float32, float64 as float64, float80 as float80, int16 as int16, int32 as int32, int64 as int64, int8 as int8, intc as intc, integer as integer, none as none, real as real, stderr as stderr, stdout as stdout, value_const as value_const
from sympy.codegen.fnodes import allocatable as allocatable, cmplx as cmplx, dsign as dsign, elemental as elemental, intent_in as intent_in, intent_inout as intent_inout, intent_out as intent_out, isign as isign, literal_dp as literal_dp, merge as merge, pure as pure
from sympy.core import Add as Add, Float as Float, N as N, S as S, Symbol as Symbol
from sympy.core.function import Function as Function
from sympy.core.numbers import equal_valued as equal_valued
from sympy.core.relational import Eq as Eq
from sympy.printing.codeprinter import CodePrinter as CodePrinter, fcode as fcode, print_fcode as print_fcode
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE, precedence as precedence
from sympy.printing.printer import printer_context as printer_context
from sympy.sets import Range as Range
from typing import Any

known_functions: Incomplete

class FCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Fortran code"""
    printmethod: str
    language: str
    type_aliases: Incomplete
    type_mappings: Incomplete
    type_modules: Incomplete
    _default_settings: dict[str, Any]
    _operators: Incomplete
    _relationals: Incomplete
    mangled_symbols: Incomplete
    used_name: Incomplete
    known_functions: Incomplete
    module_uses: Incomplete
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    @property
    def _lead(self): ...
    def _print_Symbol(self, expr): ...
    def _rate_index_position(self, p): ...
    def _get_statement(self, codestring): ...
    def _get_comment(self, text): ...
    def _declare_number_const(self, name, value): ...
    def _print_NumberSymbol(self, expr): ...
    def _format_code(self, lines): ...
    def _traverse_matrix_indices(self, mat): ...
    def _get_loop_opening_ending(self, indices): ...
    def _print_sign(self, expr): ...
    def _print_Piecewise(self, expr): ...
    def _print_MatrixElement(self, expr): ...
    def _print_Add(self, expr): ...
    def _print_Function(self, expr): ...
    def _print_Mod(self, expr): ...
    def _print_ImaginaryUnit(self, expr): ...
    def _print_int(self, expr): ...
    def _print_Mul(self, expr): ...
    def _print_Pow(self, expr): ...
    def _print_Rational(self, expr): ...
    def _print_Float(self, expr): ...
    def _print_Relational(self, expr): ...
    def _print_Indexed(self, expr): ...
    def _print_Idx(self, expr): ...
    def _print_AugmentedAssignment(self, expr): ...
    def _print_sum_(self, sm): ...
    def _print_product_(self, prod): ...
    def _print_Do(self, do): ...
    def _print_ImpliedDoLoop(self, idl): ...
    def _print_For(self, expr): ...
    def _print_Type(self, type_): ...
    def _print_Element(self, elem): ...
    def _print_Extent(self, ext): ...
    def _print_Declaration(self, expr): ...
    def _print_Infinity(self, expr): ...
    def _print_While(self, expr): ...
    def _print_BooleanTrue(self, expr): ...
    def _print_BooleanFalse(self, expr): ...
    def _pad_leading_columns(self, lines): ...
    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
    def _print_GoTo(self, goto): ...
    def _print_Program(self, prog): ...
    def _print_Module(self, mod): ...
    def _print_Stream(self, strm): ...
    def _print_Print(self, ps): ...
    def _print_Return(self, rs): ...
    def _print_FortranReturn(self, frs): ...
    def _head(self, entity, fp, **kwargs): ...
    def _print_FunctionPrototype(self, fp): ...
    def _print_FunctionDefinition(self, fd): ...
    def _print_Subroutine(self, sub): ...
    def _print_SubroutineCall(self, scall): ...
    def _print_use_rename(self, rnm): ...
    def _print_use(self, use): ...
    def _print_BreakToken(self, _): ...
    def _print_ContinueToken(self, _): ...
    def _print_ArrayConstructor(self, ac): ...
    def _print_ArrayElement(self, elem): ...

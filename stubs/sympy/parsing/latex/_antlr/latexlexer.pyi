from antlr4 import *
from _typeshed import Incomplete
from io import StringIO as StringIO
from typing import TextIO

def serializedATN(): ...

class LaTeXLexer(Lexer):
    atn: Incomplete
    decisionsToDFA: Incomplete
    T__0: int
    T__1: int
    WS: int
    THINSPACE: int
    MEDSPACE: int
    THICKSPACE: int
    QUAD: int
    QQUAD: int
    NEGTHINSPACE: int
    NEGMEDSPACE: int
    NEGTHICKSPACE: int
    CMD_LEFT: int
    CMD_RIGHT: int
    IGNORE: int
    ADD: int
    SUB: int
    MUL: int
    DIV: int
    L_PAREN: int
    R_PAREN: int
    L_BRACE: int
    R_BRACE: int
    L_BRACE_LITERAL: int
    R_BRACE_LITERAL: int
    L_BRACKET: int
    R_BRACKET: int
    BAR: int
    R_BAR: int
    L_BAR: int
    L_ANGLE: int
    R_ANGLE: int
    FUNC_LIM: int
    LIM_APPROACH_SYM: int
    FUNC_INT: int
    FUNC_SUM: int
    FUNC_PROD: int
    FUNC_EXP: int
    FUNC_LOG: int
    FUNC_LG: int
    FUNC_LN: int
    FUNC_SIN: int
    FUNC_COS: int
    FUNC_TAN: int
    FUNC_CSC: int
    FUNC_SEC: int
    FUNC_COT: int
    FUNC_ARCSIN: int
    FUNC_ARCCOS: int
    FUNC_ARCTAN: int
    FUNC_ARCCSC: int
    FUNC_ARCSEC: int
    FUNC_ARCCOT: int
    FUNC_SINH: int
    FUNC_COSH: int
    FUNC_TANH: int
    FUNC_ARSINH: int
    FUNC_ARCOSH: int
    FUNC_ARTANH: int
    L_FLOOR: int
    R_FLOOR: int
    L_CEIL: int
    R_CEIL: int
    FUNC_SQRT: int
    FUNC_OVERLINE: int
    CMD_TIMES: int
    CMD_CDOT: int
    CMD_DIV: int
    CMD_FRAC: int
    CMD_BINOM: int
    CMD_DBINOM: int
    CMD_TBINOM: int
    CMD_MATHIT: int
    UNDERSCORE: int
    CARET: int
    COLON: int
    DIFFERENTIAL: int
    LETTER: int
    DIGIT: int
    EQUAL: int
    NEQ: int
    LT: int
    LTE: int
    LTE_Q: int
    LTE_S: int
    GT: int
    GTE: int
    GTE_Q: int
    GTE_S: int
    BANG: int
    SINGLE_QUOTES: int
    SYMBOL: int
    channelNames: Incomplete
    modeNames: Incomplete
    literalNames: Incomplete
    symbolicNames: Incomplete
    ruleNames: Incomplete
    grammarFileName: str
    _interp: Incomplete
    _actions: Incomplete
    _predicates: Incomplete
    def __init__(self, input: Incomplete | None = None, output: TextIO = ...) -> None: ...

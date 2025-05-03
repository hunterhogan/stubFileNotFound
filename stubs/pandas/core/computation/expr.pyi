import ast
from _typeshed import Incomplete
from pandas.core.computation.ops import ARITH_OPS_SYMS as ARITH_OPS_SYMS, BOOL_OPS_SYMS as BOOL_OPS_SYMS, BinOp as BinOp, CMP_OPS_SYMS as CMP_OPS_SYMS, Constant as Constant, FuncNode as FuncNode, LOCAL_TAG as LOCAL_TAG, MATHOPS as MATHOPS, Op as Op, REDUCTIONS as REDUCTIONS, Term as Term, UNARY_OPS_SYMS as UNARY_OPS_SYMS, UnaryOp as UnaryOp, is_term as is_term
from pandas.core.computation.parsing import clean_backtick_quoted_toks as clean_backtick_quoted_toks, tokenize_string as tokenize_string
from pandas.core.computation.scope import Scope as Scope
from pandas.errors import UndefinedVariableError as UndefinedVariableError
from pandas.io.formats import printing as printing
from typing import Callable, ClassVar, TypeVar

def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace local variables with a syntactically valid name.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values

    Notes
    -----
    This is somewhat of a hack in that we rewrite a string such as ``'@a'`` as
    ``'__pd_eval_local_a'`` by telling the tokenizer that ``__pd_eval_local_``
    is a ``tokenize.OP`` and to replace the ``'@'`` symbol with it.
    """
def _compose2(f, g):
    """
    Compose 2 callables.
    """
def _compose(*funcs):
    """
    Compose 2 or more callables.
    """
def _preparse(source: str, f=...) -> str:
    """
    Compose a collection of tokenization functions.

    Parameters
    ----------
    source : str
        A Python source code string
    f : callable
        This takes a tuple of (toknum, tokval) as its argument and returns a
        tuple with the same structure but possibly different elements. Defaults
        to the composition of ``_rewrite_assign``, ``_replace_booleans``, and
        ``_replace_locals``.

    Returns
    -------
    str
        Valid Python source code

    Notes
    -----
    The `f` parameter can be any callable that takes *and* returns input of the
    form ``(toknum, tokval)``, where ``toknum`` is one of the constants from
    the ``tokenize`` module and ``tokval`` is a string.
    """
def _is_type(t):
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """

_is_list: Incomplete
_is_str: Incomplete
_all_nodes: Incomplete

def _filter_nodes(superclass, all_nodes=...):
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """

_all_node_names: Incomplete
_mod_nodes: Incomplete
_stmt_nodes: Incomplete
_expr_nodes: Incomplete
_expr_context_nodes: Incomplete
_boolop_nodes: Incomplete
_operator_nodes: Incomplete
_unary_op_nodes: Incomplete
_cmp_op_nodes: Incomplete
_comprehension_nodes: Incomplete
_handler_nodes: Incomplete
_arguments_nodes: Incomplete
_keyword_nodes: Incomplete
_alias_nodes: Incomplete
_hacked_nodes: Incomplete
_unsupported_expr_nodes: Incomplete
_unsupported_nodes: Incomplete
_base_supported_nodes: Incomplete
intersection: Incomplete
_msg: Incomplete

def _node_not_implemented(node_name: str) -> Callable[..., None]:
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """
_T = TypeVar('_T')

def disallow(nodes: set[str]) -> Callable[[type[_T]], type[_T]]:
    """
    Decorator to disallow certain nodes from parsing. Raises a
    NotImplementedError instead.

    Returns
    -------
    callable
    """
def _op_maker(op_class, op_symbol):
    """
    Return a function to create an op class with its symbol already passed.

    Returns
    -------
    callable
    """

_op_classes: Incomplete

def add_ops(op_classes):
    """
    Decorator to add default implementation of ops.
    """

class BaseExprVisitor(ast.NodeVisitor):
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.

    Parameters
    ----------
    env : Scope
    engine : str
    parser : str
    preparser : callable
    """
    const_type: ClassVar[type[Term]]
    term_type: ClassVar[type[Term]]
    binary_ops: Incomplete
    binary_op_nodes: Incomplete
    binary_op_nodes_map: Incomplete
    unary_ops = UNARY_OPS_SYMS
    unary_op_nodes: Incomplete
    unary_op_nodes_map: Incomplete
    rewrite_map: Incomplete
    unsupported_nodes: tuple[str, ...]
    env: Incomplete
    engine: Incomplete
    parser: Incomplete
    preparser: Incomplete
    assigner: Incomplete
    def __init__(self, env, engine, parser, preparser=...) -> None: ...
    def visit(self, node, **kwargs): ...
    def visit_Module(self, node, **kwargs): ...
    def visit_Expr(self, node, **kwargs): ...
    def _rewrite_membership_op(self, node, left, right): ...
    def _maybe_transform_eq_ne(self, node, left: Incomplete | None = None, right: Incomplete | None = None): ...
    def _maybe_downcast_constants(self, left, right): ...
    def _maybe_eval(self, binop, eval_in_python): ...
    def _maybe_evaluate_binop(self, op, op_class, lhs, rhs, eval_in_python=('in', 'not in'), maybe_eval_in_python=('==', '!=', '<', '>', '<=', '>=')): ...
    def visit_BinOp(self, node, **kwargs): ...
    def visit_UnaryOp(self, node, **kwargs): ...
    def visit_Name(self, node, **kwargs) -> Term: ...
    def visit_NameConstant(self, node, **kwargs) -> Term: ...
    def visit_Num(self, node, **kwargs) -> Term: ...
    def visit_Constant(self, node, **kwargs) -> Term: ...
    def visit_Str(self, node, **kwargs) -> Term: ...
    def visit_List(self, node, **kwargs) -> Term: ...
    visit_Tuple = visit_List
    def visit_Index(self, node, **kwargs):
        """df.index[4]"""
    def visit_Subscript(self, node, **kwargs) -> Term: ...
    def visit_Slice(self, node, **kwargs) -> slice:
        """df.index[slice(4,6)]"""
    def visit_Assign(self, node, **kwargs):
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
    def visit_Attribute(self, node, **kwargs): ...
    def visit_Call(self, node, side: Incomplete | None = None, **kwargs): ...
    def translate_In(self, op): ...
    def visit_Compare(self, node, **kwargs): ...
    def _try_visit_binop(self, bop): ...
    def visit_BoolOp(self, node, **kwargs): ...

_python_not_supported: Incomplete
_numexpr_supported_calls: Incomplete

class PandasExprVisitor(BaseExprVisitor):
    def __init__(self, env, engine, parser, preparser=...) -> None: ...

class PythonExprVisitor(BaseExprVisitor):
    def __init__(self, env, engine, parser, preparser=...) -> None: ...

class Expr:
    """
    Object encapsulating an expression.

    Parameters
    ----------
    expr : str
    engine : str, optional, default 'numexpr'
    parser : str, optional, default 'pandas'
    env : Scope, optional, default None
    level : int, optional, default 2
    """
    env: Scope
    engine: str
    parser: str
    expr: Incomplete
    _visitor: Incomplete
    terms: Incomplete
    def __init__(self, expr, engine: str = 'numexpr', parser: str = 'pandas', env: Scope | None = None, level: int = 0) -> None: ...
    @property
    def assigner(self): ...
    def __call__(self): ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def parse(self):
        """
        Parse an expression.
        """
    @property
    def names(self):
        """
        Get the names in an expression.
        """

PARSERS: Incomplete

from _typeshed import Incomplete
from sympy.assumptions.ask import Q as Q
from sympy.assumptions.assume import AppliedPredicate as AppliedPredicate, global_assumptions as global_assumptions
from sympy.assumptions.cnf import CNF as CNF, EncodedCNF as EncodedCNF
from sympy.core.kind import NumberKind as NumberKind
from sympy.core.mul import Mul as Mul
from sympy.core.singleton import S as S
from sympy.logic.algorithms.lra_theory import ALLOWED_PRED as ALLOWED_PRED, UnhandledInput as UnhandledInput
from sympy.logic.inference import satisfiable as satisfiable
from sympy.matrices.kind import MatrixKind as MatrixKind

def lra_satask(proposition, assumptions: bool = True, context=...):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm
    in conjunction with an Linear Real Arithmetic theory solver.

    Used to handle inequalities. Should eventually be depreciated and combined
    into satask, but infinity handling and other things need to be implemented
    before that can happen.
    """

WHITE_LIST: Incomplete

def check_satisfiability(prop, _prop, factbase): ...
def _preprocess(enc_cnf):
    """
    Returns an encoded cnf with only Q.eq, Q.gt, Q.lt,
    Q.ge, and Q.le predicate.

    Converts every unequality into a disjunction of strict
    inequalities. For example, x != 3 would become
    x < 3 OR x > 3.

    Also converts all negated Q.ne predicates into
    equalities.
    """
def _pred_to_binrel(pred): ...

pred_to_pos_neg_zero: Incomplete

def get_all_pred_and_expr_from_enc_cnf(enc_cnf): ...
def extract_pred_from_old_assum(all_exprs):
    '''
    Returns a list of relevant new assumption predicate
    based on any old assumptions.

    Raises an UnhandledInput exception if any of the assumptions are
    unhandled.

    Ignored predicate:
    - commutative
    - complex
    - algebraic
    - transcendental
    - extended_real
    - real
    - all matrix predicate
    - rational
    - irrational

    Example
    =======
    >>> from sympy.assumptions.lra_satask import extract_pred_from_old_assum
    >>> from sympy import symbols
    >>> x, y = symbols("x y", positive=True)
    >>> extract_pred_from_old_assum([x, y, 2])
    [Q.positive(x), Q.positive(y)]
    '''

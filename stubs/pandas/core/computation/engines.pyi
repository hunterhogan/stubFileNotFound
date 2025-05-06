import abc
from _typeshed import Incomplete
from pandas.core.computation.align import align_terms as align_terms, reconstruct_object as reconstruct_object
from pandas.core.computation.expr import Expr as Expr
from pandas.core.computation.ops import MATHOPS as MATHOPS, REDUCTIONS as REDUCTIONS
from pandas.errors import NumExprClobberingError as NumExprClobberingError
from pandas.io.formats import printing as printing

_ne_builtins: Incomplete

def _check_ne_builtin_clash(expr: Expr) -> None:
    """
    Attempt to prevent foot-shooting in a helpful way.

    Parameters
    ----------
    expr : Expr
        Terms can contain
    """

class AbstractEngine(metaclass=abc.ABCMeta):
    """Object serving as a base class for all engines."""
    has_neg_frac: bool
    expr: Incomplete
    aligned_axes: Incomplete
    result_type: Incomplete
    def __init__(self, expr) -> None: ...
    def convert(self) -> str:
        """
        Convert an expression for evaluation.

        Defaults to return the expression as a string.
        """
    def evaluate(self) -> object:
        """
        Run the engine on the expression.

        This method performs alignment which is necessary no matter what engine
        is being used, thus its implementation is in the base class.

        Returns
        -------
        object
            The result of the passed expression.
        """
    @property
    def _is_aligned(self) -> bool: ...
    @abc.abstractmethod
    def _evaluate(self):
        """
        Return an evaluated expression.

        Parameters
        ----------
        env : Scope
            The local and global environment in which to evaluate an
            expression.

        Notes
        -----
        Must be implemented by subclasses.
        """

class NumExprEngine(AbstractEngine):
    """NumExpr engine class"""
    has_neg_frac: bool
    def _evaluate(self): ...

class PythonEngine(AbstractEngine):
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """
    has_neg_frac: bool
    def evaluate(self): ...
    def _evaluate(self) -> None: ...

ENGINES: dict[str, type[AbstractEngine]]

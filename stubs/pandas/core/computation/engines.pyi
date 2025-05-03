import _abc
import pandas.io.formats.printing as printing
from pandas.core.computation.align import align_terms as align_terms, reconstruct_object as reconstruct_object
from pandas.errors import NumExprClobberingError as NumExprClobberingError
from typing import ClassVar

TYPE_CHECKING: bool
MATHOPS: tuple
REDUCTIONS: tuple
_ne_builtins: frozenset
def _check_ne_builtin_clash(expr: Expr) -> None:
    """
    Attempt to prevent foot-shooting in a helpful way.

    Parameters
    ----------
    expr : Expr
        Terms can contain
    """

class AbstractEngine:
    has_neg_frac: ClassVar[bool] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
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
    @property
    def _is_aligned(self): ...

class NumExprEngine(AbstractEngine):
    has_neg_frac: ClassVar[bool] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def _evaluate(self): ...

class PythonEngine(AbstractEngine):
    has_neg_frac: ClassVar[bool] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def evaluate(self): ...
    def _evaluate(self) -> None: ...
ENGINES: dict

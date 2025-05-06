from sympy.core.function import Add as Add, ArgumentIndexError as ArgumentIndexError, Function as Function
from sympy.functions.elementary.trigonometric import cos as cos, sin as sin

def _cosm1(x, *, evaluate: bool = True): ...

class cosm1(Function):
    """ Minus one plus cosine of x, i.e. cos(x) - 1. For use when x is close to zero.

    Helper class for use with e.g. scipy.special.cosm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosm1.html
    """
    nargs: int
    def fdiff(self, argindex: int = 1):
        """
        Returns the first derivative of this function.
        """
    def _eval_rewrite_as_cos(self, x, **kwargs): ...
    def _eval_evalf(self, *args, **kwargs): ...
    def _eval_simplify(self, **kwargs): ...

def _powm1(x, y, *, evaluate: bool = True): ...

class powm1(Function):
    """ Minus one plus x to the power of y, i.e. x**y - 1. For use when x is close to one or y is close to zero.

    Helper class for use with e.g. scipy.special.powm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html
    """
    nargs: int
    def fdiff(self, argindex: int = 1):
        """
        Returns the first derivative of this function.
        """
    def _eval_rewrite_as_Pow(self, x, y, **kwargs): ...
    def _eval_evalf(self, *args, **kwargs): ...
    def _eval_simplify(self, **kwargs): ...

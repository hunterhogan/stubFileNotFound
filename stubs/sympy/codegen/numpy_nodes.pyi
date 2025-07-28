from .ast import Token as Token, none as none
from _typeshed import Incomplete
from sympy.core.function import Add as Add, ArgumentIndexError as ArgumentIndexError, Function as Function
from sympy.core.power import Pow as Pow
from sympy.core.singleton import S as S
from sympy.core.sorting import default_sort_key as default_sort_key
from sympy.core.sympify import sympify as sympify
from sympy.functions.elementary.exponential import exp as exp, log as log
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min

def _logaddexp(x1, x2, *, evaluate: bool = True): ...

_two: Incomplete
_ln2: Incomplete

def _lb(x, *, evaluate: bool = True): ...
def _exp2(x, *, evaluate: bool = True): ...
def _logaddexp2(x1, x2, *, evaluate: bool = True): ...

class logaddexp(Function):
    """ Logarithm of the sum of exponentiations of the inputs.

    Helper class for use with e.g. numpy.logaddexp

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    """
    nargs: int
    def __new__(cls, *args): ...
    def fdiff(self, argindex: int = 1):
        """
        Returns the first derivative of this function.
        """
    def _eval_rewrite_as_log(self, x1, x2, **kwargs): ...
    def _eval_evalf(self, *args, **kwargs): ...
    def _eval_simplify(self, *args, **kwargs): ...

class logaddexp2(Function):
    """ Logarithm of the sum of exponentiations of the inputs in base-2.

    Helper class for use with e.g. numpy.logaddexp2

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
    """
    nargs: int
    def __new__(cls, *args): ...
    def fdiff(self, argindex: int = 1):
        """
        Returns the first derivative of this function.
        """
    def _eval_rewrite_as_log(self, x1, x2, **kwargs): ...
    def _eval_evalf(self, *args, **kwargs): ...
    def _eval_simplify(self, *args, **kwargs): ...

class amin(Token):
    """ Minimum value along an axis.

    Helper class for use with e.g. numpy.amin


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amin.html
    """
    __slots__: Incomplete
    _fields: Incomplete
    defaults: Incomplete
    _construct_axis: Incomplete

class amax(Token):
    """ Maximum value along an axis.

    Helper class for use with e.g. numpy.amax


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amax.html
    """
    __slots__: Incomplete
    _fields: Incomplete
    defaults: Incomplete
    _construct_axis: Incomplete

class maximum(Function):
    """ Element-wise maximum of array elements.

    Helper class for use with e.g. numpy.maximum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    """
    def _eval_rewrite_as_Max(self, *args): ...

class minimum(Function):
    """ Element-wise minimum of array elements.

    Helper class for use with e.g. numpy.minimum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    """
    def _eval_rewrite_as_Min(self, *args): ...

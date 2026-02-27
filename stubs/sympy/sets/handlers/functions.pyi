from _typeshed import Incomplete
from sympy.calculus.singularities import singularities as singularities
from sympy.core import Add as Add, Expr as Expr
from sympy.core.function import diff as diff, expand_mul as expand_mul, FunctionClass as FunctionClass, Lambda as Lambda
from sympy.core.numbers import Float as Float, oo as oo
from sympy.core.singleton import S as S
from sympy.core.symbol import Dummy as Dummy, symbols as symbols, Wild as Wild
from sympy.functions.elementary.exponential import exp as exp, log as log, match_real_imag as match_real_imag
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min
from sympy.logic.boolalg import true as true
from sympy.multipledispatch import Dispatcher as Dispatcher
from sympy.sets import (
	Complement as Complement, FiniteSet as FiniteSet, ImageSet as ImageSet, imageset as imageset,
	Intersection as Intersection, Interval as Interval, Range as Range, Union as Union)
from sympy.sets.fancysets import Integers as Integers, Naturals as Naturals, Reals as Reals
from sympy.sets.sets import (
	EmptySet as EmptySet, is_function_invertible_in_set as is_function_invertible_in_set, Set as Set)

_x: Incomplete
_y: Incomplete
FunctionUnion: Incomplete
_set_function: Incomplete

def _(f, x) -> None: ...

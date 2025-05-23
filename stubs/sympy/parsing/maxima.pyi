from _typeshed import Incomplete
from sympy.concrete.products import product as product
from sympy.concrete.summations import Sum as Sum
from sympy.core.sympify import sympify as sympify
from sympy.functions.elementary.trigonometric import cos as cos, sin as sin

class MaximaHelpers:
    def maxima_expand(expr): ...
    def maxima_float(expr): ...
    def maxima_trigexpand(expr): ...
    def maxima_sum(a1, a2, a3, a4): ...
    def maxima_product(a1, a2, a3, a4): ...
    def maxima_csc(expr): ...
    def maxima_sec(expr): ...

sub_dict: Incomplete
var_name: Incomplete

def parse_maxima(str, globals: Incomplete | None = None, name_dict={}): ...

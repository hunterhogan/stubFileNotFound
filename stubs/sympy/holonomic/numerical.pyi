from sympy.core.sympify import sympify as sympify
from sympy.holonomic.holonomic import DMFsubs as DMFsubs

def _evalf(func, points, derivatives: bool = False, method: str = 'RK4'):
    """
    Numerical methods for numerical integration along a given set of
    points in the complex plane.
    """
def _euler(red, x0, x1, y0, a):
    """
    Euler's method for numerical integration.
    From x0 to x1 with initial values given at x0 as vector y0.
    """
def _rk4(red, x0, x1, y0, a):
    """
    Runge-Kutta 4th order numerical method.
    """

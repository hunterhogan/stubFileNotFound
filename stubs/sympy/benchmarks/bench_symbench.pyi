from sympy.abc import x as x, y as y, z as z
from sympy.core.numbers import I as I, Integer as Integer, pi as pi

def bench_R1():
    """real(f(f(f(f(f(f(f(f(f(f(i/2)))))))))))"""
def bench_R2():
    """Hermite polynomial hermite(15, y)"""
def bench_R3() -> None:
    """a = [bool(f==f) for _ in range(10)]"""
def bench_R4() -> None: ...
def bench_R5():
    """blowup(L, 8); L=uniq(L)"""
def bench_R6() -> None:
    """sum(simplify((x+sin(i))/x+(x-sin(i))/x) for i in range(100))"""
def bench_R7() -> None:
    """[f.subs(x, random()) for _ in range(10**4)]"""
def bench_R8():
    """right(x^2,0,5,10^4)"""
def _bench_R9() -> None:
    """factor(x^20 - pi^5*y^20)"""
def bench_R10():
    """v = [-pi,-pi+1/10..,pi]"""
def bench_R11() -> None:
    """a = [random() + random()*I for w in [0..1000]]"""
def bench_S1() -> None:
    """e=(x+y+z+1)**7;f=e*(e+1);f.expand()"""

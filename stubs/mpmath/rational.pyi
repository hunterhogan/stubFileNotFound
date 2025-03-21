from .libmp import HASH_MODULUS as HASH_MODULUS, bitcount as bitcount, from_man_exp as from_man_exp, int_types as int_types, mpf_hash as mpf_hash
from _typeshed import Incomplete

new: Incomplete

def create_reduced(p, q, _cache={}): ...

class mpq:
    """
    Exact rational type, currently only intended for internal use.
    """
    __slots__: Incomplete
    def __new__(cls, p, q: int = 1): ...
    def __repr__(s) -> str: ...
    def __str__(s) -> str: ...
    def __int__(s) -> int: ...
    def __nonzero__(s): ...
    __bool__ = __nonzero__
    def __hash__(s): ...
    def __eq__(s, t): ...
    def __ne__(s, t): ...
    def _cmp(s, t, op): ...
    def __lt__(s, t): ...
    def __le__(s, t): ...
    def __gt__(s, t): ...
    def __ge__(s, t): ...
    def __abs__(s): ...
    def __neg__(s): ...
    def __pos__(s): ...
    def __add__(s, t): ...
    __radd__ = __add__
    def __sub__(s, t): ...
    def __rsub__(s, t): ...
    def __mul__(s, t): ...
    __rmul__ = __mul__
    def __div__(s, t): ...
    def __rdiv__(s, t): ...
    def __pow__(s, t): ...

mpq_1: Incomplete
mpq_0: Incomplete
mpq_1_2: Incomplete
mpq_3_2: Incomplete
mpq_1_4: Incomplete
mpq_1_16: Incomplete
mpq_3_16: Incomplete
mpq_5_2: Incomplete
mpq_3_4: Incomplete
mpq_7_4: Incomplete
mpq_5_4: Incomplete

from _typeshed import Incomplete
from collections.abc import Generator

class PlotInterval:
    """
    """
    _v: Incomplete
    _v_min: Incomplete
    _v_max: Incomplete
    _v_steps: Incomplete
    def require_all_args(f): ...
    v: Incomplete
    v_min: Incomplete
    v_max: Incomplete
    v_steps: Incomplete
    def __init__(self, *args) -> None: ...
    def get_v(self): ...
    def set_v(self, v) -> None: ...
    def get_v_min(self): ...
    def set_v_min(self, v_min) -> None: ...
    def get_v_max(self): ...
    def set_v_max(self, v_max) -> None: ...
    def get_v_steps(self): ...
    def set_v_steps(self, v_steps) -> None: ...
    def get_v_len(self): ...
    v_len: Incomplete
    def fill_from(self, b) -> None: ...
    @staticmethod
    def try_parse(*args):
        """
        Returns a PlotInterval if args can be interpreted
        as such, otherwise None.
        """
    def _str_base(self): ...
    def __repr__(self) -> str:
        """
        A string representing the interval in class constructor form.
        """
    def __str__(self) -> str:
        """
        A string representing the interval in list form.
        """
    def assert_complete(self) -> None: ...
    def vrange(self) -> Generator[Incomplete]:
        """
        Yields v_steps+1 SymPy numbers ranging from
        v_min to v_max.
        """
    def vrange2(self) -> Generator[Incomplete]:
        """
        Yields v_steps pairs of SymPy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        """
    def frange(self) -> Generator[Incomplete]: ...

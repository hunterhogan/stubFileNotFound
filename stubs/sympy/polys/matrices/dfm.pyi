from ._dfm import DFM as DFM
from sympy.external.gmpy import GROUND_TYPES as GROUND_TYPES

class DFM_dummy:
    """
        Placeholder class for DFM when python-flint is not installed.
        """
    def __init__(*args, **kwargs) -> None: ...
    @classmethod
    def _supports_domain(cls, domain): ...
    @classmethod
    def _get_flint_func(cls, domain) -> None: ...
DFM = DFM_dummy

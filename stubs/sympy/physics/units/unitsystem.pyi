from .dimensions import Dimension as Dimension
from _typeshed import Incomplete
from sympy.core.add import Add as Add
from sympy.core.function import Derivative as Derivative, Function as Function
from sympy.core.mul import Mul as Mul
from sympy.core.power import Pow as Pow
from sympy.core.singleton import S as S
from sympy.physics.units.dimensions import _QuantityMapper as _QuantityMapper
from sympy.physics.units.quantities import Quantity as Quantity

class UnitSystem(_QuantityMapper):
    """
    UnitSystem represents a coherent set of units.

    A unit system is basically a dimension system with notions of scales. Many
    of the methods are defined in the same way.

    It is much better if all base units have a symbol.
    """
    _unit_systems: dict[str, UnitSystem]
    name: Incomplete
    descr: Incomplete
    _base_units: Incomplete
    _dimension_system: Incomplete
    _units: Incomplete
    _derived_units: Incomplete
    def __init__(self, base_units, units=(), name: str = '', descr: str = '', dimension_system: Incomplete | None = None, derived_units: dict[Dimension, Quantity] = {}) -> None: ...
    def __str__(self) -> str:
        """
        Return the name of the system.

        If it does not exist, then it makes a list of symbols (or names) of
        the base dimensions.
        """
    def __repr__(self) -> str: ...
    def extend(self, base, units=(), name: str = '', description: str = '', dimension_system: Incomplete | None = None, derived_units: dict[Dimension, Quantity] = {}):
        """Extend the current system into a new one.

        Take the base and normal units of the current system to merge
        them to the base and normal units given in argument.
        If not provided, name and description are overridden by empty strings.
        """
    def get_dimension_system(self): ...
    def get_quantity_dimension(self, unit): ...
    def get_quantity_scale_factor(self, unit): ...
    @staticmethod
    def get_unit_system(unit_system): ...
    @staticmethod
    def get_default_unit_system(): ...
    @property
    def dim(self):
        """
        Give the dimension of the system.

        That is return the number of units forming the basis.
        """
    @property
    def is_consistent(self):
        """
        Check if the underlying dimension system is consistent.
        """
    @property
    def derived_units(self) -> dict[Dimension, Quantity]: ...
    def get_dimensional_expr(self, expr): ...
    def _collect_factor_and_dimension(self, expr):
        """
        Return tuple with scale factor expression and dimension expression.
        """
    def get_units_non_prefixed(self) -> set[Quantity]:
        """
        Return the units of the system that do not have a prefix.
        """

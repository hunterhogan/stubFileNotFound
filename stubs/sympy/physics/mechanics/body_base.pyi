import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod

__all__ = ['BodyBase']

class BodyBase(ABC, metaclass=abc.ABCMeta):
    """Abstract class for body type objects."""
    _name: Incomplete
    points: Incomplete
    def __init__(self, name, masscenter=None, mass=None) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def name(self):
        """The name of the body."""
    @property
    def masscenter(self):
        """The body's center of mass."""
    _masscenter: Incomplete
    @masscenter.setter
    def masscenter(self, point) -> None: ...
    @property
    def mass(self):
        """The body's mass."""
    _mass: Incomplete
    @mass.setter
    def mass(self, mass) -> None: ...
    @property
    def potential_energy(self):
        """The potential energy of the body.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point
        >>> from sympy import symbols
        >>> m, g, h = symbols('m g h')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.potential_energy = m * g * h
        >>> P.potential_energy
        g*h*m

        """
    _potential_energy: Incomplete
    @potential_energy.setter
    def potential_energy(self, scalar) -> None: ...
    @abstractmethod
    def kinetic_energy(self, frame): ...
    @abstractmethod
    def linear_momentum(self, frame): ...
    @abstractmethod
    def angular_momentum(self, point, frame): ...
    @abstractmethod
    def parallel_axis(self, point, frame): ...

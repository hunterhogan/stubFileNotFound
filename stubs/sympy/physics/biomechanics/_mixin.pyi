from _typeshed import Incomplete

__all__ = ['_NamedMixin']

class _NamedMixin:
    """Mixin class for adding `name` properties.

    Valid names, as will typically be used by subclasses as a suffix when
    naming automatically-instantiated symbol attributes, must be nonzero length
    strings.

    Attributes
    ==========

    name : str
        The name identifier associated with the instance. Must be a string of
        length at least 1.

    """
    @property
    def name(self) -> str:
        """The name associated with the class instance."""
    _name: Incomplete
    @name.setter
    def name(self, name: str) -> None: ...

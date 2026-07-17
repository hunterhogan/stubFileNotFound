from _typeshed import Incomplete

class Meter:
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """
    def reset(self) -> None:
        """Reset the meter to default settings."""
    def add(self, value) -> None:
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
    def value(self) -> None:
        """Get the value of the meter in the current state."""

class AverageValueMeter(Meter):
    val: int
    def __init__(self) -> None: ...
    mean: Incomplete
    std: Incomplete
    mean_old: Incomplete
    m_s: float
    def add(self, value, n: int = 1) -> None: ...
    def value(self): ...
    n: int
    sum: float
    var: float
    def reset(self) -> None: ...

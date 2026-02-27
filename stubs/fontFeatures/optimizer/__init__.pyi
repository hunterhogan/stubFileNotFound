
from .FontFeatures import optimizations as overall_optimizations
from .Routine import optimizations as routine_optimizations
import fontFeatures

class Optimizer:
    def __init__(self, ff) -> None:
        ...

    def optimize(self, level=...): # -> None:
        ...

    def optimize_routine(self, r, level): # -> None:
        ...

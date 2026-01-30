from _typeshed import Incomplete
from collections.abc import Generator
from fontTools.misc.transform import Identity as Identity, Scale as Scale

TWO_PI: Incomplete
PI_OVER_TWO: Incomplete

def _map_point(matrix, pt): ...

class EllipticalArc:
    current_point: Incomplete
    rx: Incomplete
    ry: Incomplete
    rotation: Incomplete
    large: Incomplete
    sweep: Incomplete
    target_point: Incomplete
    angle: Incomplete
    center_point: Incomplete
    def __init__(self, current_point, rx, ry, rotation, large, sweep, target_point) -> None: ...
    theta1: Incomplete
    theta2: Incomplete
    theta_arc: Incomplete
    def _parametrize(self): ...
    def _decompose_to_cubic_curves(self) -> Generator[Incomplete]: ...
    def draw(self, pen) -> None: ...

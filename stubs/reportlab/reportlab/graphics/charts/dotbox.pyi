from _typeshed import Incomplete

from reportlab.graphics.widgetbase import Widget

class DotBox(Widget):
    xlabels: Incomplete
    ylabels: Incomplete
    labelFontName: str
    labelFontSize: int
    labelOffset: int
    strokeWidth: float
    gridDivWidth: Incomplete
    gridColor: Incomplete
    dotDiameter: Incomplete
    dotColor: Incomplete
    dotXPosition: int
    dotYPosition: int
    x: int
    y: int
    def __init__(self) -> None: ...
    def demo(self, drawing: Incomplete | None = None): ...
    def draw(self): ...

from _typeshed import Incomplete

from reportlab.graphics.widgetbase import Widget

class SlideBox(Widget):
    labelFontName: str
    labelFontSize: int
    labelStrokeColor: Incomplete
    labelFillColor: Incomplete
    startColor: Incomplete
    endColor: Incomplete
    numberOfBoxes: int
    trianglePosition: int
    triangleHeight: Incomplete
    triangleWidth: Incomplete
    triangleFillColor: Incomplete
    triangleStrokeColor: Incomplete
    triangleStrokeWidth: float
    boxHeight: Incomplete
    boxWidth: Incomplete
    boxSpacing: Incomplete
    boxOutlineColor: Incomplete
    boxOutlineWidth: float
    leftPadding: int
    rightPadding: int
    topPadding: int
    bottomPadding: int
    background: Incomplete
    sourceLabelText: str
    sourceLabelOffset: Incomplete
    sourceLabelFontName: str
    sourceLabelFontSize: int
    sourceLabelFillColor: Incomplete
    def __init__(self) -> None: ...
    def demo(self, drawing: Incomplete | None = None): ...
    def draw(self): ...

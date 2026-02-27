from .backend_cairo import cairo as cairo, FigureCanvasCairo as FigureCanvasCairo
from .backend_qt import _BackendQT as _BackendQT, FigureCanvasQT as FigureCanvasQT
from .qt_compat import QT_API as QT_API, QtCore as QtCore, QtGui as QtGui

class FigureCanvasQTCairo(FigureCanvasCairo, FigureCanvasQT):
    def draw(self) -> None: ...
    def paintEvent(self, event) -> None: ...

class _BackendQTCairo(_BackendQT):
    FigureCanvas = FigureCanvasQTCairo

from .. import backends as backends
from .backend_qtcairo import (
	_BackendQTCairo as _BackendQTCairo, FigureCanvasCairo as FigureCanvasCairo, FigureCanvasQT as FigureCanvasQT,
	FigureCanvasQTCairo as FigureCanvasQTCairo)

class _BackendQT5Cairo(_BackendQTCairo): ...

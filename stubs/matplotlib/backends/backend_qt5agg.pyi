from .. import backends as backends
from .backend_qtagg import (
	_BackendQTAgg as _BackendQTAgg, FigureCanvasAgg as FigureCanvasAgg, FigureCanvasQT as FigureCanvasQT,
	FigureCanvasQTAgg as FigureCanvasQTAgg, FigureManagerQT as FigureManagerQT,
	NavigationToolbar2QT as NavigationToolbar2QT)

class _BackendQT5Agg(_BackendQTAgg): ...

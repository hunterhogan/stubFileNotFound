from .. import backends as backends
from .backend_qt import (
	_BackendQT as _BackendQT, _create_qApp as _create_qApp, ConfigureSubplotsQt as ConfigureSubplotsQt, cursord as cursord,
	FigureCanvasBase as FigureCanvasBase, FigureCanvasQT as FigureCanvasQT, FigureManagerBase as FigureManagerBase,
	FigureManagerQT as FigureManagerQT, figureoptions as figureoptions, Gcf as Gcf, HelpQt as HelpQt,
	MainWindow as MainWindow, MouseButton as MouseButton, NavigationToolbar2 as NavigationToolbar2,
	NavigationToolbar2QT as NavigationToolbar2QT, RubberbandQt as RubberbandQt, SaveFigureQt as SaveFigureQt,
	SPECIAL_KEYS as SPECIAL_KEYS, SubplotToolQt as SubplotToolQt, TimerBase as TimerBase, TimerQT as TimerQT,
	ToolbarQt as ToolbarQt, ToolContainerBase as ToolContainerBase, ToolCopyToClipboardQT as ToolCopyToClipboardQT)

class _BackendQT5(_BackendQT): ...

def __getattr__(name): ...

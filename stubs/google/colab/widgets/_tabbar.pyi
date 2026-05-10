import contextlib
from _typeshed import Incomplete
from collections.abc import Generator
from google.colab.widgets import _widget

class TabBar(_widget.OutputAreaWidget):
    BASE_PATH: str
    TABBAR_JS: Incomplete
    TAB_CSS: Incomplete
    tab_names: Incomplete
    def __init__(self, tab_names, location: str = 'top') -> None: ...
    @contextlib.contextmanager
    def output_to(self, tab, select: bool = True) -> Generator[None]: ...
    def clear_tab(self, tab=None) -> None: ...
    def __iter__(self): ...

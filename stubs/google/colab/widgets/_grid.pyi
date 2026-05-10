import contextlib
from _typeshed import Incomplete
from collections.abc import Generator
from google.colab.widgets import _widget

class Grid(_widget.OutputAreaWidget):
    rows: Incomplete
    columns: Incomplete
    header_row: Incomplete
    header_column: Incomplete
    def __init__(self, rows, columns, header_row: bool = False, header_column: bool = False, style: str = '') -> None: ...
    def clear_cell(self, row=None, col=None) -> None: ...
    def __iter__(self): ...
    @contextlib.contextmanager
    def output_to(self, row, column) -> Generator[None]: ...

def create_grid(row_data, col_data, render, header_render=None, header_row: bool = True, header_column: bool = True): ...

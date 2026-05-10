import IPython as _IPython
from IPython import display as _display
from _typeshed import Incomplete

__all__ = ['DataTable', 'enable_dataframe_formatter', 'disable_dataframe_formatter', 'load_ipython_extension', 'unload_ipython_extension', 'display_dataframe']

class DataTable(_display.DisplayObject):
    include_index: bool
    num_rows_per_page: int
    max_rows: int
    max_columns: int
    min_width: Incomplete
    @classmethod
    def formatter(cls, dataframe, **kwargs): ...
    def __init__(self, dataframe, include_index=None, num_rows_per_page=None, max_rows=None, max_columns=None, min_width=None) -> None: ...

def display_dataframe(df) -> None: ...

class _JavascriptModuleFormatter(_IPython.core.formatters.BaseFormatter):
    format_type: Incomplete
    print_method: Incomplete

def enable_dataframe_formatter() -> None: ...
def disable_dataframe_formatter() -> None: ...
def load_ipython_extension(ipython) -> None: ...
def unload_ipython_extension(ipython) -> None: ...

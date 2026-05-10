import anywidget
from _typeshed import Incomplete

class InteractiveTable(anywidget.AnyWidget):
    data: Incomplete
    active_data: Incomplete
    columns: Incomplete
    rows: Incomplete
    page_num: Incomplete
    page_size: Incomplete
    sort_column: Incomplete
    sort_ascending: Incomplete
    def __init__(self, df=None, **kwargs) -> None: ...
    def set_page(self, page_num: int): ...
    def update_active_data(self) -> None: ...
    def set_data(self, df) -> None: ...

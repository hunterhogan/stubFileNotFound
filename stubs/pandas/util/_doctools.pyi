import pd as pd

TYPE_CHECKING: bool

class TablePlotter:
    def __init__(self, cell_width: float = ..., cell_height: float = ..., font_size: float = ...) -> None: ...
    def _shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Calculate table shape considering index levels.
        """
    def _get_cells(self, left, right, vertical) -> tuple[int, int]:
        """
        Calculate appropriate figure size based on left and right data.
        """
    def plot(self, left, right, labels: Iterable[str] = ..., vertical: bool = ...):
        """
        Plot left / right DataFrames in specified layout.

        Parameters
        ----------
        left : list of DataFrames before operation is applied
        right : DataFrame of operation result
        labels : list of str to be drawn as titles of left DataFrames
        vertical : bool, default True
            If True, use vertical layout. If False, use horizontal layout.
        """
    def _conv(self, data):
        """
        Convert each input to appropriate for table outplot.
        """
    def _insert_index(self, data): ...
    def _make_table(self, ax, df, title: str, height: float | None) -> None: ...
def main() -> None: ...

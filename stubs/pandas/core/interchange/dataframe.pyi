from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from pandas import DataFrame as DataFrame, Index as Index
from pandas.core.interchange.column import PandasColumn as PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg

class PandasDataFrameXchg(DataFrameXchg):
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.
    Instances of this (private) class are returned from
    ``pd.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """
    _df: Incomplete
    _allow_copy: Incomplete
    def __init__(self, df: DataFrame, allow_copy: bool = True) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `pd.DataFrame.__dataframe__`.
        """
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> PandasDataFrameXchg: ...
    @property
    def metadata(self) -> dict[str, Index]: ...
    def num_columns(self) -> int: ...
    def num_rows(self) -> int: ...
    def num_chunks(self) -> int: ...
    def column_names(self) -> Index: ...
    def get_column(self, i: int) -> PandasColumn: ...
    def get_column_by_name(self, name: str) -> PandasColumn: ...
    def get_columns(self) -> list[PandasColumn]: ...
    def select_columns(self, indices: Sequence[int]) -> PandasDataFrameXchg: ...
    def select_columns_by_name(self, names: list[str]) -> PandasDataFrameXchg: ...
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[PandasDataFrameXchg]:
        """
        Return an iterator yielding the chunks.
        """

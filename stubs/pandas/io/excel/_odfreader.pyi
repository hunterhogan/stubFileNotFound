import pandas as pd
import pandas.io.excel._base
from pandas._typing import ReadBuffer as ReadBuffer
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.io.excel._base import BaseExcelReader as BaseExcelReader
from pandas.util._decorators import doc as doc
from typing import ClassVar, FilePath, Scalar, StorageOptions

TYPE_CHECKING: bool
_shared_docs: dict

class ODFReader(pandas.io.excel._base.BaseExcelReader):
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    _docstring_components: ClassVar[list] = ...
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], storage_options: StorageOptions | None, engine_kwargs: dict | None) -> None:
        """
        Read tables out of OpenDocument formatted files.

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs) -> OpenDocument: ...
    def get_sheet_by_index(self, index: int): ...
    def get_sheet_by_name(self, name: str): ...
    def get_sheet_data(self, sheet, file_rows_needed: int | None) -> list[list[Scalar | NaTType]]:
        """
        Parse an ODF Table into a list of lists
        """
    def _get_row_repeat(self, row) -> int:
        """
        Return number of times this row was repeated
        Repeating an empty row appeared to be a common way
        of representing sparse rows in the table.
        """
    def _get_column_repeat(self, cell) -> int: ...
    def _get_cell_value(self, cell) -> Scalar | NaTType: ...
    def _get_cell_string_value(self, cell) -> str:
        """
        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        """
    @property
    def _workbook_class(self): ...
    @property
    def empty_value(self): ...
    @property
    def sheet_names(self): ...

from datetime import date, datetime, time, timedelta
from pandas._typing import FilePath as FilePath, NaTType as NaTType, ReadBuffer as ReadBuffer, Scalar as Scalar, StorageOptions as StorageOptions
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.shared_docs import _shared_docs as _shared_docs
from pandas.io.excel._base import BaseExcelReader as BaseExcelReader
from pandas.util._decorators import doc as doc
from python_calamine import CalamineSheet as CalamineSheet, CalamineWorkbook
from typing import Any

_CellValue = int | float | str | bool | time | date | datetime | timedelta

class CalamineReader(BaseExcelReader['CalamineWorkbook']):
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], storage_options: StorageOptions | None = None, engine_kwargs: dict | None = None) -> None:
        """
        Reader using calamine engine (xlsx/xls/xlsb/ods).

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
    @property
    def _workbook_class(self) -> type[CalamineWorkbook]: ...
    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs: Any) -> CalamineWorkbook: ...
    @property
    def sheet_names(self) -> list[str]: ...
    def get_sheet_by_name(self, name: str) -> CalamineSheet: ...
    def get_sheet_by_index(self, index: int) -> CalamineSheet: ...
    def get_sheet_data(self, sheet: CalamineSheet, file_rows_needed: int | None = None) -> list[list[Scalar | NaTType | time]]: ...

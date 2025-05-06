from pandas._typing import Scalar as Scalar, StorageOptions as StorageOptions
from pandas.io.excel._base import BaseExcelReader as BaseExcelReader
from xlrd import Book

class XlrdReader(BaseExcelReader['Book']):
    def __init__(self, filepath_or_buffer, storage_options: StorageOptions | None = None, engine_kwargs: dict | None = None) -> None:
        """
        Reader using xlrd engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
    @property
    def _workbook_class(self) -> type[Book]: ...
    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> Book: ...
    @property
    def sheet_names(self): ...
    def get_sheet_by_name(self, name): ...
    def get_sheet_by_index(self, index): ...
    def get_sheet_data(self, sheet, file_rows_needed: int | None = None) -> list[list[Scalar]]: ...

from pandas._typing import FilePath as FilePath, ReadBuffer as ReadBuffer, Scalar as Scalar, StorageOptions as StorageOptions
from pandas.io.excel._base import BaseExcelReader as BaseExcelReader
from pyxlsb import Workbook

class PyxlsbReader(BaseExcelReader['Workbook']):
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], storage_options: StorageOptions | None = None, engine_kwargs: dict | None = None) -> None:
        """
        Reader using pyxlsb engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object, or Workbook
            Object to be parsed.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
    @property
    def _workbook_class(self) -> type[Workbook]: ...
    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs) -> Workbook: ...
    @property
    def sheet_names(self) -> list[str]: ...
    def get_sheet_by_name(self, name: str): ...
    def get_sheet_by_index(self, index: int): ...
    def _convert_cell(self, cell) -> Scalar: ...
    def get_sheet_data(self, sheet, file_rows_needed: int | None = None) -> list[list[Scalar]]: ...

from odf.opendocument import OpenDocument
from pandas._libs.tslibs.nattype import NaTType as NaTType
from pandas._typing import FilePath as FilePath, ReadBuffer as ReadBuffer, Scalar as Scalar, StorageOptions as StorageOptions
from pandas.io.excel._base import BaseExcelReader as BaseExcelReader

class ODFReader(BaseExcelReader['OpenDocument']):
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], storage_options: StorageOptions | None = None, engine_kwargs: dict | None = None) -> None:
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
    @property
    def _workbook_class(self) -> type[OpenDocument]: ...
    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs) -> OpenDocument: ...
    @property
    def empty_value(self) -> str:
        """Property for compat with other readers."""
    @property
    def sheet_names(self) -> list[str]:
        """Return a list of sheet names present in the document"""
    def get_sheet_by_index(self, index: int): ...
    def get_sheet_by_name(self, name: str): ...
    def get_sheet_data(self, sheet, file_rows_needed: int | None = None) -> list[list[Scalar | NaTType]]:
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

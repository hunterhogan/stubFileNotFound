from _typeshed import Incomplete
from pandas._typing import ExcelWriterIfSheetExists as ExcelWriterIfSheetExists, FilePath as FilePath, StorageOptions as StorageOptions, WriteExcelBuffer as WriteExcelBuffer
from pandas.io.excel._base import ExcelWriter as ExcelWriter
from pandas.io.excel._util import combine_kwargs as combine_kwargs, validate_freeze_panes as validate_freeze_panes
from pandas.io.formats.excel import ExcelCell as ExcelCell
from typing import Any, overload

class ODSWriter(ExcelWriter):
    _engine: str
    _supported_extensions: Incomplete
    _book: Incomplete
    _style_dict: dict[str, str]
    def __init__(self, path: FilePath | WriteExcelBuffer | ExcelWriter, engine: str | None = None, date_format: str | None = None, datetime_format: Incomplete | None = None, mode: str = 'w', storage_options: StorageOptions | None = None, if_sheet_exists: ExcelWriterIfSheetExists | None = None, engine_kwargs: dict[str, Any] | None = None, **kwargs) -> None: ...
    @property
    def book(self):
        """
        Book instance of class odf.opendocument.OpenDocumentSpreadsheet.

        This attribute can be used to access engine-specific features.
        """
    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
    def _write_cells(self, cells: list[ExcelCell], sheet_name: str | None = None, startrow: int = 0, startcol: int = 0, freeze_panes: tuple[int, int] | None = None) -> None:
        """
        Write the frame cells using odf
        """
    def _make_table_cell_attributes(self, cell) -> dict[str, int | str]:
        """Convert cell attributes to OpenDocument attributes

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        attributes : Dict[str, Union[int, str]]
            Dictionary with attributes and attribute values
        """
    def _make_table_cell(self, cell) -> tuple[object, Any]:
        """Convert cell data to an OpenDocument spreadsheet cell

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        pvalue, cell : Tuple[str, TableCell]
            Display value, Cell value
        """
    @overload
    def _process_style(self, style: dict[str, Any]) -> str: ...
    @overload
    def _process_style(self, style: None) -> None: ...
    def _create_freeze_panes(self, sheet_name: str, freeze_panes: tuple[int, int]) -> None:
        """
        Create freeze panes in the sheet.

        Parameters
        ----------
        sheet_name : str
            Name of the spreadsheet
        freeze_panes : tuple of (int, int)
            Freeze pane location x and y
        """

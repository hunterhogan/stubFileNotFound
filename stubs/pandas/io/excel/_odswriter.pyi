import pandas.io.excel._base
from pandas.io.excel._base import ExcelWriter as ExcelWriter
from pandas.io.excel._util import combine_kwargs as combine_kwargs, validate_freeze_panes as validate_freeze_panes
from typing import Any, ClassVar

TYPE_CHECKING: bool

class ODSWriter(pandas.io.excel._base.ExcelWriter):
    _engine: ClassVar[str] = ...
    _supported_extensions: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, path: FilePath | WriteExcelBuffer | ExcelWriter, engine: str | None, date_format: str | None, datetime_format, mode: str = ..., storage_options: StorageOptions | None, if_sheet_exists: ExcelWriterIfSheetExists | None, engine_kwargs: dict[str, Any] | None, **kwargs) -> None: ...
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
    def _write_cells(self, cells: list[ExcelCell], sheet_name: str | None, startrow: int = ..., startcol: int = ..., freeze_panes: tuple[int, int] | None) -> None:
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
    def _process_style(self, style: dict[str, Any] | None) -> str | None:
        """Convert a style dictionary to a OpenDocument style sheet

        Parameters
        ----------
        style : Dict
            Style dictionary

        Returns
        -------
        style_key : str
            Unique style key for later reference in sheet
        """
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
    @property
    def book(self): ...
    @property
    def sheets(self): ...

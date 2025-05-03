from _typeshed import Incomplete
from pandas._typing import ExcelWriterIfSheetExists as ExcelWriterIfSheetExists, FilePath as FilePath, StorageOptions as StorageOptions, WriteExcelBuffer as WriteExcelBuffer
from pandas.io.excel._base import ExcelWriter as ExcelWriter
from pandas.io.excel._util import combine_kwargs as combine_kwargs, validate_freeze_panes as validate_freeze_panes
from typing import Any

class _XlsxStyler:
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]]
    @classmethod
    def convert(cls, style_dict, num_format_str: Incomplete | None = None):
        """
        converts a style_dict to an xlsxwriter format dict

        Parameters
        ----------
        style_dict : style dictionary to convert
        num_format_str : optional number format string
        """

class XlsxWriter(ExcelWriter):
    _engine: str
    _supported_extensions: Incomplete
    _book: Incomplete
    def __init__(self, path: FilePath | WriteExcelBuffer | ExcelWriter, engine: str | None = None, date_format: str | None = None, datetime_format: str | None = None, mode: str = 'w', storage_options: StorageOptions | None = None, if_sheet_exists: ExcelWriterIfSheetExists | None = None, engine_kwargs: dict[str, Any] | None = None, **kwargs) -> None: ...
    @property
    def book(self):
        """
        Book instance of class xlsxwriter.Workbook.

        This attribute can be used to access engine-specific features.
        """
    @property
    def sheets(self) -> dict[str, Any]: ...
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
    def _write_cells(self, cells, sheet_name: str | None = None, startrow: int = 0, startcol: int = 0, freeze_panes: tuple[int, int] | None = None) -> None: ...

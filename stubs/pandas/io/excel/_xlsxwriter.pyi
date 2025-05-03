import pandas.io.excel._base
from pandas.io.excel._base import ExcelWriter as ExcelWriter
from pandas.io.excel._util import combine_kwargs as combine_kwargs, validate_freeze_panes as validate_freeze_panes
from typing import Any, ClassVar

TYPE_CHECKING: bool

class _XlsxStyler:
    STYLE_MAPPING: ClassVar[dict] = ...
    @classmethod
    def convert(cls, style_dict, num_format_str):
        """
        converts a style_dict to an xlsxwriter format dict

        Parameters
        ----------
        style_dict : style dictionary to convert
        num_format_str : optional number format string
        """

class XlsxWriter(pandas.io.excel._base.ExcelWriter):
    _engine: ClassVar[str] = ...
    _supported_extensions: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, path: FilePath | WriteExcelBuffer | ExcelWriter, engine: str | None, date_format: str | None, datetime_format: str | None, mode: str = ..., storage_options: StorageOptions | None, if_sheet_exists: ExcelWriterIfSheetExists | None, engine_kwargs: dict[str, Any] | None, **kwargs) -> None: ...
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
    def _write_cells(self, cells, sheet_name: str | None, startrow: int = ..., startcol: int = ..., freeze_panes: tuple[int, int] | None) -> None: ...
    @property
    def book(self): ...
    @property
    def sheets(self): ...

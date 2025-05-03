import pandas.io.excel._base
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.io.excel._base import BaseExcelReader as BaseExcelReader
from pandas.util._decorators import doc as doc
from typing import ClassVar

TYPE_CHECKING: bool
_shared_docs: dict

class XlrdReader(pandas.io.excel._base.BaseExcelReader):
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, filepath_or_buffer, storage_options: StorageOptions | None, engine_kwargs: dict | None) -> None:
        '''
        Reader using xlrd engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        '''
    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> Book: ...
    def get_sheet_by_name(self, name): ...
    def get_sheet_by_index(self, index): ...
    def get_sheet_data(self, sheet, file_rows_needed: int | None) -> list[list[Scalar]]: ...
    @property
    def _workbook_class(self): ...
    @property
    def sheet_names(self): ...

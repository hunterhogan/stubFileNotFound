import numpy as np
import pandas as pd
from _typeshed import Incomplete
from collections import abc
from pandas import DataFrame as DataFrame, Timestamp as Timestamp, isna as isna
from pandas._libs.byteswap import read_double_with_byteswap as read_double_with_byteswap, read_float_with_byteswap as read_float_with_byteswap, read_uint16_with_byteswap as read_uint16_with_byteswap, read_uint32_with_byteswap as read_uint32_with_byteswap, read_uint64_with_byteswap as read_uint64_with_byteswap
from pandas._libs.sas import Parser as Parser, get_subheader_index as get_subheader_index
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized as cast_from_unit_vectorized
from pandas._typing import CompressionOptions as CompressionOptions, FilePath as FilePath, ReadBuffer as ReadBuffer
from pandas.errors import EmptyDataError as EmptyDataError
from pandas.io.common import get_handle as get_handle
from pandas.io.sas.sasreader import ReaderBase as ReaderBase

_unix_origin: Incomplete
_sas_origin: Incomplete

def _parse_datetime(sas_datetime: float, unit: str): ...
def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    '''
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {\'d\', \'s\'}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    '''

class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int
    def __init__(self, col_id: int, name: str | bytes, label: str | bytes, format: str | bytes, ctype: bytes, length: int) -> None: ...

class SAS7BDATReader(ReaderBase, abc.Iterator):
    """
    Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : str, 'infer', defaults to None
        String encoding acc. to Python standard encodings,
        encoding='infer' tries to detect the encoding from the file header,
        encoding=None will leave the data in binary format.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """
    _int_length: int
    _cached_page: bytes | None
    index: Incomplete
    convert_dates: Incomplete
    blank_missing: Incomplete
    chunksize: Incomplete
    encoding: Incomplete
    convert_text: Incomplete
    convert_header_text: Incomplete
    default_encoding: str
    compression: bytes
    column_names_raw: list[bytes]
    column_names: list[str | bytes]
    column_formats: list[str | bytes]
    columns: list[_Column]
    _current_page_data_subheader_pointers: list[tuple[int, int]]
    _column_data_lengths: list[int]
    _column_data_offsets: list[int]
    _column_types: list[bytes]
    _current_row_in_file_index: int
    _current_row_on_page_index: int
    handles: Incomplete
    _path_or_buf: Incomplete
    _subheader_processors: Incomplete
    def __init__(self, path_or_buf: FilePath | ReadBuffer[bytes], index: Incomplete | None = None, convert_dates: bool = True, blank_missing: bool = True, chunksize: int | None = None, encoding: str | None = None, convert_text: bool = True, convert_header_text: bool = True, compression: CompressionOptions = 'infer') -> None: ...
    def column_data_lengths(self) -> np.ndarray:
        """Return a numpy int64 array of the column data lengths"""
    def column_data_offsets(self) -> np.ndarray:
        """Return a numpy int64 array of the column offsets"""
    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
    def close(self) -> None: ...
    U64: bool
    _page_bit_offset: Incomplete
    _subheader_pointer_length: Incomplete
    byte_order: str
    need_byteswap: Incomplete
    inferred_encoding: Incomplete
    date_created: Incomplete
    date_modified: Incomplete
    header_length: Incomplete
    _page_length: Incomplete
    def _get_properties(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def _read_float(self, offset: int, width: int): ...
    def _read_uint(self, offset: int, width: int) -> int: ...
    def _read_bytes(self, offset: int, length: int): ...
    def _read_and_convert_header_text(self, offset: int, length: int) -> str | bytes: ...
    def _parse_metadata(self) -> None: ...
    def _process_page_meta(self) -> bool: ...
    _current_page_type: Incomplete
    _current_page_block_count: Incomplete
    _current_page_subheaders_count: Incomplete
    def _read_page_header(self) -> None: ...
    def _process_page_metadata(self) -> None: ...
    row_length: Incomplete
    row_count: Incomplete
    col_count_p1: Incomplete
    col_count_p2: Incomplete
    _mix_page_row_count: Incomplete
    _lcs: Incomplete
    _lcp: Incomplete
    def _process_rowsize_subheader(self, offset: int, length: int) -> None: ...
    column_count: Incomplete
    def _process_columnsize_subheader(self, offset: int, length: int) -> None: ...
    def _process_subheader_counts(self, offset: int, length: int) -> None: ...
    creator_proc: Incomplete
    def _process_columntext_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnname_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnattributes_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnlist_subheader(self, offset: int, length: int) -> None: ...
    def _process_format_subheader(self, offset: int, length: int) -> None: ...
    _string_chunk: Incomplete
    _byte_chunk: Incomplete
    _current_row_in_chunk_index: int
    def read(self, nrows: int | None = None) -> DataFrame: ...
    def _read_next_page(self): ...
    def _chunk_to_dataframe(self) -> DataFrame: ...
    def _decode_string(self, b): ...
    def _convert_header_text(self, b: bytes) -> str | bytes: ...

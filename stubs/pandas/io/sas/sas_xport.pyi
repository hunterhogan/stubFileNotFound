import pandas as pd
from _typeshed import Incomplete
from collections import abc
from pandas._typing import CompressionOptions as CompressionOptions, DatetimeNaTType as DatetimeNaTType, FilePath as FilePath, ReadBuffer as ReadBuffer
from pandas.io.sas.sasreader import ReaderBase as ReaderBase

_correct_line1: str
_correct_header1: str
_correct_header2: str
_correct_obs_header: str
_fieldkeys: Incomplete
_base_params_doc: str
_params2_doc: str
_format_params_doc: str
_iterator_doc: str
_read_sas_doc: Incomplete
_xport_reader_doc: Incomplete
_read_method_doc: str

def _parse_date(datestr: str) -> DatetimeNaTType:
    """Given a date in xport format, return Python date."""
def _split_line(s: str, parts):
    """
    Parameters
    ----------
    s: str
        Fixed-length string to split
    parts: list of (name, length) pairs
        Used to break up string, name '_' will be filtered from output.

    Returns
    -------
    Dict of name:contents of string at given location.
    """
def _handle_truncated_float_vec(vec, nbytes): ...
def _parse_float_vec(vec):
    """
    Parse a vector of float values representing IBM 8 byte floats into
    native 8 byte floats.
    """

class XportReader(ReaderBase, abc.Iterator):
    __doc__ = _xport_reader_doc
    _encoding: Incomplete
    _lines_read: int
    _index: Incomplete
    _chunksize: Incomplete
    handles: Incomplete
    filepath_or_buffer: Incomplete
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], index: Incomplete | None = None, encoding: str | None = 'ISO-8859-1', chunksize: int | None = None, compression: CompressionOptions = 'infer') -> None: ...
    def close(self) -> None: ...
    def _get_row(self): ...
    file_info: Incomplete
    member_info: Incomplete
    fields: Incomplete
    record_length: Incomplete
    record_start: Incomplete
    nobs: Incomplete
    columns: Incomplete
    _dtype: Incomplete
    def _read_header(self) -> None: ...
    def __next__(self) -> pd.DataFrame: ...
    def _record_count(self) -> int:
        """
        Get number of records in file.

        This is maybe suboptimal because we have to seek to the end of
        the file.

        Side effect: returns file position to record_start.
        """
    def get_chunk(self, size: int | None = None) -> pd.DataFrame:
        """
        Reads lines from Xport file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
    def _missing_double(self, vec): ...
    def read(self, nrows: int | None = None) -> pd.DataFrame: ...

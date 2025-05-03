import _abc
import collections.abc
import pandas.io.sas.sasreader
import pd as pd
from pandas.io.common import get_handle as get_handle
from pandas.io.sas.sasreader import ReaderBase as ReaderBase
from pandas.util._decorators import Appender as Appender
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

TYPE_CHECKING: bool
_correct_line1: str
_correct_header1: str
_correct_header2: str
_correct_obs_header: str
_fieldkeys: list
_base_params_doc: str
_params2_doc: str
_format_params_doc: str
_iterator_doc: str
_read_sas_doc: str
_xport_reader_doc: str
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

class XportReader(pandas.io.sas.sasreader.ReaderBase, collections.abc.Iterator):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], index, encoding: str | None = ..., chunksize: int | None, compression: CompressionOptions = ...) -> None: ...
    def close(self) -> None: ...
    def _get_row(self): ...
    def _read_header(self) -> None: ...
    def __next__(self) -> pd.DataFrame: ...
    def _record_count(self) -> int:
        """
        Get number of records in file.

        This is maybe suboptimal because we have to seek to the end of
        the file.

        Side effect: returns file position to record_start.
        """
    def get_chunk(self, size: int | None) -> pd.DataFrame:
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
    def read(self, nrows: int | None) -> pd.DataFrame:
        """Read observations from SAS Xport file, returning as data frame.

        Parameters
        ----------
        nrows : int
            Number of rows to read from data file; if None, read whole
            file.

        Returns
        -------
        A DataFrame.
        """

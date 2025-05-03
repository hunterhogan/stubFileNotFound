import numpy as np
import re
from _typeshed import Incomplete
from collections import abc
from collections.abc import Hashable, Iterator, Mapping, Sequence
from pandas import Index as Index, MultiIndex as MultiIndex
from pandas._libs import lib as lib
from pandas._typing import ArrayLike as ArrayLike, ReadCsvBuffer as ReadCsvBuffer, Scalar as Scalar
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_integer as is_integer, is_numeric_dtype as is_numeric_dtype
from pandas.core.dtypes.inference import is_dict_like as is_dict_like
from pandas.errors import EmptyDataError as EmptyDataError, ParserError as ParserError, ParserWarning as ParserWarning
from pandas.io.common import dedup_names as dedup_names, is_potential_multi_index as is_potential_multi_index
from pandas.io.parsers.base_parser import ParserBase as ParserBase, parser_defaults as parser_defaults
from pandas.util._decorators import cache_readonly as cache_readonly
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import IO, Literal

_BOM: str

class PythonParser(ParserBase):
    _no_thousands_columns: set[int]
    data: Iterator[str] | None
    buf: list
    pos: int
    line_pos: int
    skiprows: Incomplete
    skipfunc: Incomplete
    skipfooter: Incomplete
    delimiter: Incomplete
    quotechar: Incomplete
    escapechar: Incomplete
    doublequote: Incomplete
    skipinitialspace: Incomplete
    lineterminator: Incomplete
    quoting: Incomplete
    skip_blank_lines: Incomplete
    has_index_names: bool
    verbose: Incomplete
    thousands: Incomplete
    decimal: Incomplete
    comment: Incomplete
    _col_indices: list[int] | None
    orig_names: list[Hashable]
    _name_processed: bool
    index_names: Incomplete
    _parse_date_cols: Incomplete
    def __init__(self, f: ReadCsvBuffer[str] | list, **kwds) -> None:
        """
        Workhorse function for processing nested list into DataFrame
        """
    def num(self) -> re.Pattern: ...
    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]): ...
    _first_chunk: bool
    def read(self, rows: int | None = None) -> tuple[Index | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]]: ...
    def _exclude_implicit_index(self, alldata: list[np.ndarray]) -> tuple[Mapping[Hashable, np.ndarray], Sequence[Hashable]]: ...
    def get_chunk(self, size: int | None = None) -> tuple[Index | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]]: ...
    def _convert_data(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, ArrayLike]: ...
    def _have_mi_columns(self) -> bool: ...
    def _infer_columns(self) -> tuple[list[list[Scalar | None]], int, set[Scalar | None]]: ...
    def _header_line(self): ...
    def _handle_usecols(self, columns: list[list[Scalar | None]], usecols_key: list[Scalar | None], num_original_columns: int) -> list[list[Scalar | None]]:
        """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
    def _buffered_line(self) -> list[Scalar]:
        """
        Return a line from buffer, filling buffer if required.
        """
    def _check_for_bom(self, first_row: list[Scalar]) -> list[Scalar]:
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
    def _is_line_empty(self, line: list[Scalar]) -> bool:
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
    def _next_line(self) -> list[Scalar]: ...
    def _alert_malformed(self, msg: str, row_num: int) -> None:
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg: str
            The error message to display.
        row_num: int
            The row number where the parsing error occurred.
            Because this row number is displayed, we 1-index,
            even though we 0-index internally.
        """
    def _next_iter_line(self, row_num: int) -> list[Scalar] | None:
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
    def _check_comments(self, lines: list[list[Scalar]]) -> list[list[Scalar]]: ...
    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        '''
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : list of list of Scalars
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : list of list of Scalars
            The same array of lines with the "empty" ones removed.
        '''
    def _check_thousands(self, lines: list[list[Scalar]]) -> list[list[Scalar]]: ...
    def _search_replace_num_columns(self, lines: list[list[Scalar]], search: str, replace: str) -> list[list[Scalar]]: ...
    def _check_decimal(self, lines: list[list[Scalar]]) -> list[list[Scalar]]: ...
    def _clear_buffer(self) -> None: ...
    index_col: Incomplete
    num_original_columns: Incomplete
    _implicit_index: bool
    def _get_index_name(self) -> tuple[Sequence[Hashable] | None, list[Hashable], list[Hashable]]:
        """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
    def _rows_to_cols(self, content: list[list[Scalar]]) -> list[np.ndarray]: ...
    def _get_lines(self, rows: int | None = None) -> list[list[Scalar]]: ...
    def _remove_skipped_rows(self, new_rows: list[list[Scalar]]) -> list[list[Scalar]]: ...
    def _set_no_thousand_columns(self) -> set[int]: ...

class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """
    f: Incomplete
    buffer: Iterator | None
    delimiter: Incomplete
    comment: Incomplete
    colspecs: Incomplete
    def __init__(self, f: IO[str] | ReadCsvBuffer[str], colspecs: list[tuple[int, int]] | Literal['infer'], delimiter: str | None, comment: str | None, skiprows: set[int] | None = None, infer_nrows: int = 100) -> None: ...
    def get_rows(self, infer_nrows: int, skiprows: set[int] | None = None) -> list[str]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
    def detect_colspecs(self, infer_nrows: int = 100, skiprows: set[int] | None = None) -> list[tuple[int, int]]: ...
    def __next__(self) -> list[str]: ...

class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """
    colspecs: Incomplete
    infer_nrows: Incomplete
    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None: ...
    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> FixedWidthReader: ...
    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """

def count_empty_vals(vals) -> int: ...
def _validate_skipfooter_arg(skipfooter: int) -> int:
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : non-negative integer
        The number of rows to skip at the end of the file.

    Returns
    -------
    validated_skipfooter : non-negative integer
        The original input if the validation succeeds.

    Raises
    ------
    ValueError : 'skipfooter' was not a non-negative integer.
    """

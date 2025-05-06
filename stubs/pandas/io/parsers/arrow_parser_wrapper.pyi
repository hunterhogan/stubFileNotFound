from _typeshed import Incomplete
from pandas import DataFrame as DataFrame
from pandas._typing import ReadBuffer as ReadBuffer
from pandas.errors import ParserError as ParserError, ParserWarning as ParserWarning
from pandas.io._util import _arrow_dtype_mapping as _arrow_dtype_mapping, arrow_string_types_mapper as arrow_string_types_mapper
from pandas.io.parsers.base_parser import ParserBase as ParserBase

class ArrowParserWrapper(ParserBase):
    """
    Wrapper for the pyarrow engine for read_csv()
    """
    kwds: Incomplete
    src: Incomplete
    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None: ...
    encoding: Incomplete
    na_values: Incomplete
    def _parse_kwds(self) -> None:
        """
        Validates keywords before passing to pyarrow.
        """
    parse_options: Incomplete
    convert_options: Incomplete
    read_options: Incomplete
    def _get_pyarrow_options(self) -> None:
        """
        Rename some arguments to pass to pyarrow
        """
    names: Incomplete
    dtype: Incomplete
    def _finalize_pandas_output(self, frame: DataFrame) -> DataFrame:
        """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame: DataFrame
            The DataFrame to process.

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
    def _validate_usecols(self, usecols) -> None: ...
    def read(self) -> DataFrame:
        """
        Reads the contents of a CSV file into a DataFrame and
        processes it according to the kwargs passed in the
        constructor.

        Returns
        -------
        DataFrame
            The DataFrame created from the CSV file.
        """

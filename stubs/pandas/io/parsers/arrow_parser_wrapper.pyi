import pandas as pd
import pandas._libs.lib as lib
import pandas.io.parsers.base_parser
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._libs.lib import is_integer as is_integer
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.errors import ParserError as ParserError, ParserWarning as ParserWarning
from pandas.io._util import _arrow_dtype_mapping as _arrow_dtype_mapping, arrow_string_types_mapper as arrow_string_types_mapper
from pandas.io.parsers.base_parser import ParserBase as ParserBase
from pandas.util._exceptions import find_stack_level as find_stack_level

TYPE_CHECKING: bool

class ArrowParserWrapper(pandas.io.parsers.base_parser.ParserBase):
    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None: ...
    def _parse_kwds(self) -> None:
        """
        Validates keywords before passing to pyarrow.
        """
    def _get_pyarrow_options(self) -> None:
        """
        Rename some arguments to pass to pyarrow
        """
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

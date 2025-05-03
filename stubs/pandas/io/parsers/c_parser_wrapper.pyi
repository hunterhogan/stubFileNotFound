import pandas._libs.lib as lib
import pandas._libs.parsers as parsers
import pandas.io.parsers.base_parser
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.dtypes.concat import concat_compat as concat_compat, union_categoricals as union_categoricals
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.indexes.base import ensure_index_from_sequences as ensure_index_from_sequences
from pandas.errors import DtypeWarning as DtypeWarning, ParserError as ParserError
from pandas.io.common import dedup_names as dedup_names, is_potential_multi_index as is_potential_multi_index
from pandas.io.parsers.base_parser import ParserBase as ParserBase, is_index_col as is_index_col
from pandas.util._exceptions import find_stack_level as find_stack_level

TYPE_CHECKING: bool

class CParserWrapper(pandas.io.parsers.base_parser.ParserBase):
    def __init__(self, src: ReadCsvBuffer[str], **kwds) -> None: ...
    def close(self) -> None: ...
    def _set_noconvert_columns(self) -> None:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
    def read(self, nrows: int | None) -> tuple[Index | MultiIndex | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]]: ...
    def _filter_usecols(self, names: Sequence[Hashable]) -> Sequence[Hashable]: ...
    def _maybe_parse_dates(self, values, index: int, try_parse_dates: bool = ...): ...
def _concatenate_chunks(chunks: list[dict[int, ArrayLike]]) -> dict:
    """
    Concatenate chunks of data read with low_memory=True.

    The tricky part is handling Categoricals, where different chunks
    may have different inferred categories.
    """
def ensure_dtype_objs(dtype: DtypeArg | dict[Hashable, DtypeArg] | None) -> DtypeObj | dict[Hashable, DtypeObj] | None:
    """
    Ensure we have either None, a dtype object, or a dictionary mapping to
    dtype objects.
    """

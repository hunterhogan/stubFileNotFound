from _typeshed import Incomplete
from collections.abc import Hashable, Mapping, Sequence
from pandas import Index as Index, MultiIndex as MultiIndex
from pandas._libs import lib as lib, parsers as parsers
from pandas._typing import ArrayLike as ArrayLike, DtypeArg as DtypeArg, DtypeObj as DtypeObj, ReadCsvBuffer as ReadCsvBuffer
from pandas.core.dtypes.concat import concat_compat as concat_compat, union_categoricals as union_categoricals
from pandas.io.common import dedup_names as dedup_names, is_potential_multi_index as is_potential_multi_index
from pandas.io.parsers.base_parser import ParserBase as ParserBase, ParserError as ParserError, is_index_col as is_index_col

class CParserWrapper(ParserBase):
    low_memory: bool
    _reader: parsers.TextReader
    kwds: Incomplete
    unnamed_cols: Incomplete
    names: Incomplete
    orig_names: Incomplete
    _name_processed: bool
    index_names: Incomplete
    _implicit_index: Incomplete
    def __init__(self, src: ReadCsvBuffer[str], **kwds) -> None: ...
    def close(self) -> None: ...
    def _set_noconvert_columns(self) -> None:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
    _first_chunk: bool
    def read(self, nrows: int | None = None) -> tuple[Index | MultiIndex | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]]: ...
    def _filter_usecols(self, names: Sequence[Hashable]) -> Sequence[Hashable]: ...
    def _maybe_parse_dates(self, values, index: int, try_parse_dates: bool = True): ...

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

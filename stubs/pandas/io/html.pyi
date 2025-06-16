from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
from re import Pattern
from typing import (
    Any,
    Literal,
)

from pandas.core.frame import DataFrame

from pandas._libs.lib import NoDefault
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT1,
    HashableT2,
    HashableT3,
    HashableT4,
    HashableT5,
    HTMLFlavors,
    ReadBuffer,
    StorageOptions,
)

def read_html(
    io: FilePath | ReadBuffer[str],
    *,
    match: str | Pattern = '.+',
    flavor: HTMLFlavors | Sequence[HTMLFlavors] | None = None,
    header: int | Sequence[int] | None = None,
    index_col: int | Sequence[int] | list[HashableT1] | None = None,
    skiprows: int | Sequence[int] | slice | None = None,
    attrs: dict[str, str] | None = None,
    parse_dates: (
        bool
        | Sequence[int]
        | list[HashableT2]  # Cannot be Sequence[Hashable] to prevent str
        | Sequence[Sequence[Hashable]]
        | dict[str, Sequence[int]]
        | dict[str, list[HashableT3]]
    ) = False,
    thousands: str = ',',
    encoding: str | None = None,
    decimal: str = '.',
    converters: Mapping[int | HashableT4, Callable[[str], Any]] | None = None,
    na_values: (
        str | list[str] | dict[HashableT5, str] | dict[HashableT5, list[str]] | None
    ) = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
    extract_links: Literal["header", "footer", "body", "all"] | None = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    storage_options: StorageOptions = None,
) -> list[DataFrame]: ...

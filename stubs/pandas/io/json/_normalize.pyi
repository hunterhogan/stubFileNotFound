from pandas import DataFrame

from pandas._typing import IgnoreRaise
from typing import Any

def json_normalize(
    data: dict[Any, Any] | list[dict],
    record_path: str | list[Any] | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: IgnoreRaise = 'raise',
    sep: str = '.',
    max_level: int | None = None,
) -> DataFrame: ...

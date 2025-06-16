from typing import (
    Any,
    Literal,
)

from pandas.core.frame import DataFrame

def read_gbq(
    query: str,
    project_id: str | None = None,
    index_col: str | None = None,
    col_order: list[str] | None = None,
    reauth: bool = False,
    auth_local_webserver: bool = True,
    dialect: Literal["legacy", "standard"] | None = None,
    location: str | None = None,
    configuration: dict[str, Any] | None = None,
    # Google type, not available
    credentials: Any = None,
    use_bqstorage_api: bool | None = None,
    max_results: int | None = None,
    progress_bar_type: Literal["tqdm", "tqdm_notebook", "tqdm_gui"] | None = None,
) -> DataFrame: ...

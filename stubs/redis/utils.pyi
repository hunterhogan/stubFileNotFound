from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from redis.exceptions import DataError as DataError
from redis.typing import AbsExpiryT as AbsExpiryT, EncodableT as EncodableT, ExpiryT as ExpiryT
from typing import Any, Mapping

hiredis_version: Incomplete
HIREDIS_AVAILABLE: Incomplete
SSL_AVAILABLE: bool
CRYPTOGRAPHY_AVAILABLE: bool

def from_url(url, **kwargs):
    """
    Returns an active Redis client generated from the given database URL.

    Will attempt to extract the database id from the path url fragment, if
    none is provided.
    """
@contextmanager
def pipeline(redis_obj) -> Generator[Incomplete]: ...
def str_if_bytes(value: str | bytes) -> str: ...
def safe_str(value): ...
def dict_merge(*dicts: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge all provided dicts into 1 dict.
    *dicts : `dict`
        dictionaries to merge
    """
def list_keys_to_dict(key_list, callback): ...
def merge_result(command, res):
    """
    Merge all items in `res` into a list.

    This command is used when sending a command to multiple nodes
    and the result from each node should be merged into a single list.

    res : 'dict'
    """
def warn_deprecated(name, reason: str = '', version: str = '', stacklevel: int = 2) -> None: ...
def deprecated_function(reason: str = '', version: str = '', name: Incomplete | None = None):
    """
    Decorator to mark a function as deprecated.
    """
def warn_deprecated_arg_usage(arg_name: list | str, function_name: str, reason: str = '', version: str = '', stacklevel: int = 2): ...
def deprecated_args(args_to_warn: list = ['*'], allowed_args: list = [], reason: str = '', version: str = ''):
    """
    Decorator to mark specified args of a function as deprecated.
    If '*' is in args_to_warn, all arguments will be marked as deprecated.
    """
def _set_info_logger() -> None:
    """
    Set up a logger that log info logs to stdout.
    (This is used by the default push response handler)
    """
def get_lib_version(): ...
def format_error_message(host_error: str, exception: BaseException) -> str: ...
def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two versions.

    :return: -1 if version1 > version2
             0 if both versions are equal
             1 if version1 < version2
    """
def ensure_string(key): ...
def extract_expire_flags(ex: ExpiryT | None = None, px: ExpiryT | None = None, exat: AbsExpiryT | None = None, pxat: AbsExpiryT | None = None) -> list[EncodableT]: ...
def truncate_text(txt, max_length: int = 100): ...

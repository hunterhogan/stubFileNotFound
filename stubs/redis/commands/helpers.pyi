from redis.typing import KeysT as KeysT, KeyT as KeyT
from typing import Any

def list_or_args(keys: KeysT, args: tuple[KeyT, ...]) -> list[KeyT]: ...
def nativestr(x: Any) -> Any:
    """Return the decoded binary string, or a string, depending on type."""
def delist(x: Any) -> Any:
    """Given a list of binaries, return the stringified version."""
def parse_to_list(response: Any) -> Any:
    """Optimistically parse the response to a list."""
def parse_list_to_dict(response: Any) -> Any: ...
def random_string(length: int = 10) -> Any:
    """
    Returns a random N character long string.
    """
def decode_dict_keys(obj: Any) -> Any:
    """Decode the keys of the given dictionary with utf-8."""
def get_protocol_version(client: Any) -> Any: ...

from redis.typing import KeyT as KeyT, KeysT as KeysT

def list_or_args(keys: KeysT, args: tuple[KeyT, ...]) -> list[KeyT]: ...
def nativestr(x):
    """Return the decoded binary string, or a string, depending on type."""
def delist(x):
    """Given a list of binaries, return the stringified version."""
def parse_to_list(response):
    """Optimistically parse the response to a list."""
def parse_list_to_dict(response): ...
def random_string(length: int = 10):
    """
    Returns a random N character long string.
    """
def decode_dict_keys(obj):
    """Decode the keys of the given dictionary with utf-8."""
def get_protocol_version(client): ...

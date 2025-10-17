from _typeshed import Incomplete
from typing import Any

class NumberEncoder:
    """
    A very simple encoder between lines and strings.

    Example:

        >>> my_encoder = number_encoding.NumberEncoder('isogram')
        >>> my_encoder.encode(10000)
        'rssir'
        >>> my_encoder.encode(10000000)
        'saimmmgrg'
        >>> my_encoder.decode('saimmmgrg')
        10000000

    """

    characters: Incomplete
    def __init__(self, characters: Any) -> None: ...
    def encode(self, number: Any, minimum_length: int = 1) -> Any:
        r"""
        Encode the number into a string.

        If `minimum_length > 1`, the string will be padded (with the "zero"
        character) if the number isn\'t big enough.
        """
    def decode(self, string: Any) -> Any:
        """Decode `string` into a number."""




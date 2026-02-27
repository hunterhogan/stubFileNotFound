
from typing import Tuple, Union
import os

class PNG(bytes):
    SIGNATURE = ...
    def __new__(cls, *args, **kwargs): # -> Self:
        ...

    @property
    def size(self) -> tuple[int, int]:
        ...

    @classmethod
    def read_from(cls, path: str | os.PathLike) -> PNG:
        ...


from babelfont.Font import Font
from babelfont.fontFilters import parse_filter
import importlib
import inspect
import logging
import os
import pkgutil
import sys

logger = ...
class BaseConvertor:
    filename: str
    scratch: object
    font: Font
    compile_only: bool
    suffix = ...
    LOAD_FILTERS = ...
    COMPILE_FILTERS = ...
    SAVE_FILTERS = ...
    def __init__(self) -> None:
        ...

    @classmethod
    def can_load(cls, other, **kwargs):
        ...

    @classmethod
    def can_save(cls, other, **kwargs):
        ...

    @classmethod
    def load(cls, convertor, compile_only=..., filters=...):
        ...

    @classmethod
    def save(cls, obj, convertor, **kwargs):
        ...



class Convert:
    convertors = ...
    def __init__(self, filename) -> None:
        ...

    def load_convertor(self, **kwargs): # -> None:
        ...

    def save_convertor(self, **kwargs): # -> None:
        ...

    def load(self, **kwargs):
        ...

    def save(self, obj, **kwargs):
        ...

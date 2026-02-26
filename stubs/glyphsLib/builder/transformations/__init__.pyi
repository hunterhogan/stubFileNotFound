from . import propagate_anchors as propagate_anchors
from _typeshed import Incomplete
from glyphsLib.builder.transformations.align_alternate_layers import align_alternate_layers as align_alternate_layers
from glyphsLib.builder.transformations.propagate_anchors import propagate_all_anchors as propagate_all_anchors
from typing import ClassVar

TRANSFORMATIONS: list

class _CustomParameter(tuple):
    """_CustomParameter(name, default)"""
    _fields: ClassVar[tuple] = ...
    _field_defaults: ClassVar[dict] = ...
    __match_args__: ClassVar[tuple] = ...
    __classdictcell__: ClassVar[cell] = ...
    __orig_bases__: ClassVar[tuple] = ...
    name: Incomplete
    default: Incomplete
    def __init__(self, _cls, name: str, default: bool) -> None:
        """Create new instance of _CustomParameter(name, default)"""
    @classmethod
    def _make(cls, iterable):
        """Make a new _CustomParameter object from a sequence or iterable"""
    def __replace__(self, **kwds):
        """Return a new _CustomParameter object replacing specified fields with new values"""
    def _replace(self, **kwds):
        """Return a new _CustomParameter object replacing specified fields with new values"""
    def _asdict(self):
        """Return a new dict which maps field names to their values."""
    def __getnewargs__(self):
        """Return self as a plain tuple.  Used by copy and pickle."""
    def __annotate_func__(self, format): ...
TRANSFORMATION_CUSTOM_PARAMS: mappingproxy

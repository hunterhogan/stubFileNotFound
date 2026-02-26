from .align_alternate_layers import align_alternate_layers as align_alternate_layers
from .propagate_anchors import propagate_all_anchors as propagate_all_anchors
from _typeshed import Incomplete
from typing import NamedTuple

TRANSFORMATIONS: Incomplete

class _CustomParameter(NamedTuple):
    name: str
    default: bool

TRANSFORMATION_CUSTOM_PARAMS: Incomplete

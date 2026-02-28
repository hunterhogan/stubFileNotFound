from enum import IntEnum
from glyphsLib.classes import GSLayer as GSLayer

class Pole(IntEnum):
    MIN = 1
    MAX = 2

def normalized_location(layer, base_layer): ...
def variation_model(glyph, smart_layers, layer): ...
def get_coordinates(layer): ...
def set_coordinates(layer, coords) -> None: ...
def decompose_smart_components_in_layer(self, layer):
    """Decompose any smart components in a layer, returning a new layer with paths.

    This recursively instantiates nested smart components before extracting
    coordinates for interpolation.

    See https://github.com/googlefonts/glyphsLib/issues/1111
    """
def instantiate_smart_component(self, layer, component, pen) -> None:
    """Instantiate a smart component by interpolating and drawing to a pointPen."""

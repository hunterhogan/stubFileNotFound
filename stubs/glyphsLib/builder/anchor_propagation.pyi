COMPONENT_INFO_KEY: str
def to_ufo_propagate_font_anchors(self, ufo):
    """Copy anchors from parent glyphs' components to the parent."""
def _propagate_glyph_anchors(self, ufo, parent, processed):
    """Propagate anchors for a single parent glyph."""
def _get_anchor_data(anchor_data, ufo, components, anchor_name):
    """Get data for an anchor from a list of components."""
def _componentAnchorFromLib(_glyph, _targetComponent):
    """Pull component’s named anchor from Glyph.lib"""
def _adjust_anchors(anchor_data, ufo, parent, component):
    """Adjust anchors to which a mark component may have been attached."""
def _is_ligature_mark(glyph): ...
def _component_closest_to_origin(components, glyph_set):
    """Return the component whose (xmin, ymin) bounds are closest to origin.

    This ensures that a component that is moved below another is
    actually recognized as such. Looking only at the transformation
    offset can be misleading.
    """
def _distance(pos1, pos2): ...
def _bounds(component, glyph_set):
    """Return the (xmin, ymin) of the bounds of `component`."""

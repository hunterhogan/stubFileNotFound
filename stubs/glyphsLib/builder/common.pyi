
UFO_FORMAT: str
def to_ufo_time(datetime_obj):
    """Format a datetime object as specified for UFOs."""
def from_ufo_time(string):
    """Parses a datetime as specified for UFOs into a datetime object."""
def from_loose_ufo_time(string):
    """Parses a datetime as specified for UFOs into a datetime object,
    or as the Glyphs formet.
    """
def to_ufo_color(color): ...

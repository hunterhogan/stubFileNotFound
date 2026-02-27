from typing import Literal

class FontParseError(Exception):
    ...


def get_font_format(font_file_path) -> Literal['UFO', 'OTF', 'CFF', 'PFB', 'PFA', 'PFC'] | None:
    ...

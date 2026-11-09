from typing import TypedDict, Unpack
import io
import os

class UserOptions(TypedDict, total=False):
    in_file: str | bytes | os.PathLike[str] | io.BufferedIOBase | None
    in_buffer: bytes | None
    out_file: str | bytes | os.PathLike[str] | io.BufferedIOBase | None
    control_file: str | os.PathLike[str] | None
    control_buffer: str | bytes | None
    reference_file: str | os.PathLike[str] | None
    reference_buffer: bytes | None
    reference_index: int
    hinting_range_min: int
    hinting_range_max: int
    hinting_limit: int
    hint_composites: bool
    adjust_subglyphs: bool
    increase_x_height: int
    x_height_snapping_exceptions: str
    windows_compatibility: bool
    default_script: str
    fallback_script: str
    fallback_scaling: bool
    symbol: bool
    fallback_stem_width: int | None
    ignore_restrictions: bool
    family_suffix: str | None
    detailed_info: bool
    no_info: bool
    TTFA_info: bool
    dehint: bool
    epoch: int | None
    debug: bool
    verbose: bool
    gray_stem_width_mode: int
    gdi_cleartype_stem_width_mode: int
    dw_cleartype_stem_width_mode: int


USER_OPTIONS = ...
StemWidthMode = ...
STEM_WIDTH_MODE_OPTIONS = ...
STRONG_STEM_WIDTH_OPTIONS = ...
def validate_options(kwargs): # -> dict[str, Any]:
    ...

SPECIAL_OPT_NAMES = ...
def format_kwargs(**options: Unpack[UserOptions]) -> list[str]:
    ...

def strong_stem_width(s): # -> dict[Any, Any]:
    ...

def stem_width_mode(s): # -> dict[Any, Any]:
    ...

def format_stem_width_modes(gray_stem_width_mode, gdi_cleartype_stem_width_mode, dw_cleartype_stem_width_mode):
    ...

def stdin_or_input_path_type(s): # -> BufferedReader[_BufferedReaderStream] | None:
    ...

def stdout_or_output_path_type(s): # -> BufferedWriter | None:
    ...

def parse_args(args=..., splitfunc=...): # -> dict[str, Any] | None:
    """Parse command line arguments and return a dictionary of options
    for ttfautohint.ttfautohint function.

    `args` can be either None, a list of strings, or a single string,
    that is split into individual options with `shlex.split`.

    When `args` is None, the console's default sys.argv are used, and any
    SystemExit exceptions raised by argparse are propagated.

    If args is a string list or a string, it is assumed that the function
    was not called from a console script's `main` entry point, but from
    other client code, and thus the SystemExit exceptions are muted and
    a `None` value is returned.
    """

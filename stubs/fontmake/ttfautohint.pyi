from fontmake.errors import FontmakeError as FontmakeError, TTFAError as TTFAError

def _which_ttfautohint() -> list[str] | None: ...
def ttfautohint(in_file, out_file, args=None, **kwargs) -> None:
    """Thin wrapper around the ttfautohint command line tool.

    Can take in command line arguments directly as a string, or spelled out as
    Python keyword arguments.
    """

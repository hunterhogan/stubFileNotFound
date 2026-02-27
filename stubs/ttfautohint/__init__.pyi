
from contextlib import ExitStack
from importlib.resources import as_file, files, is_resource
from ttfautohint._version import __version__
from ttfautohint.errors import TAError
from ttfautohint.options import format_kwargs, StemWidthMode, validate_options
import atexit
import io
import os
import stat
import subprocess
import sys

__all__ = ["StemWidthMode", "TAError", "__version__", "run", "ttfautohint"]
_exit_stack = ...
_exe_basename = ...
if sys.platform == "win32":
    ...
_exe_full_path = ...
def run(args, **kwargs):
    """Run the 'ttfautohint' executable with the list of positional arguments.

    All keyword arguments are forwarded to subprocess.run function.

    The bundled copy of the 'ttfautohint' executable is tried first; if this
    was not included at installation, the version which is on $PATH is used.

    Return:
        subprocess.CompletedProcess object with the following attributes:
        args, returncode, stdout, stderr.
    """

def ttfautohint(**kwargs):
    ...

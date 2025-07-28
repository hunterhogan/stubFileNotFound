from .util import CompileError as CompileError, find_binary_of_command as find_binary_of_command, unique_list as unique_list
from _typeshed import Incomplete
from typing import Callable

class CompilerRunner:
    """ CompilerRunner base class.

    Parameters
    ==========

    sources : list of str
        Paths to sources.
    out : str
    flags : iterable of str
        Compiler flags.
    run_linker : bool
    compiler_name_exe : (str, str) tuple
        Tuple of compiler name &  command to call.
    cwd : str
        Path of root of relative paths.
    include_dirs : list of str
        Include directories.
    libraries : list of str
        Libraries to link against.
    library_dirs : list of str
        Paths to search for shared libraries.
    std : str
        Standard string, e.g. ``'c++11'``, ``'c99'``, ``'f2003'``.
    define: iterable of strings
        macros to define
    undef : iterable of strings
        macros to undefine
    preferred_vendor : string
        name of preferred vendor e.g. 'gnu' or 'intel'

    Methods
    =======

    run():
        Invoke compilation as a subprocess.

    """
    environ_key_compiler: str
    environ_key_flags: str
    environ_key_ldflags: str
    compiler_dict: dict[str, str]
    standards: tuple[None | str, ...]
    std_formater: dict[str, Callable[[str | None], str]]
    compiler_name_vendor_mapping: dict[str, str]
    sources: Incomplete
    out: Incomplete
    flags: Incomplete
    cwd: Incomplete
    compiler_binary: Incomplete
    compiler_vendor: Incomplete
    compiler_name: Incomplete
    define: Incomplete
    undef: Incomplete
    include_dirs: Incomplete
    libraries: Incomplete
    library_dirs: Incomplete
    std: Incomplete
    run_linker: Incomplete
    linkline: Incomplete
    def __init__(self, sources, out, flags=None, run_linker: bool = True, compiler=None, cwd: str = '.', include_dirs=None, libraries=None, library_dirs=None, std=None, define=None, undef=None, strict_aliasing=None, preferred_vendor=None, linkline=None, **kwargs) -> None: ...
    @classmethod
    def find_compiler(cls, preferred_vendor=None):
        """ Identify a suitable C/fortran/other compiler. """
    def cmd(self):
        """ List of arguments (str) to be passed to e.g. ``subprocess.Popen``. """
    cmd_outerr: Incomplete
    cmd_returncode: Incomplete
    def run(self): ...

class CCompilerRunner(CompilerRunner):
    environ_key_compiler: str
    environ_key_flags: str
    compiler_dict: Incomplete
    standards: Incomplete
    std_formater: Incomplete
    compiler_name_vendor_mapping: Incomplete

def _mk_flag_filter(cmplr_name): ...

class CppCompilerRunner(CompilerRunner):
    environ_key_compiler: str
    environ_key_flags: str
    compiler_dict: Incomplete
    standards: Incomplete
    std_formater: Incomplete
    compiler_name_vendor_mapping: Incomplete

class FortranCompilerRunner(CompilerRunner):
    environ_key_compiler: str
    environ_key_flags: str
    standards: Incomplete
    std_formater: Incomplete
    compiler_dict: Incomplete
    compiler_name_vendor_mapping: Incomplete

import ctypes
from _typeshed import Incomplete
from llvmlite.binding import ffi as ffi, targets as targets

class _LinkElement(ctypes.Structure):
    _fields_: Incomplete

class _SymbolAddress(ctypes.Structure):
    _fields_: Incomplete

class JITLibraryBuilder:
    """
    Create a library for linking by OrcJIT

    OrcJIT operates like a linker: a number of compilation units and
    dependencies are collected together and linked into a single dynamic library
    that can export functions to other libraries or to be consumed directly as
    entry points into JITted code. The native OrcJIT has a lot of memory
    management complications so this API is designed to work well with Python's
    garbage collection.

    The creation of a new library is a bit like a linker command line where
    compilation units, mostly as LLVM IR, and previously constructed libraries
    are linked together, then loaded into memory, and the addresses of exported
    symbols are extracted. Any static initializers are run and the exported
    addresses and a resource tracker is produced. As long as the resource
    tracker is referenced somewhere in Python, the exported addresses will be
    valid. Once the resource tracker is garbage collected, the static
    destructors will run and library will be unloaded from memory.
    """
    __entries: Incomplete
    __exports: Incomplete
    __imports: Incomplete
    def __init__(self) -> None: ...
    def add_ir(self, llvmir):
        """
        Adds a compilation unit to the library using LLVM IR as the input
        format.

        This takes a string or an object that can be converted to a string,
        including IRBuilder, that contains LLVM IR.
        """
    def add_native_assembly(self, asm):
        """
        Adds a compilation unit to the library using native assembly as the
        input format.

        This takes a string or an object that can be converted to a string that
        contains native assembly, which will be
        parsed by LLVM.
        """
    def add_object_img(self, data):
        """
        Adds a compilation unit to the library using pre-compiled object code.

        This takes the bytes of the contents of an object artifact which will be
        loaded by LLVM.
        """
    def add_object_file(self, file_path):
        """
        Adds a compilation unit to the library using pre-compiled object file.

        This takes a string or path-like object that references an object file
        which will be loaded by LLVM.
        """
    def add_jit_library(self, name):
        """
        Adds an existing JIT library as prerequisite.

        The name of the library must match the one provided in a previous link
        command.
        """
    def add_current_process(self):
        """
        Allows the JITted library to access symbols in the current binary.

        That is, it allows exporting the current binary's symbols, including
        loaded libraries, as imports to the JITted
        library.
        """
    def import_symbol(self, name, address):
        """
        Register the *address* of global symbol *name*.  This will make
        it usable (e.g. callable) from LLVM-compiled functions.
        """
    def export_symbol(self, name):
        """
        During linking, extract the address of a symbol that was defined in one
        of the compilation units.

        This allows getting symbols, functions or global variables, out of the
        JIT linked library. The addresses will be
        available when the link method is called.
        """
    def link(self, lljit, library_name):
        """
        Link all the current compilation units into a JITted library and extract
        the address of exported symbols.

        An instance of the OrcJIT instance must be provided and this will be the
        scope that is used to find other JITted libraries that are dependencies
        and also be the place where this library will be defined.

        After linking, the method will return a resource tracker that keeps the
        library alive. This tracker also knows the addresses of any exported
        symbols that were requested.

        The addresses will be valid as long as the resource tracker is
        referenced.

        When the resource tracker is destroyed, the library will be cleaned up,
        however, the name of the library cannot be reused.
        """

class ResourceTracker(ffi.ObjectRef):
    '''
    A resource tracker is created for each loaded JIT library and keeps the
    module alive.

    OrcJIT supports unloading libraries that are no longer used. This resource
    tracker should be stored in any object that reference functions or constants
    for a JITted library. When all references to the resource tracker are
    dropped, this will trigger LLVM to unload the library and destroy any
    functions.

    Failure to keep resource trackers while calling a function or accessing a
    symbol can result in crashes or memory corruption.

    LLVM internally tracks references between different libraries, so only
    "leaf" libraries need to be tracked.
    '''
    __addresses: Incomplete
    __name: Incomplete
    def __init__(self, ptr, name, addresses) -> None: ...
    def __getitem__(self, item):
        """
        Get the address of an exported symbol as an integer
        """
    @property
    def name(self): ...
    def _dispose(self) -> None: ...

class LLJIT(ffi.ObjectRef):
    '''
    A OrcJIT-based LLVM JIT engine that can compile and run LLVM IR as a
    collection of JITted dynamic libraries

    The C++ OrcJIT API has a lot of memory ownership patterns that do not work
    with Python. This API attempts to provide ones that are safe at the expense
    of some features. Each LLJIT instance is a collection of JIT-compiled
    libraries. In the C++ API, there is a "main" library; this API does not
    provide access to the main library. Use the JITLibraryBuilder to create a
    new named library instead.
    '''
    _td: Incomplete
    def __init__(self, ptr) -> None: ...
    def lookup(self, dylib, fn):
        """
        Find a function in this dynamic library and construct a new tracking
        object for it

        If the library or function do not exist, an exception will occur.

        Parameters
        ----------
        dylib : str or None
           the name of the library containing the symbol
        fn : str
           the name of the function to get
        """
    @property
    def target_data(self):
        """
        The TargetData for this LLJIT instance.
        """
    def _dispose(self) -> None: ...

def create_lljit_compiler(target_machine: Incomplete | None = None, *, use_jit_link: bool = False, suppress_errors: bool = False):
    """
    Create an LLJIT instance
    """

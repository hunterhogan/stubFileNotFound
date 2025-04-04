from _typeshed import Incomplete
from llvmlite import opaque_pointers_enabled as opaque_pointers_enabled
from llvmlite.binding import ffi as ffi
from llvmlite.binding.common import _decode_string as _decode_string, _encode_string as _encode_string
from llvmlite.binding.initfini import llvm_version_info as llvm_version_info
from typing import NamedTuple

class Triple(NamedTuple):
    Arch: Incomplete
    SubArch: Incomplete
    Vendor: Incomplete
    OS: Incomplete
    Env: Incomplete
    ObjectFormat: Incomplete

def get_process_triple():
    """
    Return a target triple suitable for generating code for the current process.
    An example when the default triple from ``get_default_triple()`` is not be
    suitable is when LLVM is compiled for 32-bit but the process is executing
    in 64-bit mode.
    """
def get_triple_parts(triple: str):
    """
    Return a tuple of the parts of the given triple.
    """

class FeatureMap(dict):
    """
    Maps feature name to a boolean indicating the availability of the feature.
    Extends ``dict`` to add `.flatten()` method.
    """
    def flatten(self, sort: bool = True):
        """
        Args
        ----
        sort: bool
            Optional.  If True, the features are sorted by name; otherwise,
            the ordering is unstable between python session due to hash
            randomization.  Defaults to True.

        Returns a string suitable for use as the ``features`` argument to
        ``Target.create_target_machine()``.

        """

def get_host_cpu_features():
    '''
    Returns a dictionary-like object indicating the CPU features for current
    architecture and whether they are enabled for this CPU.  The key-value pairs
    are the feature name as string and a boolean indicating whether the feature
    is available.  The returned value is an instance of ``FeatureMap`` class,
    which adds a new method ``.flatten()`` for returning a string suitable for
    use as the "features" argument to ``Target.create_target_machine()``.

    If LLVM has not implemented this feature or it fails to get the information,
    this function will raise a RuntimeError exception.
    '''
def get_default_triple():
    """
    Return the default target triple LLVM is configured to produce code for.
    """
def get_host_cpu_name():
    """
    Get the name of the host's CPU, suitable for using with
    :meth:`Target.create_target_machine()`.
    """

llvm_version_major: Incomplete
_object_formats: Incomplete

def get_object_format(triple: Incomplete | None = None):
    """
    Get the object format for the given *triple* string (or the default
    triple if omitted).
    A string is returned
    """
def create_target_data(layout):
    """
    Create a TargetData instance for the given *layout* string.
    """

class TargetData(ffi.ObjectRef):
    """
    A TargetData provides structured access to a data layout.
    Use :func:`create_target_data` to create instances.
    """
    def __str__(self) -> str: ...
    def _dispose(self) -> None: ...
    def get_abi_size(self, ty):
        """
        Get ABI size of LLVM type *ty*.
        """
    def get_element_offset(self, ty, position):
        """
        Get byte offset of type's ty element at the given position
        """
    def get_abi_alignment(self, ty):
        """
        Get minimum ABI alignment of LLVM type *ty*.
        """
    def get_pointee_abi_size(self, ty):
        """
        Get ABI size of pointee type of LLVM pointer type *ty*.
        """
    def get_pointee_abi_alignment(self, ty):
        """
        Get minimum ABI alignment of pointee type of LLVM pointer type *ty*.
        """

RELOC: Incomplete
CODEMODEL: Incomplete

class Target(ffi.ObjectRef):
    _triple: str
    @classmethod
    def from_default_triple(cls):
        """
        Create a Target instance for the default triple.
        """
    @classmethod
    def from_triple(cls, triple):
        """
        Create a Target instance for the given triple (a string).
        """
    @property
    def name(self): ...
    @property
    def description(self): ...
    @property
    def triple(self): ...
    def __str__(self) -> str: ...
    def create_target_machine(self, cpu: str = '', features: str = '', opt: int = 2, reloc: str = 'default', codemodel: str = 'jitdefault', printmc: bool = False, jit: bool = False, abiname: str = ''):
        '''
        Create a new TargetMachine for this target and the given options.

        Specifying codemodel=\'default\' will result in the use of the "small"
        code model. Specifying codemodel=\'jitdefault\' will result in the code
        model being picked based on platform bitness (32="small", 64="large").

        The `printmc` option corresponds to llvm\'s `-print-machineinstrs`.

        The `jit` option should be set when the target-machine is to be used
        in a JIT engine.

        The `abiname` option specifies the ABI. RISC-V targets with hard-float
        needs to pass the ABI name to LLVM.
        '''

class TargetMachine(ffi.ObjectRef):
    def _dispose(self) -> None: ...
    def add_analysis_passes(self, pm) -> None:
        """
        Register analysis passes for this target machine with a pass manager.
        """
    def set_asm_verbosity(self, verbose) -> None:
        """
        Set whether this target machine will emit assembly with human-readable
        comments describing control flow, debug information, and so on.
        """
    def emit_object(self, module):
        """
        Represent the module as a code object, suitable for use with
        the platform's linker.  Returns a byte string.
        """
    def emit_assembly(self, module):
        """
        Return the raw assembler of the module, as a string.

        llvm.initialize_native_asmprinter() must have been called first.
        """
    def _emit_to_memory(self, module, use_object: bool = False):
        """Returns bytes of object code of the module.

        Args
        ----
        use_object : bool
            Emit object code or (if False) emit assembly code.
        """
    @property
    def target_data(self): ...
    @property
    def triple(self): ...

def has_svml():
    """
    Returns True if SVML was enabled at FFI support compile time.
    """

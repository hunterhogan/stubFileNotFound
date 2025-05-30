from _typeshed import Incomplete
from llvmlite.binding import ffi as ffi

def initialize() -> None:
    """
    Initialize the LLVM core.
    """
def initialize_all_targets() -> None:
    """
    Initialize all targets. Necessary before targets can be looked up
    via the :class:`Target` class.
    """
def initialize_all_asmprinters() -> None:
    """
    Initialize all code generators. Necessary before generating
    any assembly or machine code via the :meth:`TargetMachine.emit_object`
    and :meth:`TargetMachine.emit_assembly` methods.
    """
def initialize_native_target() -> None:
    """
    Initialize the native (host) target.  Necessary before doing any
    code generation.
    """
def initialize_native_asmprinter() -> None:
    """
    Initialize the native ASM printer.
    """
def initialize_native_asmparser() -> None:
    """
    Initialize the native ASM parser.
    """
def shutdown() -> None: ...
def _version_info(): ...

llvm_version_info: Incomplete

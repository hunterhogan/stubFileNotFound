from _typeshed import Incomplete
from llvmlite.ir import types as types
from llvmlite.ir._utils import _HasMetadata as _HasMetadata
from llvmlite.ir.values import ArgumentAttributes as ArgumentAttributes, AttributeSet as AttributeSet, Block as Block, Constant as Constant, Function as Function, MetaDataArgument as MetaDataArgument, MetaDataString as MetaDataString, NamedValue as NamedValue, Undefined as Undefined, Value as Value

class Instruction(NamedValue, _HasMetadata):
    opname: Incomplete
    operands: Incomplete
    flags: Incomplete
    metadata: Incomplete
    def __init__(self, parent, typ, opname, operands, name: str = '', flags=()) -> None: ...
    @property
    def function(self): ...
    @property
    def module(self): ...
    def descr(self, buf) -> None: ...
    def replace_usage(self, old, new) -> None: ...
    def __repr__(self) -> str: ...

class CallInstrAttributes(AttributeSet):
    _known: Incomplete

TailMarkerOptions: Incomplete

class FastMathFlags(AttributeSet):
    _known: Incomplete

class CallInstr(Instruction):
    cconv: Incomplete
    tail: Incomplete
    fastmath: Incomplete
    attributes: Incomplete
    arg_attributes: Incomplete
    def __init__(self, parent, func, args, name: str = '', cconv: Incomplete | None = None, tail: Incomplete | None = None, fastmath=(), attrs=(), arg_attrs: Incomplete | None = None) -> None: ...
    @property
    def callee(self): ...
    @callee.setter
    def callee(self, newcallee) -> None: ...
    @property
    def args(self): ...
    def replace_callee(self, newfunc) -> None: ...
    @property
    def called_function(self):
        """The callee function"""
    def _descr(self, buf, add_metadata): ...
    def descr(self, buf) -> None: ...

class InvokeInstr(CallInstr):
    opname: str
    normal_to: Incomplete
    unwind_to: Incomplete
    def __init__(self, parent, func, args, normal_to, unwind_to, name: str = '', cconv: Incomplete | None = None, fastmath=(), attrs=(), arg_attrs: Incomplete | None = None) -> None: ...
    def descr(self, buf) -> None: ...

class Terminator(Instruction):
    def __init__(self, parent, opname, operands) -> None: ...
    def descr(self, buf) -> None: ...

class PredictableInstr(Instruction):
    def set_weights(self, weights) -> None: ...

class Ret(Terminator):
    def __init__(self, parent, opname, return_value: Incomplete | None = None) -> None: ...
    @property
    def return_value(self): ...
    def descr(self, buf) -> None: ...

class Branch(Terminator): ...
class ConditionalBranch(PredictableInstr, Terminator): ...

class IndirectBranch(PredictableInstr, Terminator):
    destinations: Incomplete
    def __init__(self, parent, opname, addr) -> None: ...
    @property
    def address(self): ...
    def add_destination(self, block) -> None: ...
    def descr(self, buf) -> None: ...

class SwitchInstr(PredictableInstr, Terminator):
    default: Incomplete
    cases: Incomplete
    def __init__(self, parent, opname, val, default) -> None: ...
    @property
    def value(self): ...
    def add_case(self, val, block) -> None: ...
    def descr(self, buf) -> None: ...

class Resume(Terminator): ...

class SelectInstr(Instruction):
    def __init__(self, parent, cond, lhs, rhs, name: str = '', flags=()) -> None: ...
    @property
    def cond(self): ...
    @property
    def lhs(self): ...
    @property
    def rhs(self): ...
    def descr(self, buf) -> None: ...

class CompareInstr(Instruction):
    OPNAME: str
    VALID_OP: Incomplete
    op: Incomplete
    def __init__(self, parent, op, lhs, rhs, name: str = '', flags=[]) -> None: ...
    def descr(self, buf) -> None: ...

class ICMPInstr(CompareInstr):
    OPNAME: str
    VALID_OP: Incomplete
    VALID_FLAG: Incomplete

class FCMPInstr(CompareInstr):
    OPNAME: str
    VALID_OP: Incomplete
    VALID_FLAG: Incomplete

class CastInstr(Instruction):
    def __init__(self, parent, op, val, typ, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class LoadInstr(Instruction):
    align: Incomplete
    def __init__(self, parent, ptr, name: str = '', typ: Incomplete | None = None) -> None: ...
    def descr(self, buf) -> None: ...

class StoreInstr(Instruction):
    def __init__(self, parent, val, ptr) -> None: ...
    def descr(self, buf) -> None: ...

class LoadAtomicInstr(Instruction):
    ordering: Incomplete
    align: Incomplete
    def __init__(self, parent, ptr, ordering, align, name: str = '', typ: Incomplete | None = None) -> None: ...
    def descr(self, buf) -> None: ...

class StoreAtomicInstr(Instruction):
    ordering: Incomplete
    align: Incomplete
    def __init__(self, parent, val, ptr, ordering, align) -> None: ...
    def descr(self, buf) -> None: ...

class AllocaInstr(Instruction):
    allocated_type: Incomplete
    align: Incomplete
    def __init__(self, parent, typ, count, name) -> None: ...
    def descr(self, buf) -> None: ...

class GEPInstr(Instruction):
    source_etype: Incomplete
    pointer: Incomplete
    indices: Incomplete
    inbounds: Incomplete
    def __init__(self, parent, ptr, indices, inbounds, name, source_etype: Incomplete | None = None) -> None: ...
    def descr(self, buf) -> None: ...

class PhiInstr(Instruction):
    incomings: Incomplete
    def __init__(self, parent, typ, name, flags=()) -> None: ...
    def descr(self, buf) -> None: ...
    def add_incoming(self, value, block) -> None: ...
    def replace_usage(self, old, new) -> None: ...

class ExtractElement(Instruction):
    def __init__(self, parent, vector, index, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class InsertElement(Instruction):
    def __init__(self, parent, vector, value, index, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class ShuffleVector(Instruction):
    def __init__(self, parent, vector1, vector2, mask, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class ExtractValue(Instruction):
    aggregate: Incomplete
    indices: Incomplete
    def __init__(self, parent, agg, indices, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class InsertValue(Instruction):
    aggregate: Incomplete
    value: Incomplete
    indices: Incomplete
    def __init__(self, parent, agg, elem, indices, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class Unreachable(Instruction):
    def __init__(self, parent) -> None: ...
    def descr(self, buf) -> None: ...

class InlineAsm:
    type: Incomplete
    function_type: Incomplete
    asm: Incomplete
    constraint: Incomplete
    side_effect: Incomplete
    def __init__(self, ftype, asm, constraint, side_effect: bool = False) -> None: ...
    def descr(self, buf) -> None: ...
    def get_reference(self): ...
    def __str__(self) -> str: ...

class AtomicRMW(Instruction):
    operation: Incomplete
    ordering: Incomplete
    def __init__(self, parent, op, ptr, val, ordering, name) -> None: ...
    def descr(self, buf) -> None: ...

class CmpXchg(Instruction):
    """This instruction has changed since llvm3.5.  It is not compatible with
    older llvm versions.
    """
    ordering: Incomplete
    failordering: Incomplete
    def __init__(self, parent, ptr, cmp, val, ordering, failordering, name) -> None: ...
    def descr(self, buf) -> None: ...

class _LandingPadClause:
    value: Incomplete
    def __init__(self, value) -> None: ...
    def __str__(self) -> str: ...

class CatchClause(_LandingPadClause):
    kind: str

class FilterClause(_LandingPadClause):
    kind: str
    def __init__(self, value) -> None: ...

class LandingPadInstr(Instruction):
    cleanup: Incomplete
    clauses: Incomplete
    def __init__(self, parent, typ, name: str = '', cleanup: bool = False) -> None: ...
    def add_clause(self, clause) -> None: ...
    def descr(self, buf) -> None: ...

class Fence(Instruction):
    '''
    The `fence` instruction.

    As of LLVM 5.0.1:

    fence [syncscope("<target-scope>")] <ordering>  ; yields void
    '''
    VALID_FENCE_ORDERINGS: Incomplete
    ordering: Incomplete
    targetscope: Incomplete
    def __init__(self, parent, ordering, targetscope: Incomplete | None = None, name: str = '') -> None: ...
    def descr(self, buf) -> None: ...

class Comment(Instruction):
    """
    A line comment.
    """
    text: Incomplete
    def __init__(self, parent, text) -> None: ...
    def descr(self, buf) -> None: ...

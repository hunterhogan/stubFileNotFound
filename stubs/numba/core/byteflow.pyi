from _typeshed import Incomplete
from collections.abc import Generator
from enum import Enum
from numba.core.controlflow import CFGraph as CFGraph, NEW_BLOCKERS as NEW_BLOCKERS
from numba.core.errors import UnsupportedBytecodeError as UnsupportedBytecodeError
from numba.core.ir import Loc as Loc
from numba.core.utils import ALL_BINOPS_TO_OPERATORS as ALL_BINOPS_TO_OPERATORS, PYVERSION as PYVERSION, UniqueDict as UniqueDict, _lazy_pformat as _lazy_pformat
from typing import NamedTuple

_logger: Incomplete
_EXCEPT_STACK_OFFSET: int
_FINALLY_POP = _EXCEPT_STACK_OFFSET
_NO_RAISE_OPS: Incomplete

class CALL_INTRINSIC_1_Operand(Enum):
    INTRINSIC_STOPITERATION_ERROR = 3
    UNARY_POSITIVE = 5
    INTRINSIC_LIST_TO_TUPLE = 6
ci1op = CALL_INTRINSIC_1_Operand

class BlockKind:
    """Kinds of block to make related code safer than just `str`.
    """
    _members: Incomplete
    _value: Incomplete
    def __init__(self, value) -> None: ...
    def __hash__(self): ...
    def __lt__(self, other): ...
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class Flow:
    """Data+Control Flow analysis.

    Simulate execution to recover dataflow and controlflow information.
    """
    _bytecode: Incomplete
    block_infos: Incomplete
    def __init__(self, bytecode) -> None: ...
    def run(self):
        """Run a trace over the bytecode over all reachable path.

        The trace starts at bytecode offset 0 and gathers stack and control-
        flow information by partially interpreting each bytecode.
        Each ``State`` instance in the trace corresponds to a basic-block.
        The State instances forks when a jump instruction is encountered.
        A newly forked state is then added to the list of pending states.
        The trace ends when there are no more pending states.
        """
    def _run_handle_exception(self, runner, state): ...
    def _run_handle_exception(self, runner, state): ...
    cfgraph: Incomplete
    def _build_cfg(self, all_states) -> None: ...
    def _prune_phis(self, runner): ...
    def _is_implicit_new_block(self, state): ...
    def _guard_with_as(self, state) -> None:
        """Checks if the next instruction after a SETUP_WITH is something other
        than a POP_TOP, if it is something else it'll be some sort of store
        which is not supported (this corresponds to `with CTXMGR as VAR(S)`)."""

def _is_null_temp_reg(reg): ...

class TraceRunner:
    """Trace runner contains the states for the trace and the opcode dispatch.
    """
    debug_filename: Incomplete
    pending: Incomplete
    finished: Incomplete
    def __init__(self, debug_filename) -> None: ...
    def get_debug_loc(self, lineno): ...
    def dispatch(self, state) -> None: ...
    def _adjust_except_stack(self, state) -> None:
        """
        Adjust stack when entering an exception handler to match expectation
        by the bytecode.
        """
    def op_NOP(self, state, inst) -> None: ...
    def op_RESUME(self, state, inst) -> None: ...
    def op_CACHE(self, state, inst) -> None: ...
    def op_PRECALL(self, state, inst) -> None: ...
    def op_PUSH_NULL(self, state, inst) -> None: ...
    def op_RETURN_GENERATOR(self, state, inst) -> None: ...
    def op_FORMAT_SIMPLE(self, state, inst) -> None: ...
    def op_FORMAT_VALUE(self, state, inst) -> None:
        """
        FORMAT_VALUE(flags): flags argument specifies format spec which is
        not supported yet. Currently, we just call str() on the value.
        Pops a value from stack and pushes results back.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE
        """
    def op_BUILD_STRING(self, state, inst) -> None:
        """
        BUILD_STRING(count): Concatenates count strings from the stack and
        pushes the resulting string onto the stack.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING
        """
    def op_POP_TOP(self, state, inst) -> None: ...
    def op_TO_BOOL(self, state, inst) -> None: ...
    def op_LOAD_GLOBAL(self, state, inst) -> None: ...
    def op_LOAD_GLOBAL(self, state, inst) -> None: ...
    def op_LOAD_GLOBAL(self, state, inst) -> None: ...
    def op_COPY_FREE_VARS(self, state, inst) -> None: ...
    def op_MAKE_CELL(self, state, inst) -> None: ...
    def op_LOAD_DEREF(self, state, inst) -> None: ...
    def op_LOAD_CONST(self, state, inst) -> None: ...
    def op_LOAD_ATTR(self, state, inst) -> None: ...
    def op_LOAD_FAST(self, state, inst) -> None: ...
    def op_LOAD_FAST_LOAD_FAST(self, state, inst) -> None: ...
    def op_STORE_FAST_LOAD_FAST(self, state, inst) -> None: ...
    def op_STORE_FAST_STORE_FAST(self, state, inst) -> None: ...
    op_LOAD_FAST_CHECK = op_LOAD_FAST
    op_LOAD_FAST_AND_CLEAR = op_LOAD_FAST
    def op_DELETE_FAST(self, state, inst) -> None: ...
    def op_DELETE_ATTR(self, state, inst) -> None: ...
    def op_STORE_ATTR(self, state, inst) -> None: ...
    def op_STORE_DEREF(self, state, inst) -> None: ...
    def op_STORE_FAST(self, state, inst) -> None: ...
    def op_SLICE_1(self, state, inst) -> None:
        """
        TOS = TOS1[TOS:]
        """
    def op_SLICE_2(self, state, inst) -> None:
        """
        TOS = TOS1[:TOS]
        """
    def op_SLICE_3(self, state, inst) -> None:
        """
        TOS = TOS2[TOS1:TOS]
        """
    def op_STORE_SLICE_0(self, state, inst) -> None:
        """
        TOS[:] = TOS1
        """
    def op_STORE_SLICE_1(self, state, inst) -> None:
        """
        TOS1[TOS:] = TOS2
        """
    def op_STORE_SLICE_2(self, state, inst) -> None:
        """
        TOS1[:TOS] = TOS2
        """
    def op_STORE_SLICE_3(self, state, inst) -> None:
        """
        TOS2[TOS1:TOS] = TOS3
        """
    def op_DELETE_SLICE_0(self, state, inst) -> None:
        """
        del TOS[:]
        """
    def op_DELETE_SLICE_1(self, state, inst) -> None:
        """
        del TOS1[TOS:]
        """
    def op_DELETE_SLICE_2(self, state, inst) -> None:
        """
        del TOS1[:TOS]
        """
    def op_DELETE_SLICE_3(self, state, inst) -> None:
        """
        del TOS2[TOS1:TOS]
        """
    def op_BUILD_SLICE(self, state, inst) -> None:
        """
        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)
        """
    def op_BINARY_SLICE(self, state, inst) -> None: ...
    def op_STORE_SLICE(self, state, inst) -> None: ...
    def _op_POP_JUMP_IF(self, state, inst) -> None: ...
    op_POP_JUMP_IF_TRUE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_FALSE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_NONE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_NOT_NONE = _op_POP_JUMP_IF
    def _op_JUMP_IF_OR_POP(self, state, inst) -> None: ...
    op_JUMP_IF_FALSE_OR_POP = _op_JUMP_IF_OR_POP
    op_JUMP_IF_TRUE_OR_POP = _op_JUMP_IF_OR_POP
    def op_POP_JUMP_FORWARD_IF_NONE(self, state, inst) -> None: ...
    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, state, inst) -> None: ...
    def op_POP_JUMP_BACKWARD_IF_NONE(self, state, inst) -> None: ...
    def op_POP_JUMP_BACKWARD_IF_NOT_NONE(self, state, inst) -> None: ...
    def op_POP_JUMP_FORWARD_IF_FALSE(self, state, inst) -> None: ...
    def op_POP_JUMP_FORWARD_IF_TRUE(self, state, inst) -> None: ...
    def op_POP_JUMP_BACKWARD_IF_FALSE(self, state, inst) -> None: ...
    def op_POP_JUMP_BACKWARD_IF_TRUE(self, state, inst) -> None: ...
    def op_JUMP_FORWARD(self, state, inst) -> None: ...
    def op_JUMP_BACKWARD(self, state, inst) -> None: ...
    op_JUMP_BACKWARD_NO_INTERRUPT = op_JUMP_BACKWARD
    def op_JUMP_ABSOLUTE(self, state, inst) -> None: ...
    def op_BREAK_LOOP(self, state, inst) -> None: ...
    def op_RETURN_VALUE(self, state, inst) -> None: ...
    def op_RETURN_CONST(self, state, inst) -> None: ...
    def op_YIELD_VALUE(self, state, inst) -> None: ...
    def op_RAISE_VARARGS(self, state, inst) -> None: ...
    def op_RAISE_VARARGS(self, state, inst) -> None: ...
    def op_BEGIN_FINALLY(self, state, inst) -> None: ...
    def op_END_FINALLY(self, state, inst) -> None: ...
    def op_END_FOR(self, state, inst) -> None: ...
    def op_END_FOR(self, state, inst) -> None: ...
    def op_POP_FINALLY(self, state, inst) -> None: ...
    def op_CALL_FINALLY(self, state, inst) -> None: ...
    def op_WITH_EXCEPT_START(self, state, inst) -> None: ...
    def op_WITH_CLEANUP_START(self, state, inst) -> None: ...
    def op_WITH_CLEANUP_FINISH(self, state, inst) -> None: ...
    def op_SETUP_LOOP(self, state, inst) -> None: ...
    def op_BEFORE_WITH(self, state, inst) -> None: ...
    def op_SETUP_WITH(self, state, inst) -> None: ...
    def _setup_try(self, kind, state, next, end) -> None: ...
    def op_PUSH_EXC_INFO(self, state, inst) -> None: ...
    def op_SETUP_FINALLY(self, state, inst) -> None: ...
    def op_POP_EXCEPT(self, state, inst) -> None: ...
    def op_POP_EXCEPT(self, state, inst) -> None: ...
    def op_POP_BLOCK(self, state, inst) -> None: ...
    def op_BINARY_SUBSCR(self, state, inst) -> None: ...
    def op_STORE_SUBSCR(self, state, inst) -> None: ...
    def op_DELETE_SUBSCR(self, state, inst) -> None: ...
    def op_CALL(self, state, inst) -> None: ...
    def op_KW_NAMES(self, state, inst) -> None: ...
    def op_CALL_FUNCTION(self, state, inst) -> None: ...
    def op_CALL_FUNCTION_KW(self, state, inst) -> None: ...
    def op_CALL_KW(self, state, inst) -> None: ...
    def op_CALL_FUNCTION_EX(self, state, inst) -> None: ...
    def op_CALL_FUNCTION_EX(self, state, inst) -> None: ...
    def _dup_topx(self, state, inst, count) -> None: ...
    def op_CALL_INTRINSIC_1(self, state, inst) -> None: ...
    def op_DUP_TOPX(self, state, inst) -> None: ...
    def op_DUP_TOP(self, state, inst) -> None: ...
    def op_DUP_TOP_TWO(self, state, inst) -> None: ...
    def op_COPY(self, state, inst) -> None: ...
    def op_SWAP(self, state, inst) -> None: ...
    def op_ROT_TWO(self, state, inst) -> None: ...
    def op_ROT_THREE(self, state, inst) -> None: ...
    def op_ROT_FOUR(self, state, inst) -> None: ...
    def op_UNPACK_SEQUENCE(self, state, inst) -> None: ...
    def op_BUILD_TUPLE(self, state, inst) -> None: ...
    def _build_tuple_unpack(self, state, inst) -> None: ...
    def op_BUILD_TUPLE_UNPACK_WITH_CALL(self, state, inst) -> None: ...
    def op_BUILD_TUPLE_UNPACK(self, state, inst) -> None: ...
    def op_LIST_TO_TUPLE(self, state, inst) -> None: ...
    def op_BUILD_CONST_KEY_MAP(self, state, inst) -> None: ...
    def op_BUILD_LIST(self, state, inst) -> None: ...
    def op_LIST_APPEND(self, state, inst) -> None: ...
    def op_LIST_EXTEND(self, state, inst) -> None: ...
    def op_BUILD_MAP(self, state, inst) -> None: ...
    def op_MAP_ADD(self, state, inst) -> None: ...
    def op_BUILD_SET(self, state, inst) -> None: ...
    def op_SET_UPDATE(self, state, inst) -> None: ...
    def op_DICT_UPDATE(self, state, inst) -> None: ...
    def op_GET_ITER(self, state, inst) -> None: ...
    def op_FOR_ITER(self, state, inst) -> None: ...
    def op_GEN_START(self, state, inst) -> None:
        """Pops TOS. If TOS was not None, raises an exception. The kind
        operand corresponds to the type of generator or coroutine and
        determines the error message. The legal kinds are 0 for generator,
        1 for coroutine, and 2 for async generator.

        New in version 3.10.
        """
    def op_BINARY_OP(self, state, inst) -> None: ...
    def _unaryop(self, state, inst) -> None: ...
    op_UNARY_NEGATIVE = _unaryop
    op_UNARY_POSITIVE = _unaryop
    op_UNARY_NOT = _unaryop
    op_UNARY_INVERT = _unaryop
    def _binaryop(self, state, inst) -> None: ...
    op_COMPARE_OP = _binaryop
    op_IS_OP = _binaryop
    op_CONTAINS_OP = _binaryop
    op_INPLACE_ADD = _binaryop
    op_INPLACE_SUBTRACT = _binaryop
    op_INPLACE_MULTIPLY = _binaryop
    op_INPLACE_DIVIDE = _binaryop
    op_INPLACE_TRUE_DIVIDE = _binaryop
    op_INPLACE_FLOOR_DIVIDE = _binaryop
    op_INPLACE_MODULO = _binaryop
    op_INPLACE_POWER = _binaryop
    op_INPLACE_MATRIX_MULTIPLY = _binaryop
    op_INPLACE_LSHIFT = _binaryop
    op_INPLACE_RSHIFT = _binaryop
    op_INPLACE_AND = _binaryop
    op_INPLACE_OR = _binaryop
    op_INPLACE_XOR = _binaryop
    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop
    op_BINARY_MATRIX_MULTIPLY = _binaryop
    op_BINARY_LSHIFT = _binaryop
    op_BINARY_RSHIFT = _binaryop
    op_BINARY_AND = _binaryop
    op_BINARY_OR = _binaryop
    op_BINARY_XOR = _binaryop
    def op_MAKE_FUNCTION(self, state, inst, MAKE_CLOSURE: bool = False) -> None: ...
    def op_SET_FUNCTION_ATTRIBUTE(self, state, inst) -> None: ...
    def op_MAKE_CLOSURE(self, state, inst) -> None: ...
    def op_LOAD_CLOSURE(self, state, inst) -> None: ...
    def op_LOAD_ASSERTION_ERROR(self, state, inst) -> None: ...
    def op_CHECK_EXC_MATCH(self, state, inst) -> None: ...
    def op_JUMP_IF_NOT_EXC_MATCH(self, state, inst) -> None: ...
    def op_RERAISE(self, state, inst) -> None: ...
    def op_RERAISE(self, state, inst) -> None: ...
    def op_LOAD_METHOD(self, state, inst) -> None: ...
    def op_LOAD_METHOD(self, state, inst) -> None: ...
    def op_CALL_METHOD(self, state, inst) -> None: ...

class _State:
    """State of the trace
    """
    _bytecode: Incomplete
    _pc_initial: Incomplete
    _pc: Incomplete
    _nstack_initial: Incomplete
    _stack: Incomplete
    _blockstack_initial: Incomplete
    _blockstack: Incomplete
    _temp_registers: Incomplete
    _insts: Incomplete
    _outedges: Incomplete
    _terminated: bool
    _phis: Incomplete
    _outgoing_phis: Incomplete
    _used_regs: Incomplete
    def __init__(self, bytecode, pc, nstack, blockstack, nullvals=()) -> None:
        """
        Parameters
        ----------
        bytecode : numba.bytecode.ByteCode
            function bytecode
        pc : int
            program counter
        nstack : int
            stackdepth at entry
        blockstack : Sequence[Dict]
            A sequence of dictionary denoting entries on the blockstack.
        """
    def __repr__(self) -> str: ...
    def get_identity(self): ...
    def __hash__(self): ...
    def __lt__(self, other): ...
    def __eq__(self, other): ...
    @property
    def pc_initial(self):
        """The starting bytecode offset of this State.
        The PC given to the constructor.
        """
    @property
    def instructions(self):
        """The list of instructions information as a 2-tuple of
        ``(pc : int, register_map : Dict)``
        """
    @property
    def outgoing_edges(self):
        """The list of outgoing edges.

        Returns
        -------
        edges : List[State]
        """
    @property
    def outgoing_phis(self):
        """The dictionary of outgoing phi nodes.

        The keys are the name of the PHI nodes.
        The values are the outgoing states.
        """
    @property
    def blockstack_initial(self):
        """A copy of the initial state of the blockstack
        """
    @property
    def stack_depth(self):
        """The current size of the stack

        Returns
        -------
        res : int
        """
    def find_initial_try_block(self):
        """Find the initial *try* block.
        """
    def has_terminated(self): ...
    def get_inst(self): ...
    def advance_pc(self) -> None: ...
    def make_temp(self, prefix: str = ''): ...
    def append(self, inst, **kwargs) -> None:
        """Append new inst"""
    def get_tos(self): ...
    def peek(self, k):
        """Return the k'th element on the stack
        """
    def push(self, item) -> None:
        """Push to stack"""
    def pop(self):
        """Pop the stack"""
    def swap(self, idx) -> None:
        """Swap stack[idx] with the tos"""
    def push_block(self, synblk) -> None:
        """Push a block to blockstack
        """
    def reset_stack(self, depth):
        """Reset the stack to the given stack depth.
        Returning the popped items.
        """
    def make_block(self, kind, end, reset_stack: bool = True, handler: Incomplete | None = None):
        """Make a new block
        """
    def pop_block(self):
        """Pop a block and unwind the stack
        """
    def pop_block_and_above(self, blk) -> None:
        """Find *blk* in the blockstack and remove it and all blocks above it
        from the stack.
        """
    def get_top_block(self, kind):
        """Find the first block that matches *kind*
        """
    def get_top_block_either(self, *kinds):
        """Find the first block that matches *kind*
        """
    def has_active_try(self):
        """Returns a boolean indicating if the top-block is a *try* block
        """
    def get_varname(self, inst):
        """Get referenced variable name from the instruction's oparg
        """
    def get_varname_by_arg(self, oparg: int):
        """Get referenced variable name from the oparg
        """
    def terminate(self) -> None:
        """Mark block as terminated
        """
    def fork(self, pc, npop: int = 0, npush: int = 0, extra_block: Incomplete | None = None) -> None:
        """Fork the state
        """
    def split_new_block(self) -> None:
        """Split the state
        """
    def get_outgoing_states(self):
        """Get states for each outgoing edges
        """
    def get_outgoing_edgepushed(self):
        """
        Returns
        -------
        Dict[int, int]
            where keys are the PC
            values are the edge-pushed stack values
        """

class StatePy311(_State):
    _kw_names: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def pop_kw_names(self): ...
    def set_kw_names(self, val) -> None: ...
    def is_in_exception(self): ...
    def get_exception(self): ...
    def in_with(self): ...
    def make_null(self): ...

class StatePy313(StatePy311):
    _make_func_attrs: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_function_attribute(self, make_func_res, **kwargs) -> None: ...
    def get_function_attributes(self, make_func_res): ...
State = StatePy313
State = StatePy311
State = _State

class Edge(NamedTuple):
    pc: Incomplete
    stack: Incomplete
    blockstack: Incomplete
    npush: Incomplete

class AdaptDFA:
    """Adapt Flow to the old DFA class expected by Interpreter
    """
    _flow: Incomplete
    def __init__(self, flow) -> None: ...
    @property
    def infos(self): ...

class AdaptBlockInfo(NamedTuple):
    insts: Incomplete
    outgoing_phis: Incomplete
    blockstack: Incomplete
    active_try_block: Incomplete
    outgoing_edgepushed: Incomplete

def adapt_state_infos(state): ...
def _flatten_inst_regs(iterable) -> Generator[Incomplete]:
    """Flatten an iterable of registers used in an instruction
    """

class AdaptCFA:
    """Adapt Flow to the old CFA class expected by Interpreter
    """
    _flow: Incomplete
    _blocks: Incomplete
    _backbone: Incomplete
    def __init__(self, flow) -> None: ...
    @property
    def graph(self): ...
    @property
    def backbone(self): ...
    @property
    def blocks(self): ...
    def iterliveblocks(self) -> Generator[Incomplete]: ...
    def dump(self) -> None: ...

class AdaptCFBlock:
    offset: Incomplete
    body: Incomplete
    def __init__(self, blockinfo, offset) -> None: ...

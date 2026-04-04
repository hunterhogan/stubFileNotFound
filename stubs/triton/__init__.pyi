from . import language as language, runtime as runtime, testing as testing, tools as tools
from .compiler import CompilationError as CompilationError, compile as compile  # noqa: A004
from .errors import TritonError as TritonError
from .runtime import (
	autotune as autotune, Config as Config, heuristics as heuristics, InterpreterError as InterpreterError, JITFunction as JITFunction,
	KernelInterface as KernelInterface, MockTensor as MockTensor, OutOfResources as OutOfResources, reinterpret as reinterpret,
	TensorWrapper as TensorWrapper)
from .runtime._allocation import set_allocator as set_allocator
from .runtime._async_compile import AsyncCompileMode as AsyncCompileMode, FutureKernel as FutureKernel
from .runtime.jit import constexpr_function as constexpr_function, jit as jit

__all__ = ['AsyncCompileMode', 'CompilationError', 'Config', 'FutureKernel', 'InterpreterError', 'JITFunction', 'KernelInterface', 'MockTensor', 'OutOfResources', 'TensorWrapper', 'TritonError', 'autotune', 'cdiv', 'compile', 'constexpr_function', 'heuristics', 'jit', 'language', 'must_use_result', 'next_power_of_2', 'reinterpret', 'runtime', 'set_allocator', 'testing', 'tools']

must_use_result = language.core.must_use_result

@constexpr_function
def cdiv(x: int, y: int) -> int: ...
@constexpr_function
def next_power_of_2(n: int) -> int: ...


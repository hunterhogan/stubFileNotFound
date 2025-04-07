from typing import Any, Dict, List, Union, Optional, overload
from collections.abc import Callable
import logging

_logger: logging.Logger
_msg_deprecated_signature_arg: str

@overload
def jit(signature_or_function: None = None,
		locals: Dict[str, Any] = {},
		cache: bool = False,
		pipeline_class: Optional[Any] = None,
		boundscheck: Optional[Any] = None,
		**options: Union[bool, str, Callable[..., Any], None]) -> Callable[..., Any]: ...

@overload
def jit(signature_or_function: Union[str, List[str]],
		locals: Dict[str, Any] = {},
		cache: bool = False,
		pipeline_class: Optional[Any] = None,
		boundscheck: Optional[Any] = None,
		**options: Union[bool, str, Callable[..., Any], None]) -> Callable[..., Any]: ...

@overload
def jit(signature_or_function: Callable[..., Any],
		locals: Dict[str, Any] = {},
		cache: bool = False,
		pipeline_class: Optional[Any] = None,
		boundscheck: Optional[Any] = None,
		**options: Union[bool, str, Callable[..., Any], None]) -> Any: ...

def jit(signature_or_function: Optional[Union[Callable[..., Any], str, List[str]]] = None,
		locals: Dict[str, Any] = {},
		cache: bool = False,
		pipeline_class: Optional[Any] = None,
		boundscheck: Optional[Any] = None,
		**options: Union[bool, str, Callable[..., Any], None]) -> Union[Callable[..., Any], Any]:
	'''
	This decorator is used to compile a Python function into native code.

	Args
	-----
	signature_or_function:
		The (optional) signature or list of signatures to be compiled.
		If not passed, required signatures will be compiled when the
		decorated function is called, depending on the argument values.
		As a convenience, you can directly pass the function to be compiled
		instead.

	locals: dict
		Mapping of local variable names to Numba types. Used to override the
		types deduced by Numba\'s type inference engine.

	pipeline_class: type numba.compiler.CompilerBase
			The compiler pipeline type for customizing the compilation stages.

	boundscheck: bool or None
		Set to True to enable bounds checking for array indices. Out
		of bounds accesses will raise IndexError. The default is to
		not do bounds checking. If False, bounds checking is disabled,
		out of bounds accesses can produce garbage results or segfaults.
		However, enabling bounds checking will slow down typical
		functions, so it is recommended to only use this flag for
		debugging. You can also set the NUMBA_BOUNDSCHECK environment
		variable to 0 or 1 to globally override this flag. The default
		value is None, which under normal execution equates to False,
		but if debug is set to True then bounds checking will be
		enabled.

	options:
		For a cpu target, valid options are:
			nopython: bool
				Set to True to disable the use of PyObjects and Python API
				calls. The default behavior is to allow the use of PyObjects
				and Python API. Default value is True.

			forceobj: bool
				Set to True to force the use of PyObjects for every value.
				Default value is False.

			looplift: bool
				Set to True to enable jitting loops in nopython mode while
				leaving surrounding code in object mode. This allows functions
				to allocate NumPy arrays and use Python objects, while the
				tight loops in the function can still be compiled in nopython
				mode. Any arrays that the tight loop uses should be created
				before the loop is entered. Default value is True.

			error_model: str
				The error-model affects divide-by-zero behavior.
				Valid values are \'python\' and \'numpy\'. The \'python\' model
				raises exception.  The \'numpy\' model sets the result to
				*+/-inf* or *nan*. Default value is \'python\'.

			inline: str or callable
				The inline option will determine whether a function is inlined
				at into its caller if called. String options are \'never\'
				(default) which will never inline, and \'always\', which will
				always inline. If a callable is provided it will be called with
				the call expression node that is requesting inlining, the
				caller\'s IR and callee\'s IR as arguments, it is expected to
				return Truthy as to whether to inline.
				NOTE: This inlining is performed at the Numba IR level and is in
				no way related to LLVM inlining.

			nogil: bool
				Set to True to release the GIL inside the compiled function.
				The default value is False.

			fastmath: bool
				Set to True to enable fast-math optimizations. These optimizations
				can affect the numerical precision of floating point operations and
				are not IEEE 754 compliant. Default value is False.

			parallel: bool
				Set to True to enable automatic parallelization of suitable array
				operations. Default value is False.

			target: str
				The target architecture to compile for, for example 'cpu',
				'cuda', etc. Default value is 'cpu'.

			debug: bool
				Set to True to enable debugging features. Default value is False.

			no_rewrites: bool
				Set to True to disable automatic rewriting of expressions.
				Default value is False.

			forceinline: bool
				Set to True to force inlining of all function calls.
				Default value is False.

			no_cpython_wrapper: bool
				Set to True to disable generation of CPython wrapper.
				Default value is False.

			no_cfunc_wrapper: bool
				Set to True to disable generation of C function wrapper.
				Default value is False.

			_nrt: bool
				Internal option for managing Numba's memory handling.
				Default value is True.

			_dbg_extend_lifetimes: bool
				Internal debugging option for extending variable lifetimes.
				Default value is False.

			_dbg_optnone: bool
				Internal debugging option to disable LLVM optimizations.
				Default value is False.

	Returns
	--------
	A callable usable as a compiled function. Actual compiling will be
	done lazily if no explicit signatures are passed.

	Examples
	--------
	The function can be used in the following ways:

	1) jit(signatures, **targetoptions) -> jit(function)

		Equivalent to:

			d = dispatcher(function, targetoptions)
			for signature in signatures:
				d.compile(signature)

		Create a dispatcher object for a python function. Then, compile
		the function with the given signature(s).

		Example:

			@jit("int32(int32, int32)")
			def foo(x, y):
				return x + y

			@jit(["int32(int32, int32)", "float32(float32, float32)"])
			def bar(x, y):
				return x + y

	2) jit(function, **targetoptions) -> dispatcher

		Create a dispatcher function object that specializes at call site.

		Examples:

			@jit
			def foo(x, y):
				return x + y

			@jit(nopython=True)
			def bar(x, y):
				return x + y

	'''

def _jit(sigs: Optional[Union[List[str], str]],
		 locals: Dict[str, Any],
		 target: str,
		 cache: bool,
		 targetoptions: Dict[str, Any],
		 **dispatcher_args: Any) -> Callable[..., Any]: ...

@overload
def njit(function: Callable[..., Any], **kws: Any) -> Any: ...

@overload
def njit(signature_or_function: Optional[Union[str, List[str]]] = None, **kws: Any) -> Callable[..., Any]: ...

def njit(*args: Any, **kws: Any) -> Union[Any, Callable[..., Any]]:
	"""
	Equivalent to jit(nopython=True)

	See documentation for jit function/decorator for full description.
	"""

def cfunc(sig: str, locals: Dict[str, Any] = {}, cache: bool = False,
		  pipeline_class: Optional[Any] = None, **options: Any) -> Callable[..., Any]:
	'''
	This decorator is used to compile a Python function into a C callback
	usable with foreign C libraries.

	Usage:
		@cfunc("float64(float64, float64)", nopython=True, cache=True)
		def add(a, b):
			return a + b

	Parameters
	----------
	sig: str
		The signature of the function, which includes the return type and
		the types of all arguments.

	locals: dict
		Mapping of local variable names to Numba types. Used to override the
		types deduced by Numba's type inference engine.

	cache: bool
		Set to True to enable caching of the compiled C callback function.

	pipeline_class: type numba.compiler.CompilerBase
		The compiler pipeline type for customizing the compilation stages.

	options: dict
		Additional keyword arguments to pass to the underlying jit decorator.

	Returns
	-------
	A decorator that takes a function and returns a C callback object.
	'''

def jit_module(**kwargs: Any) -> None:
	""" Automatically ``jit``-wraps functions defined in a Python module

	Note that ``jit_module`` should only be called at the end of the module to
	be jitted. In addition, only functions which are defined in the module
	``jit_module`` is called from are considered for automatic jit-wrapping.
	See the Numba documentation for more information about what can/cannot be
	jitted.

	Parameters
	----------
	kwargs: dict
		Keyword arguments to pass to ``jit`` such as ``nopython``
		or ``error_model``.

	Valid keyword arguments include:
		nopython: bool
			Set to True to disable the use of PyObjects and Python API
			calls. The default behavior is to allow the use of PyObjects
			and Python API. Default value is True.

		forceobj: bool
			Set to True to force the use of PyObjects for every value.
			Default value is False.

		looplift: bool
			Set to True to enable jitting loops in nopython mode while
			leaving surrounding code in object mode. This allows functions
			to allocate NumPy arrays and use Python objects, while the
			tight loops in the function can still be compiled in nopython
			mode. Any arrays that the tight loop uses should be created
			before the loop is entered. Default value is True.

		error_model: str
			The error-model affects divide-by-zero behavior.
			Valid values are 'python' and 'numpy'. The 'python' model
			raises exception.  The 'numpy' model sets the result to
			*+/-inf* or *nan*. Default value is 'python'.

		inline: str or callable
			The inline option will determine whether a function is inlined
			at into its caller if called. String options are 'never'
			(default) which will never inline, and 'always', which will
			always inline. If a callable is provided it will be called with
			the call expression node that is requesting inlining, the
			caller's IR and callee's IR as arguments, it is expected to
			return Truthy as to whether to inline.
			NOTE: This inlining is performed at the Numba IR level and is in
			no way related to LLVM inlining.

		boundscheck: bool or None
			Set to True to enable bounds checking for array indices. Out
			of bounds accesses will raise IndexError. The default is to
			not do bounds checking. If False, bounds checking is disabled,
			out of bounds accesses can produce garbage results or segfaults.
			However, enabling bounds checking will slow down typical
			functions, so it is recommended to only use this flag for
			debugging. You can also set the NUMBA_BOUNDSCHECK environment
			variable to 0 or 1 to globally override this flag. The default
			value is None, which under normal execution equates to False,
			but if debug is set to True then bounds checking will be
			enabled.

		nogil: bool
			Set to True to release the GIL inside the compiled function.
			The default value is False.

		fastmath: bool
			Set to True to enable fast-math optimizations. These optimizations
			can affect the numerical precision of floating point operations and
			are not IEEE 754 compliant. Default value is False.

		parallel: bool
			Set to True to enable automatic parallelization of suitable array
			operations. Default value is False.

		target: str
			The target architecture to compile for, for example 'cpu',
			'cuda', etc. Default value is 'cpu'.

		debug: bool
			Set to True to enable debugging features. Default value is False.

		no_rewrites: bool
			Set to True to disable automatic rewriting of expressions.
			Default value is False.

		forceinline: bool
			Set to True to force inlining of all function calls.
			Default value is False.

		no_cpython_wrapper: bool
			Set to True to disable generation of CPython wrapper.
			Default value is False.

		no_cfunc_wrapper: bool
			Set to True to disable generation of C function wrapper.
			Default value is False.

		_nrt: bool
			Internal option for managing Numba's memory handling.
			Default value is True.

		_dbg_extend_lifetimes: bool
			Internal debugging option for extending variable lifetimes.
			Default value is False.

		_dbg_optnone: bool
			Internal debugging option to disable LLVM optimizations.
			Default value is False.
	"""

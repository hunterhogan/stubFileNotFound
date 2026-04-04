from .. import knobs as knobs  # noqa: TID252
from .cache import get_cache_manager as get_cache_manager, triton_key as triton_key
from .driver import driver as driver
from .errors import AutotunerError as AutotunerError, OutOfResources as OutOfResources, PTXASError as PTXASError
from .jit import JITFunction as JITFunction, KernelInterface
from collections.abc import Callable
from functools import cached_property
from triton._C.libtriton import get_cache_invalidating_env_vars as get_cache_invalidating_env_vars  # ty:ignore[unresolved-import]
from triton.backends.driver import Benchmarker
from typing import Any

class Autotuner(KernelInterface[Any]):
	configs: list[Config]
	keys: list[str]
	cache: dict[tuple[Any, ...], Config]
	arg_names: list[str]
	cache_results: bool
	reset_to_zero: list[str] | None
	restore_value: list[str] | None
	pre_hook: Callable[..., Any] | None
	post_hook: Callable[..., Any] | None
	user_defined_pre_hook: bool
	user_defined_post_hook: bool
	restore_copies: dict[str, Any] | None
	perf_model: Callable[..., Any] | None
	configs_top_k: float
	early_config_prune: Callable[..., Any] | None
	fn: JITFunction[Any]
	base_fn: JITFunction[Any]
	num_warmups: int
	num_reps: int
	use_cuda_graph: bool
	configs_timings: dict[Config, float] | None
	nargs: dict[str, Any]
	bench_time: float
	best_config: Config | None
	def __init__(self, fn: JITFunction[Any], arg_names: list[str], configs: list[Config], key: list[str], reset_to_zero: list[str] | None, restore_value: list[str] | None, pre_hook: Callable[..., Any] | None = None, post_hook: Callable[..., Any] | None = None, prune_configs_by: dict[str, Any] | None = None, warmup: int | None = None, rep: int | None = None, use_cuda_graph: bool = False, do_bench: Callable[..., Any] | None = None, cache_results: bool = False) -> None:
		"""Autotuner.

		:param prune_configs_by: a dict of functions that are used to prune configs, fields:
			'perf_model': performance model used to predicate running time with different configs, returns running time
			'top_k': number of configs to bench
			'early_config_prune': a function used to prune configs. It should have the signature
				`prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
				and return pruned configs. It should return at least one config.
		"""

	@cached_property
	def do_bench(self) -> Benchmarker | Callable[..., Any | list[Any] | None]: ...
	def check_disk_cache(self, tuning_key: tuple[Any, ...], configs: list[Config], bench_fn: Callable[..., Any]) -> bool: ...
	def run(self, *args: Any, **kwargs: Any) -> Any: ...
	def prune_configs(self, kwargs: dict[str, Any]) -> list[Config]: ...
	def warmup(self, *args: Any, **kwargs: Any) -> list[Any]: ...

class Config:
	"""
	An object that represents a possible kernel configuration for the auto-tuner to try.

	:ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
	:type kwargs: dict[Str, Any]
	:ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
					`num_warps=8`, then each kernel instance will be automatically parallelized to
					cooperatively execute using `8 * 32 = 256` threads.
	:type num_warps: int
	:ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
					Mostly useful for matrix multiplication workloads on SM80+ GPUs.
	:type num_stages: int
	:ivar num_ctas: number of blocks in a block cluster. SM90+ only.
	:type num_ctas: int
	:type maxnreg: Optional[int]
	:ivar maxnreg: maximum number of registers one thread can use.  Corresponds
					to ptx .maxnreg directive.  Not supported on all platforms.
	:ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
					function are args.
	:ivar ir_override: filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).
	"""
	kwargs: dict[str, Any]
	num_warps: int
	num_ctas: int
	num_stages: int
	maxnreg: int | None
	pre_hook: Callable[..., Any] | None
	ir_override: str | None
	def __init__(self, kwargs: dict[str, Any], num_warps: int = 4, num_stages: int = 3, num_ctas: int = 1, maxnreg: int | None = None, pre_hook: Callable[..., Any] | None = None, ir_override: str | None = None) -> None: ...
	def __setstate__(self, state: Any) -> None: ...
	def all_kwargs(self) -> dict[str, Any]: ...
	def __hash__(self) -> int: ...
	def __eq__(self, other: object) -> bool: ...

def autotune(configs: list[Config], key: list[str], prune_configs_by: dict[str, Any] | None = None, reset_to_zero: list[str] | None = None, restore_value: list[str] | None = None, pre_hook: Callable[..., Any] | None = None, post_hook: Callable[..., Any] | None = None, warmup: int | None = None, rep: int | None = None, use_cuda_graph: bool = False, do_bench: Callable[..., Any] | None = None, cache_results: bool = False) -> Callable[[JITFunction[Any]], Autotuner]:
	"""
	Decorator for auto-tuning a :code:`triton.jit`'d function.

	.. highlight:: python
	.. code-block:: python

		@triton.autotune(configs=[
			triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
			triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
		],
		key=['x_size'] # the two above configs will be evaluated anytime
						# the value of x_size changes
		)
		@triton.jit
		def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
			...
	:note: When all the configurations are evaluated, the kernel will run multiple times.
		This means that whatever value the kernel updates will be updated multiple times.
		To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
		resets the value of the provided tensor to `zero` before running any configuration.

	If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
	:code:`"1"`, Triton will print a message to stdout after autotuning each
	kernel, including the time spent autotuning and the best configuration.

	:param configs: a list of :code:`triton.Config` objects
	:type configs: list[triton.Config]
	:param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
	:type key: list[str]
	:param prune_configs_by: a dict of functions that are used to prune configs, fields:
		'perf_model': performance model used to predicate running time with different configs, returns running time
		'top_k': number of configs to bench
		'early_config_prune': a function used to prune configs. It should have the signature
				`prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
				and return pruned configs. It should return at least one config.
	:param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
	:type reset_to_zero: list[str]
	:param restore_value: a list of argument names whose value will be restored after evaluating any configs.
	:type restore_value: list[str]
	:param pre_hook: a function that will be called before the kernel is called.
		This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
		'kwargs': a dict of all arguments passed to the kernel.
		'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
	:type pre_hook: lambda args, reset_only
	:param post_hook: a function that will be called after the kernel is called.
		This overrides the default post_hook used for 'restore_value'.
		'kwargs': a dict of all arguments passed to the kernel.
		'exception': the exception raised by the kernel in case of a compilation or runtime error.
	:type post_hook: lambda args, exception
	:param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
	:type warmup: int
	:param rep: repetition time (in ms) to pass to benchmarking (deprecated).
	:type rep: int
	:param do_bench: a benchmark function to measure the time of each run.
	:type do_bench: lambda fn, quantiles
	:param cache_results: whether to cache autotune timings to disk.  Defaults to False.
	"type cache_results: bool
	"""

class Heuristics(KernelInterface[Any]):
	fn: JITFunction[Any]
	values: dict[str, Callable[[dict[str, Any]], Any]]
	arg_names: list[str]
	def __init__(self, fn: JITFunction[Any], arg_names: list[str], values: dict[str, Callable[[dict[str, Any]], Any]]) -> None: ...
	def run(self, *args: Any, **kwargs: Any) -> Any: ...

def heuristics(values: dict[str, Callable[[dict[str, Any]], Any]]) -> Callable[[JITFunction[Any]], Heuristics]:
	"""
	Decorator for specifying how the values of certain meta-parameters may be computed.

	This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

	.. highlight:: python
	.. code-block:: python

		# smallest power-of-two >= x_size
		@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
		@triton.jit
		def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
			...
	:param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
				each such function takes a list of positional arguments as input.
	:type values: dict[str, Callable[[dict[str, Any]], Any]]
	"""

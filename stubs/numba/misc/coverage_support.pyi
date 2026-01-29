from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numba.core import config as config, ir as ir
from typing import no_type_check
import abc
import coverage

coverage_available: bool

@no_type_check
def get_active_coverage():
    """Get active coverage instance or return None if not found.
    """

_the_registry: Callable[[], NotifyLocBase | None]

def get_registered_loc_notify() -> Sequence[NotifyLocBase]:
    """
    Returns a list of the registered NotifyLocBase instances.
    """

class NotifyLocBase(ABC, metaclass=abc.ABCMeta):
    """Interface for notifying visiting of a ``numba.core.ir.Loc``."""

    @abstractmethod
    def notify(self, loc: ir.Loc) -> None: ...
    @abstractmethod
    def close(self) -> None: ...

class NotifyCompilerCoverage(NotifyLocBase):
    """
    Use to notify ``coverage`` about compiled lines.

    The compiled lines are under the "numba_compiled" context in the coverage
    data.
    """

    _collector: Incomplete
    _tracer: Incomplete
    def __init__(self, collector) -> None: ...
    def notify(self, loc: ir.Loc): ...
    def close(self) -> None: ...

@_the_registry.append
def _register_coverage_notifier(): ...

@dataclass(kw_only=True)
class NumbaTracer(coverage.types.Tracer):
    """
    Not actually a tracer as in the coverage implementation, which will
    setup a Python trace function. This implementation pretends to trace
    but instead receives fake trace events for each line the compiler has
    visited.

    See coverage.PyTracer
    """

    data: coverage.types.TTraceData
    trace_arcs: bool
    should_trace: coverage.types.TShouldTraceFn
    should_trace_cache: Mapping[str, coverage.types.TFileDisposition | None]
    should_start_context: coverage.types.TShouldStartContextFn | None
    switch_context: Callable[[str | None], None] | None
    lock_data: Callable[[], None]
    unlock_data: Callable[[], None]
    warn: coverage.types.TWarnFn
    packed_arcs: bool
    _activity: bool = ...
    def start(self) -> coverage.types.TTraceFn | None:
        """Start this tracer, return a trace function if based on
        sys.settrace.
        """
    def stop(self) -> None:
        """Stop this tracer."""
    def activity(self) -> bool:
        """Has there been any activity?"""
    def reset_activity(self) -> None:
        """Reset the activity() flag."""
    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""
    def trace(self, loc: ir.Loc) -> None:
        """Insert coverage data given source location.
        """

def _pack_arcs(l1: int, l2: int) -> int:
    """Pack arcs into a single integer for compatibility with .packed_arcs
    option.

    See
    https://github.com/nedbat/coveragepy/blob/e7c05fe91ee36c0c94e144bb88d25db4fc3d02fd/coverage/ctracer/tracer.c#L171
    """

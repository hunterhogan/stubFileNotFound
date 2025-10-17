from _typeshed import Incomplete
from python_toolbox import cute_iter_tools as cute_iter_tools, freezing as freezing
from typing import Any

class EmitterSystem:
    """
    A system of emitters, representing a set of possible events in a program.

    `EmitterSystem` offers a few advantages over using plain emitters.

    There are the `bottom_emitter` and `top_emitter`, which allow,
    respectively, to keep track of each `emit`ting that goes on, and to
    generate an `emit`ting that affects all emitters in the system.

    The `EmitterSystem` also offers a context manager,
    `.freeze_cache_rebuilding`. When you do actions using this context manager,
    the emitters will not rebuild their cache when changing their
    inputs/outputs. When the outermost context manager has exited, all the
    caches for these emitters will get rebuilt.
    """

    emitters: Incomplete
    bottom_emitter: Incomplete
    top_emitter: Incomplete
    def __init__(self) -> None: ...
    cache_rebuilding_freezer: Incomplete
    @cache_rebuilding_freezer.on_thaw
    def _recalculate_all_cache(self) -> None:
        """Recalculate the cache for all the emitters."""
    def make_emitter(self, inputs: Any=(), outputs: Any=(), name: Any=None) -> Any:
        """Create an emitter in this emitter system. Returns the emitter."""
    def remove_emitter(self, emitter: Any) -> None:
        """Remove an emitter from this system, disconnecting it from everything."""




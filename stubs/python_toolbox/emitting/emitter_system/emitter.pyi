from ..emitter import Emitter as OriginalEmitter
from _typeshed import Incomplete
from typing import Any

class Emitter(OriginalEmitter):
    """
    An emitter you can `emit` from to call all its callable outputs.

    This is an extension of the original `Emitter`, see its documentation for
    more info.

    What this adds is that it keeps track of which emitter system this emitter
    belongs to, and it allows freezing the cache rebuilding for better speed
    when adding many emitters to the system.

    See documentation of `EmitterSystem` for more info.
    """

    emitter_system: Incomplete
    def __init__(self, emitter_system: Any, inputs: Any=(), outputs: Any=(), name: Any=None) -> None:
        """
        Construct the emitter.

        `emitter_system` is the emitter system to which this emitter belongs.

        `inputs` is a list of inputs, all of them must be emitters.

        `outputs` is a list of outputs, they must be either emitters or
        callables.

        `name` is a string name for the emitter.
        """
    def _recalculate_total_callable_outputs_recursively(self) -> None:
        """
        Recalculate `__total_callable_outputs_cache` recursively.

        This will to do the recalculation for this emitter and all its inputs.

        Will not do anything if `_cache_rebuilding_frozen` is positive.
        """
    def add_input(self, emitter: Any) -> None:
        """
        Add an emitter as an input to this emitter.

        Every time that emitter will emit, it will cause this emitter to emit
        as well.

        Emitter must be member of this emitter's emitter system.
        """
    def add_output(self, thing: Any) -> None:
        """
        Add an emitter or a callable as an output to this emitter.

        If adding a callable, every time this emitter will emit the callable
        will be called.

        If adding an emitter, every time this emitter will emit the output
        emitter will emit as well. Note that the output emitter must be a
        member of this emitter's emitter system.
        """




from _typeshed import Incomplete
from python_toolbox import address_tools as address_tools, cute_iter_tools as cute_iter_tools, misc_tools as misc_tools
from typing import Any

class Emitter:
    """
    An emitter you can `emit` from to call all its callable outputs.

    The emitter idea is a variation on the publisher-subscriber design pattern.

    Every emitter has a set of inputs and a set of outputs. The inputs, if
    there are any, must be emitters themselves. So when you `emit` on any of
    this emitter's inputs, it's as if you `emit`ted on this emitter as well.
    (Recursively, of course.)

    The outputs are a bit different. An emitter can have as outputs both (a)
    other emitters and (b) callable objects. (Which means, functions or
    function-like objects.)

    There's no need to explain (a): If `emitter_1` has as an output
    `emitter_2`, then `emitter_2` has as an input `emitter_1`, which works like
    how we explained above about inputs.

    But now (b): An emitter can have callables as outputs. (Without these, the
    emitter idea won't have much use.) These callables simply get called
    whenever the emitter or one of its inputs get `emit`ted.

    The callables that you register as outputs are functions that need to be
    called when the original event that caused the `emit` action happens.
    """

    _is_atomically_pickleable: bool
    _inputs: Incomplete
    _outputs: Incomplete
    __total_callable_outputs_cache: Incomplete
    name: Incomplete
    def __init__(self, inputs: Any=(), outputs: Any=(), name: Any=None) -> None:
        """
        Construct the emitter.

        `inputs` is an iterable of inputs, all of which must be emitters. (You
        can also pass in a single input without using an iterable.)

        `outputs` is an iterable of outputs, which may be either emitters or
        callables. (You can also pass in a single output without using an
        iterable.)

        `name` is a string name for the emitter. (Optional, helps with
        debugging.)
        """
    def get_inputs(self) -> Any:
        """Get the emitter's inputs."""
    def get_outputs(self) -> Any:
        """Get the emitter's outputs."""
    def _get_input_layers(self) -> Any:
        """
        Get the emitter's inputs as a list of layers.

        Every item in the list will be a list of emitters on that layer. For
        example, the first item will be a list of direct inputs of our emitter.
        The second item will be a list of *their* inputs. Etc.

        Every emitter can appear only once in this scheme: It would appear on
        the closest layer that it's on.
        """
    def _recalculate_total_callable_outputs_recursively(self) -> None:
        """
        Recalculate `__total_callable_outputs_cache` recursively.

        This will to do the recalculation for this emitter and all its inputs.
        """
    def _recalculate_total_callable_outputs(self) -> None:
        """
        Recalculate `__total_callable_outputs_cache` for this emitter.

        This will to do the recalculation for this emitter and all its inputs.
        """
    def add_input(self, emitter: Any) -> None:
        """
        Add an emitter as an input to this emitter.

        Every time that emitter will emit, it will cause this emitter to emit
        as well.
        """
    def remove_input(self, emitter: Any) -> None:
        """Remove an input from this emitter."""
    def add_output(self, thing: Any) -> None:
        """
        Add an emitter or a callable as an output to this emitter.

        If adding a callable, every time this emitter will emit the callable
        will be called.

        If adding an emitter, every time this emitter will emit the output
        emitter will emit as well.
        """
    def remove_output(self, thing: Any) -> None:
        """Remove an output from this emitter."""
    def disconnect_from_all(self) -> None:
        """Disconnect the emitter from all its inputs and outputs."""
    def _get_callable_outputs(self) -> Any:
        """Get the direct callable outputs of this emitter."""
    def _get_emitter_outputs(self) -> Any:
        """Get the direct emitter outputs of this emitter."""
    def get_total_callable_outputs(self) -> Any:
        """
        Get the total of callable outputs of this emitter.

        This means the direct callable outputs, and the callable outputs of
        emitter outputs.
        """
    def emit(self) -> None:
        """
        Call all of the (direct or indirect) callable outputs of this emitter.

        This is the most important method of the emitter. When you `emit`, all
        the callable outputs get called in succession.
        """




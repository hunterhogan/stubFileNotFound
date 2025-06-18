from _typeshed import Incomplete
from dask.callbacks import Callback

__all__ = ['TqdmCallback']

class TqdmCallback(Callback):
    """Dask callback for task progress."""
    tqdm_class: Incomplete
    def __init__(self, start: Incomplete | None = None, pretask: Incomplete | None = None, tqdm_class=..., **tqdm_kwargs) -> None:
        """
        Parameters
        ----------
        tqdm_class  : optional
            `tqdm` class to use for bars [default: `tqdm.auto.tqdm`].
        tqdm_kwargs  : optional
            Any other arguments used for all bars.
        """
    pbar: Incomplete
    def _start_state(self, _, state) -> None: ...
    def _posttask(self, *_, **__) -> None: ...
    def _finish(self, *_, **__) -> None: ...
    def display(self) -> None:
        """Displays in the current cell in Notebooks."""

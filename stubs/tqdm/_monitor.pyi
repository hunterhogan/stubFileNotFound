from _typeshed import Incomplete
from threading import Thread

__all__ = ['TMonitor', 'TqdmSynchronisationWarning']

class TqdmSynchronisationWarning(RuntimeWarning):
    """tqdm multi-thread/-process errors which may cause incorrect nesting
    but otherwise no adverse effects"""

class TMonitor(Thread):
    """
    Monitoring thread for tqdm bars.
    Monitors if tqdm bars are taking too much time to display
    and readjusts miniters automatically if necessary.

    Parameters
    ----------
    tqdm_cls  : class
        tqdm class to use (can be core tqdm or a submodule).
    sleep_interval  : float
        Time to sleep between monitoring checks.
    """
    _test: Incomplete
    daemon: bool
    woken: int
    tqdm_cls: Incomplete
    sleep_interval: Incomplete
    _time: Incomplete
    was_killed: Incomplete
    def __init__(self, tqdm_cls, sleep_interval) -> None: ...
    def exit(self): ...
    def get_instances(self): ...
    def run(self) -> None: ...
    def report(self): ...

import logging
from _typeshed import Incomplete

log_glyph: str
log_instance: str
log_dimension: str

class DuplicateMessageFilter(logging.Filter):
    """
    Suppresses any log message that was reported before in the same module and
    for the same logging level. We check for module and level number in
    addition to the message just in case, though checking the message only is
    probably enough.
    """
    logs: Incomplete
    def __init__(self) -> None: ...
    def filter(self, record): ...

class otfautoLogFormatter(logging.Formatter):
    verbose: Incomplete
    def __init__(self, fmt, datefmt=None, verbose: bool = False) -> None: ...
    def format(self, record): ...

def set_log_parameters(dimension=None, glyph=None, instance=None) -> None: ...
def logging_conf(verbose, logfile=None, handlers=None): ...
def log_receiver(logQueue) -> None: ...
def logging_reconfig(logQueue, verbose: int = 0) -> None: ...

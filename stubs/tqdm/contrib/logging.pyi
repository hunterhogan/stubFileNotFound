import logging
from ..std import tqdm as std_tqdm
from _typeshed import Incomplete
from contextlib import contextmanager
from typing import Iterator

class _TqdmLoggingHandler(logging.StreamHandler):
    tqdm_class: Incomplete
    def __init__(self, tqdm_class: type[std_tqdm] = ...) -> None: ...
    def emit(self, record) -> None: ...

def _is_console_logging_handler(handler): ...
def _get_first_found_console_logging_handler(handlers): ...
@contextmanager
def logging_redirect_tqdm(loggers: tuple[list[logging.Logger] | None] = None, tqdm_class: type[std_tqdm] = ...) -> Iterator[None]:
    '''
    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == \'__main__\':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    '''
@contextmanager
def tqdm_logging_redirect(*args, **kwargs) -> Iterator[None]:
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """

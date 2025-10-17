from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import caching as caching, import_tools as import_tools
from typing import Any

def is_multiprocessing_queue(queue: Any) -> Any:
    """Return whether `queue` is a multiprocessing queue."""
def dump(queue: Any) -> Any:
    """
    Empty all pending items in a queue and return them in a list.

    Use only when no other processes/threads are reading from the queue.
    """
def iterate(queue: Any, block: bool = False, limit_to_original_size: bool = False, _prefetch_if_no_qsize: bool = False) -> Generator[Incomplete, Incomplete]:
    """
    Iterate over the items in the queue.

    `limit_to_original_size=True` will limit the number of the items fetched to
    the original number of items in the queue in the beginning.
    """
def get_item(queue: Any, i: Any) -> Any:
    """
    Get an item from the queue by index number without removing any items.

    Note: This was designed for `Queue.Queue`. Don't try to use this, for
    example, on `multiprocessing.Queue`.
    """
def queue_as_list(queue: Any) -> Any:
    """
    Get all the items in the queue as a `list` without removing them.

    Note: This was designed for `Queue.Queue`. Don't try to use this, for
    example, on `multiprocessing.Queue`.
    """
def _platform_supports_multiprocessing_qsize() -> Any:
    """
    Return whether this platform supports `multiprocessing.Queue().qsize()`.

    I'm looking at you, Mac OS.
    """




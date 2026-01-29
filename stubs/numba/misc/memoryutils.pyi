from _typeshed import Incomplete
from collections.abc import Generator
import contextlib

_HAS_PSUTIL: bool
IS_SUPPORTED = _HAS_PSUTIL

def get_available_memory() -> int | None:
    """
    Get current available system memory in bytes.

    Used for memory threshold checking in parallel test execution.

    Returns
    -------
        int or None: Available memory in bytes, or None if unavailable
    """
def get_memory_usage() -> dict[str, int | None]:
    """
    Get memory usage information needed for monitoring.

    Returns only RSS and available memory which are the fields
    actually used by the MemoryTracker.

    Returns
    -------
        dict: Memory usage information including:
            - rss: Current process RSS (physical memory currently used)
            - available: Available system memory
    """

class MemoryTracker:
    """
    A simple memory monitor that tracks RSS delta and timing.

    Stores monitoring data in instance attributes for later access.
    Each instance is typically used for monitoring a single operation.
    """

    pid: int
    name: str
    start_time: float | None
    end_time: float | None
    start_memory: dict[str, int | None] | None
    end_memory: dict[str, int | None] | None
    duration: float | None
    rss_delta: int | None
    def __init__(self, name: str) -> None:
        """Initialize a MemoryTracker with empty monitoring data."""
    @contextlib.contextmanager
    def monitor(self) -> Generator[Incomplete]:
        """
        Context manager to monitor memory usage during function execution.

        Records start/end memory usage and timing, calculates RSS delta,
        and stores all data in instance attributes.

        Args:
            name (str): Name/identifier for the function or operation being
                        monitored

        Yields
        ------
            self: The MemoryTracker instance for accessing stored data
        """
    def get_summary(self) -> str:
        """
        Return a formatted summary of the memory monitoring data.

        Formats the stored monitoring data into a human-readable string
        containing name, PID, RSS delta, available memory, duration,
        and start time.

        Returns
        -------
            str: Formatted summary string with monitoring results

        Note:
            Should be called after monitor() context has completed
            to ensure all data is available.
        """

from typing import Any

def set_process_priority(priority: Any, pid: Any=None) -> None:
    """
    Set the priority of a Windows process.

    Priority is a value between 0-5 where 2 is normal priority. Default sets
    the priority of the current Python process but can take any valid process
    ID.
    """




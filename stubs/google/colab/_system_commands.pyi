from _typeshed import Incomplete

class ShellResult:
    args: Incomplete
    returncode: Incomplete
    output: Incomplete
    def __init__(self, args, returncode, command_output) -> None: ...
    def check_returncode(self) -> None: ...

class _MonitorProcessState:
    process_output: Incomplete
    is_pty_still_connected: bool
    def __init__(self) -> None: ...

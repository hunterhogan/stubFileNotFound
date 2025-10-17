from _typeshed import Incomplete
from python_toolbox import caching as caching, misc_tools as misc_tools
from typing import Any
import abc

class BaseProfileHandler(metaclass=abc.ABCMeta):
    """Profile handler which saves the profiling result in some way."""

    profile: Incomplete
    profile_data: Incomplete
    def __call__(self, profile: Any) -> Any: ...
    @abc.abstractmethod
    def handle(self) -> Any: ...
    make_file_name: Incomplete

class AuxiliaryThreadProfileHandler(BaseProfileHandler, metaclass=abc.ABCMeta):
    """Profile handler that does its action on a separate thread."""

    thread: Incomplete
    def handle(self) -> None: ...
    @abc.abstractmethod
    def thread_job(self) -> Any: ...

class EmailProfileHandler(AuxiliaryThreadProfileHandler):
    """Profile handler that sends the profile via email on separate thread."""

    email_address: Incomplete
    smtp_server: Incomplete
    smtp_user: Incomplete
    smtp_password: Incomplete
    use_tls: Incomplete
    def __init__(self, email_address: Any, smtp_server: Any, smtp_user: Any, smtp_password: Any, use_tls: bool = True) -> None: ...
    def thread_job(self) -> None: ...

class FolderProfileHandler(AuxiliaryThreadProfileHandler):
    """Profile handler that saves the profile to disk on separate thread."""

    folder: Incomplete
    def __init__(self, folder: Any) -> None: ...
    def thread_job(self) -> None: ...

class PrintProfileHandler(BaseProfileHandler):
    """Profile handler that prints profile data to standard output."""

    sort_order: Incomplete
    def __init__(self, sort_order: Any) -> None: ...
    def handle(self) -> None: ...

def get_profile_handler(profile_handler_string: Any) -> Any:
    """Parse `profile_handler_string` into a `ProfileHandler` class."""




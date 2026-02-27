from ..config import PyPIRCCommand
from _typeshed import Incomplete
from typing import ClassVar

class upload(PyPIRCCommand):
    description: ClassVar[str]
    username: str
    password: str
    show_response: int
    sign: bool
    identity: Incomplete
    def initialize_options(self) -> None: ...
    repository: Incomplete
    realm: Incomplete
    def finalize_options(self) -> None: ...
    def run(self) -> None: ...
    def upload_file(self, command: str, pyversion: str, filename: str) -> None: ...

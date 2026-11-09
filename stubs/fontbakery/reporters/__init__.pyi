from collections.abc import Iterable
from fontbakery.checkrunner import CheckRunner
from fontbakery.status import Status

class FontbakeryReporter:
	def __init__(self, *, is_async: bool = False, runner: CheckRunner | None = None, output_file: str | None = None, loglevels: Iterable[Status] | None = None, collect_results_by: str | None = None, succinct: bool = False, quiet: bool = False) -> None: ...
	@property
	def worst_check_status(self) -> Status | None: ...

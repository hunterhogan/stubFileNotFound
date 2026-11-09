from collections.abc import Iterable
from fontbakery.checkrunner import CheckRunner
from fontbakery.reporters import FontbakeryReporter
from fontbakery.status import Status

class TerminalReporter(FontbakeryReporter):
	def __init__(self, *, is_async: bool = False, runner: CheckRunner | None = None, output_file: str | None = None, loglevels: Iterable[Status] | None = None, collect_results_by: str | None = None, succinct: bool = False, quiet: bool = False, print_progress: bool = True, theme: dict[str, object] | None = None) -> None: ...

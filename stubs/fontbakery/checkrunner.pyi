from collections.abc import Sequence
from fontbakery.configuration import Configuration
from fontbakery.profile import Profile
from fontbakery.reporters import FontbakeryReporter
from fontbakery.testable import CheckRunContext

class CheckRunner:
	def __init__(self, profile: Profile, context: CheckRunContext, config: Configuration, jobs: int = 0) -> None: ...
	def run(self, reporters: Sequence[FontbakeryReporter]) -> None: ...

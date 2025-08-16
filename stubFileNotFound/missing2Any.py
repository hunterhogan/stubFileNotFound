"""File-level operations."""

from pathlib import Path
from stubFileNotFound import settingsPackage
from stubFileNotFound.missing2AnyTransformers import processStubString

# Configuration Settings
pathStubs = Path(settingsPackage.pathPackage / '..' / 'stubs').resolve()
pathSuffix: str = 'pyi'
listRelativePaths: list[str] = ['pandas']

# File Discovery Functions
def discoverStubFiles() -> list[Path]:
	"""Discover all stub files based on configured paths and suffix.

	Returns
	-------
	listPathFilenamesStub : list[Path]
		List of all discovered stub file paths.

	"""
	listPathFilenamesStub: list[Path] = []

	for relativePathTarget in listRelativePaths:
		pathTargetStubs = pathStubs / relativePathTarget
		if pathTargetStubs.exists():
			listPathFilenamesStub.extend(pathTargetStubs.rglob(f'*.{pathSuffix}'))

	return listPathFilenamesStub

def processStubFile(pathFilenameStub: Path) -> None:
	"""Process a single stub file to add missing Any annotations by reading from disk, processing, and writing if needed."""
	contentOriginal: str = pathFilenameStub.read_text(encoding='utf-8')
	contentModified: str | None = processStubString(contentOriginal)
	if contentModified is not None:
		pathFilenameStub.write_text(contentModified, encoding='utf-8')
		print(f"Updated: {pathFilenameStub}")

def processAllStubFiles() -> None:
	"""Process all discovered stub files."""
	listPathFilenames = discoverStubFiles()

	for pathFilename in listPathFilenames:
		processStubFile(pathFilename)

if __name__ == "__main__":
	processAllStubFiles()
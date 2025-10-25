"""File-level operations."""

from pathlib import Path
from stubFileNotFound import settingsPackage
from stubFileNotFound.missing2AnyTransformers import processStubString

# Configuration Settings
pathStubs: Path = Path(settingsPackage.pathPackage / '..' / 'stubs').resolve()
pathSuffix: str = 'pyi'

def discoverStubFiles(listRelativePaths: list[str]) -> list[Path]:
	"""Discover all stub files based on configured paths and suffix.

	Returns
	-------
	listPathFilenames : list[Path]
		List of all discovered stub file paths.
	"""
	listPathFilenames: list[Path] = []

	for relativePathTarget in listRelativePaths:
		pathTarget: Path = pathStubs / relativePathTarget
		if pathTarget.exists():
			listPathFilenames.extend(pathTarget.rglob(f'*.{pathSuffix}'))

	return listPathFilenames

def processStubFile(pathFilename: Path) -> None:
	"""Process a single stub file to add missing Any annotations by reading from disk, processing, and writing if needed."""
	pythonSource: str | None = pathFilename.read_text(encoding='utf-8')
	pythonSource = processStubString(pythonSource)
	if pythonSource is not None:
		pathFilename.write_text(pythonSource, encoding='utf-8')
		print(f"Updated: {pathFilename}")

def processAllStubFiles(listRelativePaths: list[str]) -> None:
	"""Process all discovered stub files."""
	listPathFilenames = discoverStubFiles(listRelativePaths)

	for pathFilename in listPathFilenames:
		processStubFile(pathFilename)

if __name__ == "__main__":
	listRelativePaths: list[str] = ['pandas']
	processAllStubFiles(listRelativePaths)

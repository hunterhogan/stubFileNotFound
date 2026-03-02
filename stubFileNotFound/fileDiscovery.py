"""File-level operations."""

from pathlib import Path
from stubFileNotFound import settingsPackage

# Configuration Settings
pathStubs: Path = Path(settingsPackage.pathPackage.parent, 'stubs').resolve()
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



from hunterMakesPy.filesystemToolkit import settings_autoflakeDEFAULT, settings_isortDEFAULT, writePython  # noqa: D100
from pathlib import Path
from stubFileNotFound.fileDiscovery import discoverStubFiles
import subprocess

# ruff: noqa: S607
if __name__ == "__main__":
	listRelativePaths: list[str] = ['fontTools']

	convertFilesTOutf8: bool = True
	stubdefaulter吗: bool = True
	pyupgrade吗: bool = True
	pyupgradeVersion: str = 'py311-plus'
	ruffFix吗: bool = True
	autoflake吗: bool = True
	isort吗: bool = True

	listPathFilenames: list[Path] = discoverStubFiles(listRelativePaths)

	settings = {}
	if autoflake吗:
		settings = {'autoflake': settings_autoflakeDEFAULT.copy()}
		settings['autoflake']['remove_all_unused_imports'] = False
	if isort吗:
		settings['isort'] = settings_isortDEFAULT.copy() # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-assignment]

	for pathFilename in listPathFilenames:
		if convertFilesTOutf8:
			subprocess.run(['normalizer', '-n', '-m', '-r', '-f', str(pathFilename)], check=False)
		# stubdefaulter
		if pyupgrade吗:
			subprocess.run(['pyupgrade', f'--{pyupgradeVersion}', str(pathFilename)], check=False)
		if ruffFix吗:
			subprocess.run(['ruff', 'check', '--fix', '--config', 'ruff.toml', str(pathFilename)], check=False)
		if autoflake吗 or isort吗:
			pythonSource: str = pathFilename.read_text(encoding='utf-8')
			pythonSource = pythonSource.rstrip('\n')
			writePython(pythonSource, pathFilename, settings)

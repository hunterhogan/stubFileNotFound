# ruff: noqa: D100
from __future__ import annotations

from hunterMakesPy.filesystemToolkit import settings_autoflakeDEFAULT, settings_isortDEFAULT, writePython
from stubFileNotFound.fileDiscovery import discoverStubFiles
from typing import TYPE_CHECKING
import subprocess  # noqa: S404

if TYPE_CHECKING:
	from pathlib import Path

# ruff: noqa: S607
if __name__ == "__main__":
	listRelativePaths: list[str] = ['hyper_connections', 'PoPE_pytorch', 'rotary_embedding_torch']

	convertFilesTOutf8: bool = True
	stubdefaulter吗: bool = True
	pyupgrade吗: bool = True
	pyupgradeVersion: str = 'py310-plus'
	ruffFix吗: bool = True
	autoflake吗: bool = True
	isort吗: bool = True

	listPathFilenames: list[Path] = discoverStubFiles(listRelativePaths)

	settings = {}
	if autoflake吗:
		settings = {'autoflake': settings_autoflakeDEFAULT.copy()}
		settings['autoflake']['remove_all_unused_imports'] = False
	if isort吗:
		settings['isort'] = settings_isortDEFAULT.copy()  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-assignment]

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

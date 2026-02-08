from hunterMakesPy.filesystemToolkit import settings_autoflakeDEFAULT, settings_isortDEFAULT, writePython
from pathlib import Path
from stubFileNotFound.missing2Any import discoverStubFiles

if __name__ == "__main__":
	listRelativePaths: list[str] = ['sympy']
	listPathFilenames: list[Path] = discoverStubFiles(listRelativePaths)

	settings = {'autoflake': settings_autoflakeDEFAULT.copy(), 'isort': settings_isortDEFAULT.copy()}
	settings['autoflake']['remove_all_unused_imports'] = False

	for pathFilename in listPathFilenames:
		if pathFilename.stem == '__init__':
			continue
		writePython(pathFilename.read_text(encoding='utf-8'), pathFilename)

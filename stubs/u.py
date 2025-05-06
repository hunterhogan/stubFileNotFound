from pathlib import Path
import subprocess

pathStubs = Path.cwd()

skipDirectories = [
'_soundfile_data',
'atheris',
'gdb',
'moisesdb',
'RPi.GPIO',
'ruamel',
'stdlib',
'uWSGI',
]

py312 = ['tree-sitter-languages', 'tensorflow']

subprocessCMD = ['stubdefaulter', '-p', 'ruamel.yaml', ]

for pathPackage in pathStubs.iterdir():
	if pathPackage.is_file():
		continue
	package = pathPackage.name
	if package in skipDirectories:
		continue
	if package in py312:
		continue
	subprocessCMD.append(package)

subprocess.run(' '.join(subprocessCMD))

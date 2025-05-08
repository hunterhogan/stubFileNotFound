from pathlib import Path
import typeshed_client
import Ast_Stubgen
import subprocess
import sys
import shutil

packageTarget = 'pandas'

pathRoot = Path('/apps/stubFileNotFound')
pathExistingStubs = pathRoot / 'stubs'
pathPackage = pathRoot / 'stubFileNotFound'
pathScratchPad = pathPackage / 'scratchpad'

pathTypeshed = pathRoot / 'typeshed'
pathTypeshedStdlib = pathTypeshed / 'stdlib'
pathTypeshedStubs = pathTypeshed / 'stubs'
searchContext = typeshed_client.get_search_context(typeshed=pathTypeshedStdlib, search_path=[pathTypeshedStubs])

def generateAstStubgenStub(packageTarget: str):
	pathAstStubgen = pathScratchPad / 'ast_stubgen'
	pathAstStubgen.mkdir(exist_ok=True)

	Ast_Stubgen.generate_stub(
		source_file_path=f"{sys.prefix}/lib/site-packages/{packageTarget}",
		output_file_path=str(pathAstStubgen)
	)

def generateMonkeyTypeStub(packageTarget: str):
	pathMonkeyType = pathScratchPad / 'monkeytype'
	pathMonkeyType.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'monkeytype', 'stub', packageTarget
	], cwd=str(pathMonkeyType))

def generateMypyStub(packageTarget: str):
	# Generate with default options
	pathMypyDefault = pathScratchPad / 'mypy_default'
	pathMypyDefault.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'stubgen', '--package', packageTarget,
		'--output', str(pathMypyDefault)
	])

	# Generate with inspect mode
	pathMypyInspect = pathScratchPad / 'mypy_inspect'
	pathMypyInspect.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'stubgen', '--package', packageTarget,
		'--output', str(pathMypyInspect), '--inspect-mode'
	])

def generatePyrightStub(packageTarget: str):
	pathPyright = pathScratchPad / 'pyright'
	pathPyright.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'pyright', '--createstub', packageTarget
	], cwd=str(pathPyright))

def generatePytypeStub(packageTarget: str):
	pathPytype = pathScratchPad / 'pytype'
	pathPytype.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'pytype.single', '--output', str(pathPytype),
		f"{sys.prefix}/lib/site-packages/{packageTarget}/__init__.py"
	])

def generateStubDefaulterStub(packageTarget: str):
	pathStubDefaulter = pathScratchPad / 'stubdefaulter'
	pathStubDefaulter.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'stubdefaulter', packageTarget,
		'--output-dir', str(pathStubDefaulter)
	])

def generateStubgenPyxStub(packageTarget: str):
	pathStubgenPyx = pathScratchPad / 'stubgen_pyx'
	pathStubgenPyx.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'stubgen_pyx',
		f"{sys.prefix}/lib/site-packages/{packageTarget}",
		'-o', str(pathStubgenPyx)
	])

def generateStubGeneratorStub(packageTarget: str):
	pathStubGenerator = pathScratchPad / 'stub_generator'
	pathStubGenerator.mkdir(exist_ok=True)

	subprocess.run([
		sys.executable, '-m', 'stub_generator', packageTarget,
		'--out', str(pathStubGenerator)
	])

if __name__ == "__main__":
	# Allow custom package target via command-line argument
	if len(sys.argv) > 1:
		packageTarget = sys.argv[1]

	for directory in pathScratchPad.glob('*'):
		if directory.is_dir():
			shutil.rmtree(directory)

	# Generate stubs using different tools
	# generateAstStubgenStub(packageTarget)
	generateMonkeyTypeStub(packageTarget)
	generateMypyStub(packageTarget)
	generatePyrightStub(packageTarget)
	generatePytypeStub(packageTarget)
	generateStubDefaulterStub(packageTarget)
	generateStubgenPyxStub(packageTarget)
	generateStubGeneratorStub(packageTarget)

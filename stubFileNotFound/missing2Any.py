from libcst import matchers
from pathlib import Path
from stubFileNotFound import settingsPackage
import libcst

"""
'Fix' unknown types with `Any` and `...`.
If necessary, add `from typing import Any`.

Type annotation is missing for parameter: add `Any`

Examples:
	Expected type arguments for generic class "Callable": add `[..., Any]`.
	Expected type arguments for generic class "ndarray": add `[Any, Any]`.
	Expected type arguments for generic class "BaseGroupBy": add the correct number of `Any` arguments.

Separate file handling from libcst operations.
Leverage existing packages, such as `mypy`, `pyright`, and `hunterMakesPy` for everything and anything: write as little new code as possible.
"""

pathStubs = Path(settingsPackage.pathPackage / '..' / 'stubs').resolve()
pathSuffix: str = 'pyi'

listRelativePaths: list[str] = ['pandas']

def discoverStubFiles() -> list[Path]:
	"""Discover all stub files based on configured paths and suffix."""
	listPathFilenamesStub: list[Path] = []

	for relativePathTarget in listRelativePaths:
		pathTargetStubs = pathStubs / relativePathTarget
		if pathTargetStubs.exists():
			listPathFilenamesStub.extend(pathTargetStubs.rglob(f'*.{pathSuffix}'))

	return listPathFilenamesStub

class TypingImportAdder(libcst.CSTTransformer):
	"""Add 'from typing import Any' if not already present and Any is needed."""

	def __init__(self) -> None:
		super().__init__()
		self.hasTypingAnyImport: bool = False
		self.needsTypingAnyImport: bool = False

	def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
		"""Check if 'from typing import Any' already exists."""
		if (node.module and
			(matchers.matches(node.module, matchers.Attribute(value=matchers.Name("typing"))) or
			matchers.matches(node.module, matchers.Name("typing")))):

			if node.names and not isinstance(node.names, libcst.ImportStar):
				for nameImport in node.names:
					if isinstance(nameImport, libcst.ImportAlias):
						if isinstance(nameImport.name, libcst.Name) and nameImport.name.value == "Any":
							self.hasTypingAnyImport = True

	def leave_ImportFrom(self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom) -> libcst.ImportFrom:
		"""Update typing import to include Any if needed."""
		if (not self.hasTypingAnyImport and self.needsTypingAnyImport and
			updated_node.module and
			(matchers.matches(updated_node.module, matchers.Attribute(value=matchers.Name("typing"))) or
			matchers.matches(updated_node.module, matchers.Name("typing")))):

			if updated_node.names and not isinstance(updated_node.names, libcst.ImportStar):
				listNamesUpdated: list[libcst.ImportAlias] = list(updated_node.names)
				listNamesUpdated.append(libcst.ImportAlias(name=libcst.Name("Any")))
				self.hasTypingAnyImport = True
				return updated_node.with_changes(names=listNamesUpdated)

		return updated_node

	def visit_FunctionDef(self, node: libcst.FunctionDef) -> None:
		"""Check if function parameters need Any annotations."""
		for indexParameter, parameterFunction in enumerate(node.params.params):
			if parameterFunction.annotation is None and not self._isImplicitParameter(parameterFunction, indexParameter):
				self.needsTypingAnyImport = True

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Add Any annotations to parameters that lack type annotations."""
		listParametersUpdated: list[libcst.Param] = []
		wasParameterModified: bool = False

		for indexParameter, parameterFunction in enumerate(updated_node.params.params):
			if parameterFunction.annotation is None and not self._isImplicitParameter(parameterFunction, indexParameter):
				parameterUpdated = parameterFunction.with_changes(
					annotation=libcst.Annotation(annotation=libcst.Name("Any"))
				)
				listParametersUpdated.append(parameterUpdated)
				wasParameterModified = True
				self.needsTypingAnyImport = True
			else:
				listParametersUpdated.append(parameterFunction)

		if wasParameterModified:
			parametersUpdated = updated_node.params.with_changes(params=listParametersUpdated)
			return updated_node.with_changes(params=parametersUpdated)

		return updated_node

	def _isImplicitParameter(self, parameterFunction: libcst.Param, indexParameter: int) -> bool:
		"""Check if parameter is an implicit parameter (self, cls) that should not be annotated."""
		if indexParameter == 0 and isinstance(parameterFunction.name, libcst.Name):
			nameParameter = parameterFunction.name.value
			# Skip 'self' and 'cls' as first parameters
			return nameParameter in ("self", "cls")
		return False

	def leave_Module(self, original_node: libcst.Module, updated_node: libcst.Module) -> libcst.Module:
		"""Add typing import at the top if needed and not already present."""
		if self.needsTypingAnyImport and not self.hasTypingAnyImport:
			importStatement = libcst.SimpleStatementLine(
				body=[libcst.ImportFrom(
					module=libcst.Name("typing"),
					names=[libcst.ImportAlias(name=libcst.Name("Any"))]
				)]
			)

			# Find the best position to insert the import
			listBodyUpdated: list[libcst.CSTNode] = list(updated_node.body)
			indexInsert: int = 0

			# Skip docstrings and existing imports
			for index, statement in enumerate(listBodyUpdated):
				if isinstance(statement, libcst.SimpleStatementLine):
					if any(isinstance(stmt, (libcst.Import, libcst.ImportFrom)) for stmt in statement.body):
						indexInsert = index + 1
					else:
						break
				elif not isinstance(statement, libcst.SimpleStatementLine):
					break

			listBodyUpdated.insert(indexInsert, importStatement)
			return updated_node.with_changes(body=listBodyUpdated)

		return updated_node

class ImplicitParameterCleaner(libcst.CSTTransformer):
	"""Remove incorrect Any annotations from self and cls parameters."""

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Remove Any annotations from self and cls parameters."""
		listParametersUpdated: list[libcst.Param] = []
		wasParameterModified: bool = False

		for indexParameter, parameterFunction in enumerate(updated_node.params.params):
			if self._shouldRemoveAnyAnnotation(parameterFunction, indexParameter):
				parameterUpdated = parameterFunction.with_changes(annotation=None)
				listParametersUpdated.append(parameterUpdated)
				wasParameterModified = True
			else:
				listParametersUpdated.append(parameterFunction)

		if wasParameterModified:
			parametersUpdated = updated_node.params.with_changes(params=listParametersUpdated)
			return updated_node.with_changes(params=parametersUpdated)

		return updated_node

	def _shouldRemoveAnyAnnotation(self, parameterFunction: libcst.Param, indexParameter: int) -> bool:
		"""Check if parameter has incorrect Any annotation that should be removed."""
		if (indexParameter == 0 and
			isinstance(parameterFunction.name, libcst.Name) and
			parameterFunction.annotation is not None):

			nameParameter = parameterFunction.name.value
			if nameParameter in ("self", "cls"):
				# Check if annotation is ': Any'
				if isinstance(parameterFunction.annotation, libcst.Annotation):
					if isinstance(parameterFunction.annotation.annotation, libcst.Name):
						return parameterFunction.annotation.annotation.value == "Any"
		return False

def processStubFile(pathFilenameStub: Path) -> None:
	"""Process a single stub file to add missing Any annotations."""
	try:
		contentOriginal: str = pathFilenameStub.read_text(encoding='utf-8')
		treeCST = libcst.parse_module(contentOriginal)

		transformerTyping = TypingImportAdder()
		treeModified = treeCST.visit(transformerTyping)

		if treeModified != treeCST:
			contentModified: str = treeModified.code
			pathFilenameStub.write_text(contentModified, encoding='utf-8')
			print(f"Updated: {pathFilenameStub}")
		else:
			print(f"No changes needed: {pathFilenameStub}")

	except Exception as ERRORmessage:
		print(f"Failed to process {pathFilenameStub}: {ERRORmessage}")

def processAllStubFiles() -> None:
	"""Process all discovered stub files."""
	listPathFilenamesStub = discoverStubFiles()

	if not listPathFilenamesStub:
		print(f"No stub files found in paths: {listRelativePaths}")
		return

	print(f"Found {len(listPathFilenamesStub)} stub files to process")

	for pathFilenameStub in listPathFilenamesStub:
		processStubFile(pathFilenameStub)

if __name__ == "__main__":
	processAllStubFiles()
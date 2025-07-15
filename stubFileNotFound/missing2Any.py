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

```
  c:/apps/stubFileNotFound/stubs/pandas/core/reshape/concat.pyi:125:18 - error: Expression of type "None" cannot be assigned to parameter of type "bool"
    "None" is not assignable to "bool" (reportArgumentType)
```

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

class GenericTypeArgumentAdder(libcst.CSTTransformer):
	"""Add missing type arguments to generic classes with appropriate Any equivalents."""

	def __init__(self) -> None:
		super().__init__()
		self.needsTypingAnyImport: bool = False
		self.currentContext: str = "unknown"
		self.insideSubscript: bool = False
		self.insideImport: bool = False
		# Mapping of generic types to their appropriate Any-equivalent type arguments
		self.genericTypeArgumentMapping: dict[str, list[libcst.BaseExpression]] = {
			'_IndexSliceTuple': [libcst.Name("Any")],
			'_PivotTableColumnsTypes': [libcst.Name("Any")],
			'_PivotTableIndexTypes': [libcst.Name("Any")],
			'_PivotTableValuesTypes': [libcst.Name("Any")],
			'AbstractSet': [libcst.Name("Any")],
			'AggFuncTypeDictFrame': [libcst.Name("Any")],
			'AggFuncTypeDictSeries': [libcst.Name("Any")],
			'AsyncContextManager': [libcst.Name("Any")],
			'AsyncGenerator': [libcst.Name("Any"), libcst.Name("Any")],
			'AsyncIterable': [libcst.Name("Any")],
			'AsyncIterator': [libcst.Name("Any")],
			'Awaitable': [libcst.Name("Any")],
			'AxesData': [libcst.Name("Any")],
			'BaseGroupBy': [libcst.Name("Any")],
			'Callable': [libcst.Ellipsis(), libcst.Name("Any")],
			'CategoricalIndex': [libcst.Name("Any")],
			'Collection': [libcst.Name("Any")],
			'Container': [libcst.Name("Any")],
			'ContextManager': [libcst.Name("Any")],
			'Coroutine': [libcst.Name("Any"), libcst.Name("Any"), libcst.Name("Any")],
			'DataFrameGroupBy': [libcst.Name("Any"), libcst.Name("Any")],
			'dict': [libcst.Name("Any"), libcst.Name("Any")],
			'frozenset': [libcst.Name("Any")],
			'Generator': [libcst.Name("Any"), libcst.Name("Any"), libcst.Name("Any")],
			'GroupByObjectNonScalar': [libcst.Name("Any")],
			'Index': [libcst.Name("Any")],
			'ItemsView': [libcst.Name("Any"), libcst.Name("Any")],
			'Iterable': [libcst.Name("Any")],
			'Iterator': [libcst.Name("Any")],
			'KeysView': [libcst.Name("Any")],
			'list': [libcst.Name("Any")],
			'Mapping': [libcst.Name("Any"), libcst.Name("Any")],
			'MutableMapping': [libcst.Name("Any"), libcst.Name("Any")],
			'MutableSequence': [libcst.Name("Any")],
			'MutableSet': [libcst.Name("Any")],
			'ndarray': [libcst.Name("Any"), libcst.Name("Any")],
			'NDArray': [libcst.Name("Any")],
			'Pattern': [libcst.Name("Any")],
			'recarray': [libcst.Name("Any"), libcst.Name("Any")],
			'ReplaceValue': [libcst.Name("Any"), libcst.Name("Any")],
			'Reversible': [libcst.Name("Any")],
			'Sequence': [libcst.Name("Any")],
			'SeriesGroupBy': [libcst.Name("Any"), libcst.Name("Any")],
			'set': [libcst.Name("Any")],
			'tuple': [libcst.Name("Any"), libcst.Ellipsis()],
			'ValuesView': [libcst.Name("Any")],
		}

	def visit_Subscript(self, node: libcst.Subscript) -> None:
		"""Track when we're inside a subscript to avoid transforming already-subscripted types."""
		self.insideSubscript = True

	def leave_Subscript(self, original_node: libcst.Subscript, updated_node: libcst.Subscript) -> libcst.Subscript:
		"""Reset subscript flag when leaving."""
		self.insideSubscript = False
		return updated_node

	def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
		"""Track when we're inside an import statement."""
		self.insideImport = True

	def leave_ImportFrom(self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom) -> libcst.ImportFrom:
		"""Reset import flag when leaving."""
		self.insideImport = False
		return updated_node

	def visit_Import(self, node: libcst.Import) -> None:
		"""Track when we're inside an import statement."""
		self.insideImport = True

	def leave_Import(self, original_node: libcst.Import, updated_node: libcst.Import) -> libcst.Import:
		"""Reset import flag when leaving."""
		self.insideImport = False
		return updated_node

	def visit_AnnAssign(self, node: libcst.AnnAssign) -> None:
		"""Track when we're visiting type annotations."""
		# Check if this is a TypeAlias assignment
		if (node.annotation and
			isinstance(node.annotation.annotation, libcst.Name) and
			node.annotation.annotation.value == "TypeAlias"):
			self.currentContext = "type_annotation"
		else:
			self.currentContext = "type_annotation"

	def leave_AnnAssign(self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign) -> libcst.AnnAssign:
		"""Reset context after leaving type annotations."""
		self.currentContext = "unknown"
		return updated_node

	def visit_Annotation(self, node: libcst.Annotation) -> None:
		"""Track when we're in annotation context."""
		self.currentContext = "type_annotation"

	def leave_Annotation(self, original_node: libcst.Annotation, updated_node: libcst.Annotation) -> libcst.Annotation:
		"""Reset context after leaving annotations."""
		self.currentContext = "unknown"
		return updated_node

	def visit_SimpleStatementLine(self, node: libcst.SimpleStatementLine) -> None:
		"""Track when we're in a type alias or other type-related statement."""
		for statement in node.body:
			if isinstance(statement, libcst.AnnAssign):
				if (statement.annotation and
					isinstance(statement.annotation.annotation, libcst.Name) and
					statement.annotation.annotation.value == "TypeAlias"):
					self.currentContext = "type_annotation"
			elif isinstance(statement, libcst.Assign):
				# Check if this is a type alias assignment without annotation
				for target in statement.targets:
					if isinstance(target.target, libcst.Name):
						# This could be a TypeAlias - we'll treat assignments in stub files as type-related
						self.currentContext = "type_annotation"

	def leave_SimpleStatementLine(self, original_node: libcst.SimpleStatementLine, updated_node: libcst.SimpleStatementLine) -> libcst.SimpleStatementLine:
		"""Reset context after leaving statements."""
		self.currentContext = "unknown"
		return updated_node

	def visit_FunctionDef(self, node: libcst.FunctionDef) -> None:
		"""Track when we're in function definition context."""
		self.currentContext = "function_def"

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Reset context after leaving function definition."""
		self.currentContext = "unknown"
		return updated_node

	def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name | libcst.Subscript:
		"""Transform bare generic type names to include appropriate type arguments."""
		nameType = updated_node.value

		# Only transform if:
		# 1. It's a known generic type
		# 2. We're in a type annotation context
		# 3. We're not already inside a subscript (already has type args)
		# 4. We're not in an import statement
		if (nameType in self.genericTypeArgumentMapping and
			self.currentContext == "type_annotation" and
			not self.insideSubscript and
			not self.insideImport):

			argumentsType = self.genericTypeArgumentMapping[nameType]
			self.needsTypingAnyImport = True

			return libcst.Subscript(
				value=updated_node,
				slice=[
					libcst.SubscriptElement(
						slice=libcst.Index(value=argumentType)
					)
					for argumentType in argumentsType
				]
			)

		return updated_node

class CombinedStubTransformer(libcst.CSTTransformer):
	"""Combined transformer that applies all stub file transformations."""

	def __init__(self) -> None:
		super().__init__()
		self.typingImportAdder = TypingImportAdder()
		self.genericTypeArgumentAdder = GenericTypeArgumentAdder()
		self.implicitParameterCleaner = ImplicitParameterCleaner()

	def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
		self.typingImportAdder.visit_ImportFrom(node)
		self.genericTypeArgumentAdder.visit_ImportFrom(node)

	def leave_ImportFrom(self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom) -> libcst.ImportFrom:
		updated_node = self.typingImportAdder.leave_ImportFrom(original_node, updated_node)
		return self.genericTypeArgumentAdder.leave_ImportFrom(original_node, updated_node)

	def visit_Import(self, node: libcst.Import) -> None:
		self.genericTypeArgumentAdder.visit_Import(node)

	def leave_Import(self, original_node: libcst.Import, updated_node: libcst.Import) -> libcst.Import:
		return self.genericTypeArgumentAdder.leave_Import(original_node, updated_node)

	def visit_Subscript(self, node: libcst.Subscript) -> None:
		self.genericTypeArgumentAdder.visit_Subscript(node)

	def leave_Subscript(self, original_node: libcst.Subscript, updated_node: libcst.Subscript) -> libcst.Subscript:
		return self.genericTypeArgumentAdder.leave_Subscript(original_node, updated_node)

	def visit_FunctionDef(self, node: libcst.FunctionDef) -> None:
		self.typingImportAdder.visit_FunctionDef(node)
		self.genericTypeArgumentAdder.visit_FunctionDef(node)

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		updated_node = self.typingImportAdder.leave_FunctionDef(original_node, updated_node)
		updated_node = self.implicitParameterCleaner.leave_FunctionDef(original_node, updated_node)
		return self.genericTypeArgumentAdder.leave_FunctionDef(original_node, updated_node)

	def visit_AnnAssign(self, node: libcst.AnnAssign) -> None:
		self.genericTypeArgumentAdder.visit_AnnAssign(node)

	def leave_AnnAssign(self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign) -> libcst.AnnAssign:
		return self.genericTypeArgumentAdder.leave_AnnAssign(original_node, updated_node)

	def visit_Annotation(self, node: libcst.Annotation) -> None:
		self.genericTypeArgumentAdder.visit_Annotation(node)

	def leave_Annotation(self, original_node: libcst.Annotation, updated_node: libcst.Annotation) -> libcst.Annotation:
		return self.genericTypeArgumentAdder.leave_Annotation(original_node, updated_node)

	def visit_SimpleStatementLine(self, node: libcst.SimpleStatementLine) -> None:
		self.genericTypeArgumentAdder.visit_SimpleStatementLine(node)

	def leave_SimpleStatementLine(self, original_node: libcst.SimpleStatementLine, updated_node: libcst.SimpleStatementLine) -> libcst.SimpleStatementLine:
		return self.genericTypeArgumentAdder.leave_SimpleStatementLine(original_node, updated_node)

	def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name | libcst.Subscript:
		return self.genericTypeArgumentAdder.leave_Name(original_node, updated_node)

	def leave_Module(self, original_node: libcst.Module, updated_node: libcst.Module) -> libcst.Module:
		# Check if any transformer needs typing import
		needsTypingImport = (
			self.typingImportAdder.needsTypingAnyImport or
			self.genericTypeArgumentAdder.needsTypingAnyImport
		)

		if needsTypingImport and not self.typingImportAdder.hasTypingAnyImport:
			self.typingImportAdder.needsTypingAnyImport = True
			return self.typingImportAdder.leave_Module(original_node, updated_node)

		return updated_node

def processStubFile(pathFilenameStub: Path) -> None:
	"""Process a single stub file to add missing Any annotations."""
	contentOriginal: str = pathFilenameStub.read_text(encoding='utf-8')
	treeCST = libcst.parse_module(contentOriginal)

	transformerCombined = CombinedStubTransformer()
	treeModified = treeCST.visit(transformerCombined)

	if treeModified != treeCST:
		contentModified: str = treeModified.code
		pathFilenameStub.write_text(contentModified, encoding='utf-8')
		print(f"Updated: {pathFilenameStub}")

def processAllStubFiles() -> None:
	"""Process all discovered stub files."""
	listPathFilenames = discoverStubFiles()

	for pathFilename in listPathFilenames:
		processStubFile(pathFilename)

if __name__ == "__main__":
	processAllStubFiles()
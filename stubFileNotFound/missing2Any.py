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

TODO:
- nested generics; e.g., list[Callable]
- apply to variables in classes
- TypeAlias
- after stubdefaulter: Expression of type "None" cannot be assigned to parameter of type "bool"

"""

# Configuration Settings
pathStubs = Path(settingsPackage.pathPackage / '..' / 'stubs').resolve()
pathSuffix: str = 'pyi'
listRelativePaths: list[str] = ['pandas']

# Generic Type Arguments Configuration
dictionaryGenericTypeArguments: dict[str, list[libcst.BaseExpression]] = {
	'_IndexSliceTuple': [libcst.Name("Any")],
	'_IsLeapYearProperty': [libcst.Name("Any")],
	'_PeriodProperties': [libcst.Name("Any"), libcst.Name("Any"), libcst.Name("Any"), libcst.Name("Any"), libcst.Name("Any")],
	'_PivotAggCallable': [libcst.Name("Any")],
	'_PivotTableColumnsTypes': [libcst.Name("Any")],
	'_PivotTableIndexTypes': [libcst.Name("Any")],
	'_PivotTableValuesTypes': [libcst.Name("Any")],
	'_PlotAccessorColor': [libcst.Name("Any")],
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
	'defaultdict': [libcst.Name("Any"), libcst.Name("Any")],
	'dict': [libcst.Name("Any"), libcst.Name("Any")],
	'FromStatement': [libcst.Name("Any")],
	'frozenset': [libcst.Name("Any")],
	'GroupBy': [libcst.Name("Any")],
	'GroupByObjectNonScalar': [libcst.Name("Any")],
	'Index': [libcst.Name("Any")],
	'Interval': [libcst.Name("Any")],
	'IntervalIndex': [libcst.Name("Any")],
	'IOHandles': [libcst.Name("Any")],
	'ItemsView': [libcst.Name("Any"), libcst.Name("Any")],
	'Iterable': [libcst.Name("Any")],
	'Iterator': [libcst.Name("Any")],
	'KeysView': [libcst.Name("Any")],
	'list': [libcst.Name("Any")],
	'ListLikeHashable': [libcst.Name("Any")],
	'Mapping': [libcst.Name("Any"), libcst.Name("Any")],
	'MutableMapping': [libcst.Name("Any"), libcst.Name("Any")],
	'MutableSequence': [libcst.Name("Any")],
	'MutableSet': [libcst.Name("Any")],
	'ndarray': [libcst.Name("Any"), libcst.Name("Any")],
	'NDArray': [libcst.Name("Any")],
	'OrderedDict': [libcst.Name("Any"), libcst.Name("Any")],
	'ParseDatesArg': [libcst.Name("Any"), libcst.Name("Any")],
	'Pattern': [libcst.Name("Any")],
	'recarray': [libcst.Name("Any"), libcst.Name("Any")],
	'ReplaceValue': [libcst.Name("Any"), libcst.Name("Any")],
	'Reversible': [libcst.Name("Any")],
	'Select': [libcst.Name("Any")],
	'Sequence': [libcst.Name("Any")],
	'SequenceNotStr': [libcst.Name("Any")],
	'SeriesGroupBy': [libcst.Name("Any"), libcst.Name("Any")],
	'set': [libcst.Name("Any")],
	'Subset': [libcst.Name("Any")],
	'tuple': [libcst.Name("Any"), libcst.Ellipsis()],
	'UsecolsArgType': [libcst.Name("Any")],
	'ValuesView': [libcst.Name("Any")],
}

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

# Core Transformer Classes
class TypingImportAdder(libcst.CSTTransformer):
	"""Add 'from typing import Any' if not already present and Any is needed."""

	def __init__(self) -> None:
		"""Initialize the transformer with tracking flags."""
		super().__init__()
		self.hasTypingAnyImport: bool = False
		self.needsTypingAnyImport: bool = False

	def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
		"""Check if 'from typing import Any' already exists.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.ImportFrom
			The import statement node to examine.

		"""
		if (node.module and
			(matchers.matches(node.module, matchers.Attribute(value=matchers.Name("typing"))) or
			matchers.matches(node.module, matchers.Name("typing")))):

			if node.names and not isinstance(node.names, libcst.ImportStar):
				for nameImport in node.names:
					if isinstance(nameImport, libcst.ImportAlias):
						if isinstance(nameImport.name, libcst.Name) and nameImport.name.value == "Any":
							self.hasTypingAnyImport = True

	def leave_ImportFrom(self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom) -> libcst.ImportFrom:
		"""Update typing import to include Any if needed.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.ImportFrom
			The original import statement node.
		updated_node : libcst.ImportFrom
			The potentially modified import statement node.

		Returns
		-------
		updatedImportStatement : libcst.ImportFrom
			The import statement with Any added if necessary.

		"""
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
		"""Check if function parameters or return need Any annotations.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.FunctionDef
			The function definition to analyze for missing annotations.

		"""
		for parameterFunction in [*node.params.params, *node.params.posonly_params, *node.params.kwonly_params]:
			if parameterFunction.annotation is None and parameterFunction.name.value not in ("self", "cls"):
				self.needsTypingAnyImport = True

		if node.params.star_arg and isinstance(node.params.star_arg, libcst.Param) and node.params.star_arg.annotation is None:
			self.needsTypingAnyImport = True

		if node.params.star_kwarg and node.params.star_kwarg.annotation is None:
			self.needsTypingAnyImport = True

		if node.returns is None:
			self.needsTypingAnyImport = True

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Add Any annotations to parameters and return that lack type annotations.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.FunctionDef
			The original function definition node.
		updated_node : libcst.FunctionDef
			The potentially modified function definition node.

		Returns
		-------
		functionDefinitionUpdated : libcst.FunctionDef
			The function definition with Any annotations added where needed.

		"""
		listParametersUpdated: list[libcst.Param] = []
		wasParameterModified: bool = False

		for parameterFunction in updated_node.params.params:
			if parameterFunction.annotation is None and parameterFunction.name.value not in ("self", "cls"):
				parameterUpdated = parameterFunction.with_changes(
					annotation=libcst.Annotation(annotation=libcst.Name("Any"))
				)
				listParametersUpdated.append(parameterUpdated)
				wasParameterModified = True
				self.needsTypingAnyImport = True
			else:
				listParametersUpdated.append(parameterFunction)

		listPosOnlyParamsUpdated: list[libcst.Param] = []
		for parameterFunction in updated_node.params.posonly_params:
			if parameterFunction.annotation is None and parameterFunction.name.value not in ("self", "cls"):
				parameterUpdated = parameterFunction.with_changes(
					annotation=libcst.Annotation(annotation=libcst.Name("Any"))
				)
				listPosOnlyParamsUpdated.append(parameterUpdated)
				wasParameterModified = True
				self.needsTypingAnyImport = True
			else:
				listPosOnlyParamsUpdated.append(parameterFunction)

		listKwOnlyParamsUpdated: list[libcst.Param] = []
		for parameterFunction in updated_node.params.kwonly_params:
			if parameterFunction.annotation is None and parameterFunction.name.value not in ("self", "cls"):
				parameterUpdated = parameterFunction.with_changes(
					annotation=libcst.Annotation(annotation=libcst.Name("Any"))
				)
				listKwOnlyParamsUpdated.append(parameterUpdated)
				wasParameterModified = True
				self.needsTypingAnyImport = True
			else:
				listKwOnlyParamsUpdated.append(parameterFunction)

		starArgUpdated = updated_node.params.star_arg
		if (starArgUpdated and
			isinstance(starArgUpdated, libcst.Param) and
			starArgUpdated.annotation is None):
			starArgUpdated = starArgUpdated.with_changes(
				annotation=libcst.Annotation(annotation=libcst.Name("Any"))
			)
			wasParameterModified = True
			self.needsTypingAnyImport = True

		starKwargUpdated = updated_node.params.star_kwarg
		if starKwargUpdated and starKwargUpdated.annotation is None:
			starKwargUpdated = starKwargUpdated.with_changes(
				annotation=libcst.Annotation(annotation=libcst.Name("Any"))
			)
			wasParameterModified = True
			self.needsTypingAnyImport = True

		returnsUpdated = updated_node.returns
		if returnsUpdated is None:
			returnsUpdated = libcst.Annotation(annotation=libcst.Name("Any"))
			wasParameterModified = True
			self.needsTypingAnyImport = True

		if wasParameterModified:
			parametersUpdated = updated_node.params.with_changes(
				params=listParametersUpdated,
				posonly_params=listPosOnlyParamsUpdated,
				kwonly_params=listKwOnlyParamsUpdated,
				star_arg=starArgUpdated,
				star_kwarg=starKwargUpdated
			)
			return updated_node.with_changes(params=parametersUpdated, returns=returnsUpdated)

		return updated_node

	def leave_Module(self, original_node: libcst.Module, updated_node: libcst.Module) -> libcst.Module:
		"""Add typing import at the top if needed and not already present.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Module
			The original module node.
		updated_node : libcst.Module
			The potentially modified module node.

		Returns
		-------
		moduleUpdated : libcst.Module
			The module with typing import added if necessary.

		"""
		if self.needsTypingAnyImport and not self.hasTypingAnyImport:
			importStatement = libcst.SimpleStatementLine(
				body=[libcst.ImportFrom(
					module=libcst.Name("typing"),
					names=[libcst.ImportAlias(name=libcst.Name("Any"))]
				)]
			)

			listBodyUpdated: list[libcst.CSTNode] = list(updated_node.body)
			indexInsert: int = 0

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
	"""Remove incorrect Any annotations from self and cls parameters.

	(AI generated docstring)

	This transformer cleans up cases where `self` and `cls` parameters have been
	incorrectly annotated with `Any` type hints.

	"""

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Remove Any annotations from self and cls parameters.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.FunctionDef
			The original function definition node.
		updated_node : libcst.FunctionDef
			The potentially modified function definition node.

		Returns
		-------
		functionDefinitionCleaned : libcst.FunctionDef
			The function definition with implicit parameter annotations removed.

		"""
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
		"""Check if parameter has incorrect Any annotation that should be removed.

		(AI generated docstring)

		Parameters
		----------
		parameterFunction : libcst.Param
			The parameter to examine.
		indexParameter : int
			The index position of the parameter in the function signature.

		Returns
		-------
		shouldRemoveAnnotation : bool
			True if the Any annotation should be removed from this parameter.

		"""
		if (indexParameter == 0 and
			isinstance(parameterFunction.name, libcst.Name) and
			parameterFunction.annotation is not None):

			nameParameter = parameterFunction.name.value
			if nameParameter in ("self", "cls"):
				if isinstance(parameterFunction.annotation, libcst.Annotation):
					if isinstance(parameterFunction.annotation.annotation, libcst.Name):
						return parameterFunction.annotation.annotation.value == "Any"
		return False

class GenericTypeArgumentAdder(libcst.CSTTransformer):
	"""Add missing type arguments to generic classes with appropriate Any equivalents.

	(AI generated docstring)

	This transformer identifies bare generic type names in type annotation contexts
	and adds the appropriate type arguments using the configured mapping.

	"""

	def __init__(self) -> None:
		"""Initialize the transformer with context tracking and configuration.

		(AI generated docstring)

		"""
		super().__init__()
		self.needsTypingAnyImport: bool = False
		self.currentContext: str = "unknown"
		self.insideSubscript: bool = False
		self.insideIndex: bool = False
		self.insideImport: bool = False
		self.insideTypeAlias: bool = False
		self.unionDepth: int = 0
		# Track names that are declared as TypeAlias in this module; never parameterize these
		self.setTypeAliasNames: set[str] = set()
		# Track the current TypeAlias target name when inside an alias definition
		self.currentTypeAliasTarget: str | None = None

	def visit_Subscript(self, node: libcst.Subscript) -> None:
		"""Track when we're inside a subscript to avoid transforming already-subscripted types.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.Subscript
			The subscript node being visited.

		"""
		self.insideSubscript = True

	def leave_Subscript(self, original_node: libcst.Subscript, updated_node: libcst.Subscript) -> libcst.BaseExpression:
		"""Reset subscript flag when leaving.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Subscript
			The original subscript node.
		updated_node : libcst.Subscript
			The potentially modified subscript node.

		Returns
		-------
		subscriptNode : libcst.Subscript
			The subscript node unchanged.

		"""
		# Normalize chained subscripts like Sequence[Any][T] -> Sequence[T]
		normalized_sub = self._normalizeSubscriptChain(updated_node)
		self.insideSubscript = False
		return normalized_sub

	def visit_BinaryOperation(self, node: libcst.BinaryOperation) -> None:
		# Track PEP 604 unions ("|") in type annotations
		if isinstance(node.operator, libcst.BitOr):
			self.unionDepth += 1

	def leave_BinaryOperation(self, original_node: libcst.BinaryOperation, updated_node: libcst.BinaryOperation) -> libcst.BinaryOperation:
		if isinstance(original_node.operator, libcst.BitOr):
			# In union context, ensure each side's bare generics are parameterized
			left = updated_node.left
			right = updated_node.right
			# Only wrap when appropriate: either top-level unions (not inside subscript)
			# or unions used as type arguments (inside an Index)
			if (
				self.currentContext in ("type_annotation", "function_def")
				and not self.insideImport
				and (
					not self.insideSubscript or self.insideIndex
				)
			):
				if not isinstance(left, libcst.Subscript):
					left, _ = self.wrapGenericIfBare(left)
				if not isinstance(right, libcst.Subscript):
					right, _ = self.wrapGenericIfBare(right)
				updated_node = updated_node.with_changes(left=left, right=right)

			self.unionDepth = max(0, self.unionDepth - 1)
			return updated_node
		return updated_node

	def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
		"""Track when we're inside an import statement.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.ImportFrom
			The import from statement being visited.

		"""
		self.insideImport = True

	def leave_ImportFrom(self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom) -> libcst.ImportFrom:
		"""Reset import flag when leaving.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.ImportFrom
			The original import from node.
		updated_node : libcst.ImportFrom
			The potentially modified import from node.

		Returns
		-------
		importFromNode : libcst.ImportFrom
			The import from node unchanged.

		"""
		self.insideImport = False
		return updated_node

	def visit_Import(self, node: libcst.Import) -> None:
		"""Track when we're inside an import statement.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.Import
			The import statement being visited.

		"""
		self.insideImport = True

	def leave_Import(self, original_node: libcst.Import, updated_node: libcst.Import) -> libcst.Import:
		"""Reset import flag when leaving.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Import
			The original import node.
		updated_node : libcst.Import
			The potentially modified import node.

		Returns
		-------
		importNode : libcst.Import
			The import node unchanged.

		"""
		self.insideImport = False
		return updated_node

	def visit_AnnAssign(self, node: libcst.AnnAssign) -> None:
		"""Track when we're visiting type annotations.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.AnnAssign
			The annotated assignment being visited.

		"""
		if (node.annotation and
			isinstance(node.annotation.annotation, libcst.Name) and
			node.annotation.annotation.value == "TypeAlias"):
			self.currentContext = "type_annotation"
			self.insideTypeAlias = True
			# Collect the target name for type alias declarations
			if isinstance(node.target, libcst.Name):
				self.setTypeAliasNames.add(node.target.value)
				self.currentTypeAliasTarget = node.target.value
		else:
			self.currentContext = "type_annotation"

	def leave_AnnAssign(self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign) -> libcst.AnnAssign:
		"""Reset context after leaving type annotations.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.AnnAssign
			The original annotated assignment node.
		updated_node : libcst.AnnAssign
			The potentially modified annotated assignment node.

		Returns
		-------
		annotatedAssignmentNode : libcst.AnnAssign
			The annotated assignment node unchanged.

		"""
		# If this is a TypeAlias declaration, post-process the RHS to parameterize
		# any bare generics (including generic type aliases) that need type args.
		is_type_alias = (
			updated_node.annotation
			and isinstance(updated_node.annotation.annotation, libcst.Name)
			and updated_node.annotation.annotation.value == "TypeAlias"
		)
		if is_type_alias and updated_node.value is not None:
			new_value = self._wrapGenericsInTypeAliasValue(updated_node.value)
			if new_value is not updated_node.value:
				updated_node = updated_node.with_changes(value=new_value)

		self.currentContext = "unknown"
		self.insideTypeAlias = False
		self.currentTypeAliasTarget = None
		return updated_node

	def visit_Annotation(self, node: libcst.Annotation) -> None:
		"""Track when we're in annotation context.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.Annotation
			The annotation being visited.

		"""
		self.currentContext = "type_annotation"

	def leave_Annotation(self, original_node: libcst.Annotation, updated_node: libcst.Annotation) -> libcst.Annotation:
		"""Reset context after leaving annotations.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Annotation
			The original annotation node.
		updated_node : libcst.Annotation
			The potentially modified annotation node.

		Returns
		-------
		annotationNode : libcst.Annotation
			The annotation node unchanged.

		"""
		self.currentContext = "unknown"
		return updated_node

	def visit_SimpleStatementLine(self, node: libcst.SimpleStatementLine) -> None:
		"""Track when we're in a type alias or other type-related statement.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.SimpleStatementLine
			The simple statement line being visited.

		"""
		for statement in node.body:
			if isinstance(statement, libcst.AnnAssign):
				if (statement.annotation and
					isinstance(statement.annotation.annotation, libcst.Name) and
					statement.annotation.annotation.value == "TypeAlias"):
					self.currentContext = "type_annotation"
					self.insideTypeAlias = True
					# Register the name as a type alias
					if isinstance(statement.target, libcst.Name):
						self.setTypeAliasNames.add(statement.target.value)
						self.currentTypeAliasTarget = statement.target.value
			elif isinstance(statement, libcst.Assign):
				for target in statement.targets:
					if isinstance(target.target, libcst.Name):
						self.currentContext = "type_annotation"

	def leave_SimpleStatementLine(self, original_node: libcst.SimpleStatementLine, updated_node: libcst.SimpleStatementLine) -> libcst.SimpleStatementLine:
		"""Reset context after leaving statements.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.SimpleStatementLine
			The original simple statement line node.
		updated_node : libcst.SimpleStatementLine
			The potentially modified simple statement line node.

		Returns
		-------
		simpleStatementLineNode : libcst.SimpleStatementLine
			The simple statement line node unchanged.

		"""
		self.currentContext = "unknown"
		self.insideTypeAlias = False
		return updated_node

	def visit_FunctionDef(self, node: libcst.FunctionDef) -> None:
		"""Track when we're in function definition context.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.FunctionDef
			The function definition being visited.

		"""
		self.currentContext = "function_def"

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Reset context after leaving function definition.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.FunctionDef
			The original function definition node.
		updated_node : libcst.FunctionDef
			The potentially modified function definition node.

		Returns
		-------
		functionDefinitionNode : libcst.FunctionDef
			The function definition node unchanged.

		"""
		self.currentContext = "unknown"
		return updated_node

	def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name | libcst.Subscript:
		"""Transform bare generic type names to include appropriate type arguments.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Name
			The original name node.
		updated_node : libcst.Name
			The potentially modified name node.

		Returns
		-------
		typeNode : libcst.Name | libcst.Subscript
			Either the original name or a subscript with type arguments.

		"""
		nameType = updated_node.value

		if (nameType in dictionaryGenericTypeArguments and
			self.currentContext in ("type_annotation", "function_def") and
			not self.insideImport and
			not self.insideSubscript and
			(
				not self.insideTypeAlias or
				(self.currentTypeAliasTarget is None or nameType != self.currentTypeAliasTarget)
			)):

			argumentsType = dictionaryGenericTypeArguments[nameType]
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

	def leave_Attribute(self, original_node: libcst.Attribute, updated_node: libcst.Attribute) -> libcst.Attribute | libcst.Subscript:
		# Transform bare attribute generics (e.g., np.ndarray) similarly to Name, incl. inside unions
		# Allow this inside TypeAlias bodies as well (but never for the alias target itself)
		if (self.currentContext in ("type_annotation", "function_def") and
			not self.insideImport and
			not self.insideSubscript):
			baseName = self.extractBaseTypeNameFromExpression(updated_node)
			# Do not parameterize attribute if it refers to a recorded type alias name
			# Also skip if we're inside a TypeAlias and the baseName equals the current alias target
			if (baseName and
				baseName in dictionaryGenericTypeArguments and
				baseName not in self.setTypeAliasNames and
				(not self.insideTypeAlias or (self.currentTypeAliasTarget is None or baseName != self.currentTypeAliasTarget))):
				args = dictionaryGenericTypeArguments[baseName]
				self.needsTypingAnyImport = True
				return libcst.Subscript(
					value=updated_node,
					slice=[libcst.SubscriptElement(slice=libcst.Index(value=a)) for a in args],
				)

		return updated_node

	def wrapGenericIfBare(self, expression: libcst.BaseExpression) -> tuple[libcst.BaseExpression, bool]:
		# Only wrap plain Name/Attribute that match known generics
		if isinstance(expression, (libcst.Name, libcst.Attribute)):
			baseName = self.extractBaseTypeNameFromExpression(expression)
			if baseName and baseName in dictionaryGenericTypeArguments:
				args = dictionaryGenericTypeArguments[baseName]
				self.needsTypingAnyImport = True
				return (
					libcst.Subscript(
						value=expression,
						slice=[libcst.SubscriptElement(slice=libcst.Index(value=a)) for a in args],
					),
					True,
				)

		return (expression, False)

	# --- Nested generic handling ---
	def extractBaseTypeNameFromExpression(self, expression: libcst.BaseExpression) -> str | None:
		if isinstance(expression, libcst.Name):
			return expression.value
		if isinstance(expression, libcst.Attribute):
			if isinstance(expression.attr, libcst.Name):
				return expression.attr.value
		return None

	def selectTypeArgumentsForType(self, typeName: str) -> list[libcst.BaseExpression] | None:
		if typeName in dictionaryGenericTypeArguments:
			return dictionaryGenericTypeArguments[typeName]
		return None

	def visit_Index(self, node: libcst.Index) -> None:
		self.insideIndex = True

	def leave_Index(self, original_node: libcst.Index, updated_node: libcst.Index) -> libcst.Index:
		# We're leaving an index (type argument position) context
		self.insideIndex = False
		valueExpression = updated_node.value
		if isinstance(valueExpression, (libcst.Name, libcst.Attribute)):
			typeName = self.extractBaseTypeNameFromExpression(valueExpression)
			if typeName is not None and typeName not in self.setTypeAliasNames:
				argumentsType = self.selectTypeArgumentsForType(typeName)
				if argumentsType is not None:
					newValue = libcst.Subscript(
						value=valueExpression,
						slice=[libcst.SubscriptElement(slice=libcst.Index(value=argumentType)) for argumentType in argumentsType],
					)
					self.needsTypingAnyImport = True
					return updated_node.with_changes(value=newValue)
		return updated_node

	def leave_ClassDef(self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef) -> libcst.ClassDef:
		# Handle missing type arguments when a generic class is used as a base.
		if not updated_node.bases:
			return updated_node

		basesUpdated: list[libcst.Arg] = []
		modified = False
		for baseArg in updated_node.bases:
			baseValue: libcst.BaseExpression = baseArg.value
			# If base is already subscripted (e.g., Base[T, U]), leave it unchanged
			if isinstance(baseValue, libcst.Subscript):
				basesUpdated.append(baseArg)
				continue

			# If base is Name or Attribute, check if it's a known generic
			if isinstance(baseValue, (libcst.Name, libcst.Attribute)):
				baseName = self.extractBaseTypeNameFromExpression(baseValue)
				if baseName is not None and baseName not in self.setTypeAliasNames:
					argsForBase = self.selectTypeArgumentsForType(baseName)
					if argsForBase is not None:
						newBase = libcst.Subscript(
							value=baseValue,
							slice=[libcst.SubscriptElement(slice=libcst.Index(value=a)) for a in argsForBase],
						)
						basesUpdated.append(baseArg.with_changes(value=newBase))
						modified = True
						self.needsTypingAnyImport = True
						continue
			# Otherwise unchanged
			basesUpdated.append(baseArg)

		if modified:
			return updated_node.with_changes(bases=basesUpdated)
		return updated_node

	def _normalizeSubscriptChain(self, node: libcst.Subscript) -> libcst.BaseExpression:
		"""Collapse chained subscriptions on the same base type into a single subscription.

		Examples:
		- Sequence[Any][T] -> Sequence[T]
		- Mapping[Any, Any][K, V] -> Mapping[K, V]
		- Callable[..., Any][[X], Y] -> Callable[[X], Y]
		Note: If the base is a declared TypeAlias, return the node unchanged.
		"""
		# Flatten the chain of subscripts
		flattened_slices: list[list[libcst.SubscriptElement]] = []
		base_expr: libcst.BaseExpression | libcst.Subscript = node
		while isinstance(base_expr, libcst.Subscript):
			slice_list: list[libcst.SubscriptElement] = list(base_expr.slice)
			flattened_slices.append(slice_list)
			base_expr = base_expr.value

		base_name = self.extractBaseTypeNameFromExpression(base_expr) if isinstance(base_expr, (libcst.Name, libcst.Attribute)) else None
		if base_name is None:
			return node

		# Note: Declared TypeAlias names may themselves be generic; do not drop subscripts here.

		# Choose the last slice list with any element that's not a bare Any; else the first
		def is_meaningful(element: libcst.SubscriptElement) -> bool:
			val: libcst.BaseSlice = element.slice
			if isinstance(val, libcst.Index):
				inner: libcst.BaseExpression = val.value
				return not (isinstance(inner, libcst.Name) and inner.value == "Any")
			return True

		chosen: list[libcst.SubscriptElement] | None = None
		for slice_list in reversed(flattened_slices):
			if any(is_meaningful(el) for el in slice_list):
				chosen = slice_list
				break
		if chosen is None and flattened_slices:
			chosen = flattened_slices[0]
		if chosen is None:
			return node

		return libcst.Subscript(value=base_expr, slice=chosen)

	def _wrapGenericsInTypeAliasValue(self, expr: libcst.BaseExpression) -> libcst.BaseExpression:
		"""Recursively wrap bare generic names/attributes on the RHS of a TypeAlias.

		Skips already-subscripted expressions and the current alias target name.
		"""
		# Name or Attribute: try wrapping
		if isinstance(expr, (libcst.Name, libcst.Attribute)):
			base_name = self.extractBaseTypeNameFromExpression(expr)
			if (
				base_name
				and base_name in dictionaryGenericTypeArguments
				and (self.currentTypeAliasTarget is None or base_name != self.currentTypeAliasTarget)
			):
				wrapped, did = self.wrapGenericIfBare(expr)
				return wrapped if did else expr
			return expr
		# Binary union (PEP 604): process both sides
		if isinstance(expr, libcst.BinaryOperation) and isinstance(expr.operator, libcst.BitOr):
			left_new: libcst.BaseExpression = self._wrapGenericsInTypeAliasValue(expr.left)
			right_new: libcst.BaseExpression = self._wrapGenericsInTypeAliasValue(expr.right)
			if left_new is not expr.left or right_new is not expr.right:
				return expr.with_changes(left=left_new, right=right_new)
			return expr
		# Subscript: recurse into slice elements
		if isinstance(expr, libcst.Subscript):
			slice_list: list[libcst.SubscriptElement] = []
			changed = False
			for el in expr.slice:
				if isinstance(el.slice, libcst.Index):
					inner: libcst.BaseExpression = el.slice.value
					inner_new: libcst.BaseExpression = self._wrapGenericsInTypeAliasValue(inner)
					if inner_new is not inner:
						changed = True
						el: libcst.SubscriptElement = el.with_changes(slice=libcst.Index(value=inner_new))
				slice_list.append(el)
			if changed:
				return expr.with_changes(slice=slice_list)
			return expr
		# Tuples: map elements
		if isinstance(expr, libcst.Tuple):
			elts_new: list[libcst.BaseElement] = []
			changed = False
			for elDTuple in expr.elements:
				val = elDTuple.value
				val_new: libcst.BaseExpression = self._wrapGenericsInTypeAliasValue(val)
				if val_new is not val:
					changed = True
					elDTuple: libcst.BaseElement = elDTuple.with_changes(value=val_new)
				elts_new.append(elDTuple)
			if changed:
				return expr.with_changes(elements=elts_new)
			return expr
		# Parentheses or other wrappers aren’t common here; return as-is
		return expr

class CombinedStubTransformer(libcst.CSTTransformer):
	"""Combined transformer that applies all stub file transformations.

	(AI generated docstring)

	This transformer coordinates multiple individual transformers to provide
	comprehensive stub file processing in a single pass.

	"""

	def __init__(self) -> None:
		"""Initialize the combined transformer with all component transformers.

		(AI generated docstring)

		"""
		super().__init__()
		self.typingImportAdder = TypingImportAdder()
		self.genericTypeArgumentAdder = GenericTypeArgumentAdder()
		self.implicitParameterCleaner = ImplicitParameterCleaner()

	def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
		"""Coordinate visit_ImportFrom across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.ImportFrom
			The import from statement being visited.

		"""
		self.typingImportAdder.visit_ImportFrom(node)
		self.genericTypeArgumentAdder.visit_ImportFrom(node)

	def leave_ImportFrom(self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom) -> libcst.ImportFrom:
		"""Coordinate leave_ImportFrom across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.ImportFrom
			The original import from node.
		updated_node : libcst.ImportFrom
			The potentially modified import from node.

		Returns
		-------
		importFromUpdated : libcst.ImportFrom
			The import from node after all transformations.

		"""
		updated_node = self.typingImportAdder.leave_ImportFrom(original_node, updated_node)
		return self.genericTypeArgumentAdder.leave_ImportFrom(original_node, updated_node)

	def visit_Import(self, node: libcst.Import) -> None:
		"""Coordinate visit_Import across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.Import
			The import statement being visited.

		"""
		self.genericTypeArgumentAdder.visit_Import(node)

	def leave_Import(self, original_node: libcst.Import, updated_node: libcst.Import) -> libcst.Import:
		"""Coordinate leave_Import across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Import
			The original import node.
		updated_node : libcst.Import
			The potentially modified import node.

		Returns
		-------
		importUpdated : libcst.Import
			The import node after all transformations.

		"""
		return self.genericTypeArgumentAdder.leave_Import(original_node, updated_node)

	def visit_Subscript(self, node: libcst.Subscript) -> None:
		"""Coordinate visit_Subscript across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.Subscript
			The subscript being visited.

		"""
		self.genericTypeArgumentAdder.visit_Subscript(node)

	def leave_Subscript(self, original_node: libcst.Subscript, updated_node: libcst.Subscript) -> libcst.BaseExpression:
		"""Coordinate leave_Subscript across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Subscript
			The original subscript node.
		updated_node : libcst.Subscript
			The potentially modified subscript node.

		Returns
		-------
		subscriptUpdated : libcst.Subscript
			The subscript node after all transformations.

		"""
		return self.genericTypeArgumentAdder.leave_Subscript(original_node, updated_node)

	def visit_FunctionDef(self, node: libcst.FunctionDef) -> None:
		"""Coordinate visit_FunctionDef across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.FunctionDef
			The function definition being visited.

		"""
		self.typingImportAdder.visit_FunctionDef(node)
		self.genericTypeArgumentAdder.visit_FunctionDef(node)

	def leave_FunctionDef(self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef) -> libcst.FunctionDef:
		"""Coordinate leave_FunctionDef across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.FunctionDef
			The original function definition node.
		updated_node : libcst.FunctionDef
			The potentially modified function definition node.

		Returns
		-------
		functionDefinitionUpdated : libcst.FunctionDef
			The function definition after all transformations.

		"""
		updated_node = self.typingImportAdder.leave_FunctionDef(original_node, updated_node)
		updated_node = self.implicitParameterCleaner.leave_FunctionDef(original_node, updated_node)
		return self.genericTypeArgumentAdder.leave_FunctionDef(original_node, updated_node)

	def visit_AnnAssign(self, node: libcst.AnnAssign) -> None:
		"""Coordinate visit_AnnAssign across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.AnnAssign
			The annotated assignment being visited.

		"""
		self.genericTypeArgumentAdder.visit_AnnAssign(node)

	def leave_AnnAssign(self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign) -> libcst.AnnAssign:
		"""Coordinate leave_AnnAssign across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.AnnAssign
			The original annotated assignment node.
		updated_node : libcst.AnnAssign
			The potentially modified annotated assignment node.

		Returns
		-------
		annotatedAssignmentUpdated : libcst.AnnAssign
			The annotated assignment after all transformations.

		"""
		return self.genericTypeArgumentAdder.leave_AnnAssign(original_node, updated_node)

	def visit_Annotation(self, node: libcst.Annotation) -> None:
		"""Coordinate visit_Annotation across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.Annotation
			The annotation being visited.

		"""
		self.genericTypeArgumentAdder.visit_Annotation(node)

	def leave_Annotation(self, original_node: libcst.Annotation, updated_node: libcst.Annotation) -> libcst.Annotation:
		"""Coordinate leave_Annotation across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Annotation
			The original annotation node.
		updated_node : libcst.Annotation
			The potentially modified annotation node.

		Returns
		-------
		annotationUpdated : libcst.Annotation
			The annotation after all transformations.

		"""
		return self.genericTypeArgumentAdder.leave_Annotation(original_node, updated_node)

	def visit_SimpleStatementLine(self, node: libcst.SimpleStatementLine) -> None:
		"""Coordinate visit_SimpleStatementLine across all transformers.

		(AI generated docstring)

		Parameters
		----------
		node : libcst.SimpleStatementLine
			The simple statement line being visited.

		"""
		self.genericTypeArgumentAdder.visit_SimpleStatementLine(node)

	def leave_SimpleStatementLine(self, original_node: libcst.SimpleStatementLine, updated_node: libcst.SimpleStatementLine) -> libcst.SimpleStatementLine:
		"""Coordinate leave_SimpleStatementLine across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.SimpleStatementLine
			The original simple statement line node.
		updated_node : libcst.SimpleStatementLine
			The potentially modified simple statement line node.

		Returns
		-------
		simpleStatementLineUpdated : libcst.SimpleStatementLine
			The simple statement line after all transformations.

		"""
		return self.genericTypeArgumentAdder.leave_SimpleStatementLine(original_node, updated_node)

	def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name | libcst.Subscript:
		"""Coordinate leave_Name across all transformers.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Name
			The original name node.
		updated_node : libcst.Name
			The potentially modified name node.

		Returns
		-------
		nameOrSubscriptUpdated : libcst.Name | libcst.Subscript
			The name or subscript after all transformations.

		"""
		return self.genericTypeArgumentAdder.leave_Name(original_node, updated_node)

	def leave_Index(self, original_node: libcst.Index, updated_node: libcst.Index) -> libcst.Index:
		return self.genericTypeArgumentAdder.leave_Index(original_node, updated_node)

	def visit_Index(self, node: libcst.Index) -> None:
		self.genericTypeArgumentAdder.visit_Index(node)

	def leave_ClassDef(self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef) -> libcst.ClassDef:
		return self.genericTypeArgumentAdder.leave_ClassDef(original_node, updated_node)

	def visit_BinaryOperation(self, node: libcst.BinaryOperation) -> None:
		self.genericTypeArgumentAdder.visit_BinaryOperation(node)

	def leave_BinaryOperation(self, original_node: libcst.BinaryOperation, updated_node: libcst.BinaryOperation) -> libcst.BinaryOperation:
		return self.genericTypeArgumentAdder.leave_BinaryOperation(original_node, updated_node)

	def leave_Attribute(self, original_node: libcst.Attribute, updated_node: libcst.Attribute) -> libcst.Attribute | libcst.Subscript:
		return self.genericTypeArgumentAdder.leave_Attribute(original_node, updated_node)

	def leave_Module(self, original_node: libcst.Module, updated_node: libcst.Module) -> libcst.Module:
		"""Coordinate leave_Module across all transformers and handle import additions.

		(AI generated docstring)

		Parameters
		----------
		original_node : libcst.Module
			The original module node.
		updated_node : libcst.Module
			The potentially modified module node.

		Returns
		-------
		moduleUpdated : libcst.Module
			The module after all transformations and import additions.

		"""
		needsTypingImport = (
			self.typingImportAdder.needsTypingAnyImport or
			self.genericTypeArgumentAdder.needsTypingAnyImport
		)

		if needsTypingImport and not self.typingImportAdder.hasTypingAnyImport:
			self.typingImportAdder.needsTypingAnyImport = True
			return self.typingImportAdder.leave_Module(original_node, updated_node)

		return updated_node

# File Processing Functions
def processStubFile(pathFilenameStub: Path) -> None:
	"""Process a single stub file to add missing Any annotations.

	(AI generated docstring)

	Parameters
	----------
	pathFilenameStub : Path
		Path to the stub file to process.

	"""
	contentOriginal: str = pathFilenameStub.read_text(encoding='utf-8')
	treeCST = libcst.parse_module(contentOriginal)

	transformerCombined = CombinedStubTransformer()
	treeModified = treeCST.visit(transformerCombined)

	if treeModified != treeCST:
		contentModified: str = treeModified.code
		pathFilenameStub.write_text(contentModified, encoding='utf-8')
		print(f"Updated: {pathFilenameStub}")

def processAllStubFiles() -> None:
	"""Process all discovered stub files.

	(AI generated docstring)

	"""
	listPathFilenames = discoverStubFiles()

	for pathFilename in listPathFilenames:
		processStubFile(pathFilename)

# Main Execution
if __name__ == "__main__":
	processAllStubFiles()
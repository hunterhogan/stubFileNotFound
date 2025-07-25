from pandas.core.computation.ops import UndefinedVariableError as UndefinedVariableError

from pandas._config.config import OptionError as OptionError

from pandas._libs.tslibs import (
    OutOfBoundsDatetime as OutOfBoundsDatetime,
    OutOfBoundsTimedelta as OutOfBoundsTimedelta,
)
from typing import Any

class IntCastingNaNError(ValueError): ...
class NullFrequencyError(ValueError): ...
class PerformanceWarning(Warning): ...
class UnsupportedFunctionCall(ValueError): ...
class UnsortedIndexError(KeyError): ...
class ParserError(ValueError): ...
class DtypeWarning(Warning): ...
class EmptyDataError(ValueError): ...
class ParserWarning(Warning): ...
class MergeError(ValueError): ...
class AccessorRegistrationWarning(Warning): ...

class AbstractMethodError(NotImplementedError):
    def __init__(self, class_instance: Any, methodtype: str = 'method') -> None: ...

class NumbaUtilError(Exception): ...
class DuplicateLabelError(ValueError): ...
class InvalidIndexError(Exception): ...
class DataError(Exception): ...
class SpecificationError(Exception): ...
class SettingWithCopyError(ValueError): ...
class SettingWithCopyWarning(Warning): ...
class NumExprClobberingError(NameError): ...
class IndexingError(Exception): ...
class PyperclipException(RuntimeError): ...

class PyperclipWindowsException(PyperclipException):
    def __init__(self, message: Any) -> None: ...

class CSSWarning(UserWarning): ...
class PossibleDataLossError(Exception): ...
class ClosedFileError(Exception): ...
class IncompatibilityWarning(Warning): ...
class AttributeConflictWarning(Warning): ...
class DatabaseError(OSError): ...
class PossiblePrecisionLoss(Warning): ...
class ValueLabelTypeMismatch(Warning): ...
class InvalidColumnName(Warning): ...
class CategoricalConversionWarning(Warning): ...

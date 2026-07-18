from _typeshed import Incomplete
from numba.core import errors as errors, types as types
from numba.core.datamodel.models import PrimitiveModel as PrimitiveModel
from numba.core.datamodel.registry import register_default as register_default
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, infer_getattr as infer_getattr, infer_global as infer_global, signature as signature
from numba.np import npdatetime_helpers as npdatetime_helpers

numpy_version: Incomplete

class _NPDatetimeBase(types.Type):
    """
    Common base class for np.datetime64 and np.timedelta64.
    """
    unit: Incomplete
    unit_code: Incomplete
    def __init__(self, unit, *args, **kws) -> None: ...
    def __lt__(self, other): ...
    def cast_python_value(self, value): ...

class NPTimedelta(_NPDatetimeBase):
    type_name: str

class NPDatetime(_NPDatetimeBase):
    type_name: str

class NPDatetimeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None: ...

class TimedeltaUnaryOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaBinOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaCmpOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaOrderedCmpOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaMixOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        ({int, float}, timedelta64) -> timedelta64
        """

class TimedeltaDivOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        (timedelta64, timedelta64) -> float
        """

class TimedeltaUnaryPos(TimedeltaUnaryOp):
    key: Incomplete

class TimedeltaUnaryNeg(TimedeltaUnaryOp):
    key: Incomplete

class TimedeltaBinAdd(TimedeltaBinOp):
    key: Incomplete

class TimedeltaBinSub(TimedeltaBinOp):
    key: Incomplete

class TimedeltaBinMult(TimedeltaMixOp):
    key: Incomplete

class TimedeltaTrueDiv(TimedeltaDivOp):
    key: Incomplete

class TimedeltaFloorDiv(TimedeltaDivOp):
    key: Incomplete

class TimedeltaCmpEq(TimedeltaOrderedCmpOp):
    key: Incomplete

class TimedeltaCmpNe(TimedeltaOrderedCmpOp):
    key: Incomplete

class TimedeltaCmpEq(TimedeltaCmpOp):
    key: Incomplete

class TimedeltaCmpNe(TimedeltaCmpOp):
    key: Incomplete

class TimedeltaCmpLt(TimedeltaOrderedCmpOp):
    key: Incomplete

class TimedeltaCmpLE(TimedeltaOrderedCmpOp):
    key: Incomplete

class TimedeltaCmpGt(TimedeltaOrderedCmpOp):
    key: Incomplete

class TimedeltaCmpGE(TimedeltaOrderedCmpOp):
    key: Incomplete

class TimedeltaAbs(TimedeltaUnaryOp): ...

class DatetimePlusTimedelta(AbstractTemplate):
    key: Incomplete
    def generic(self, args, kws): ...

class DatetimeMinusTimedelta(AbstractTemplate):
    key: Incomplete
    def generic(self, args, kws): ...

class DatetimeMinusDatetime(AbstractTemplate):
    key: Incomplete
    def generic(self, args, kws): ...

class DatetimeCmpOp(AbstractTemplate):
    def generic(self, args, kws): ...

class DatetimeCmpEq(DatetimeCmpOp):
    key: Incomplete

class DatetimeCmpNe(DatetimeCmpOp):
    key: Incomplete

class DatetimeCmpLt(DatetimeCmpOp):
    key: Incomplete

class DatetimeCmpLE(DatetimeCmpOp):
    key: Incomplete

class DatetimeCmpGt(DatetimeCmpOp):
    key: Incomplete

class DatetimeCmpGE(DatetimeCmpOp):
    key: Incomplete

class DatetimeMinMax(AbstractTemplate):
    def generic(self, args, kws): ...

class NPTimedeltaAttribute(AttributeTemplate):
    key = NPTimedelta
    def resolve___class__(self, ty): ...

class NPDatetimeAttribute(AttributeTemplate):
    key = NPDatetime
    def resolve___class__(self, ty): ...

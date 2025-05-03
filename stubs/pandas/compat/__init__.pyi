import pandas as pandas
from . import _constants as _constants, _optional as _optional, compressors as compressors, numpy as numpy, pickle_compat as pickle_compat, pyarrow as pyarrow

__all__ = ['is_numpy_dev', 'pa_version_under10p1', 'pa_version_under11p0', 'pa_version_under13p0', 'pa_version_under14p0', 'pa_version_under14p1', 'pa_version_under16p0', 'pa_version_under17p0', 'IS64', 'ISMUSL', 'PY310', 'PY311', 'PY312', 'PYPY']

IS64: bool
ISMUSL: bool
PY310: bool
PY311: bool
PY312: bool
PYPY: bool
is_numpy_dev: bool
pa_version_under10p1: bool
pa_version_under11p0: bool
pa_version_under13p0: bool
pa_version_under14p0: bool
pa_version_under14p1: bool
pa_version_under16p0: bool
pa_version_under17p0: bool

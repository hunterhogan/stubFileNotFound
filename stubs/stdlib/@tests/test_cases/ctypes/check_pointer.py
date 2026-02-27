from __future__ import annotations

from typing import assert_type, Type
import ctypes

assert_type(ctypes.POINTER(None), type[ctypes.c_void_p])
assert_type(ctypes.POINTER(ctypes.c_int), type[ctypes._Pointer[ctypes.c_int]])

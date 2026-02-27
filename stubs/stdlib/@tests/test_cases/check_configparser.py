from configparser import RawConfigParser, SectionProxy
from typing import assert_type

sp = SectionProxy(RawConfigParser(), "")
assert_type(sp.get("foo", fallback="hi"), str)

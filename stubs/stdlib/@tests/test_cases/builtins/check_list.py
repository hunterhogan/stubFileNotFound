from __future__ import annotations

from typing import assert_type, List, Union

# list.__add__ example from #8292
class Foo:
    def asd(self) -> int:
        return 1


class Bar:
    def asd(self) -> int:
        return 2


combined = [Foo()] + [Bar()]
assert_type(combined, list[Foo | Bar])
for item in combined:
    assert_type(item.asd(), int)

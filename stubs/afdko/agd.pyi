from _typeshed import Incomplete

__doc__: str
re_entry: Incomplete
re_name: Incomplete

class glyph:
    name: Incomplete
    uni: Incomplete
    fin: Incomplete
    ali: Incomplete
    sub: Incomplete
    set: Incomplete
    min: Incomplete
    maj: Incomplete
    cmp: Incomplete
    other: Incomplete
    def __init__(self, name) -> None: ...
    def parse(self, intext): ...
    def check(self): ...
    def aliases(self, addglyphs=[]): ...
    def uniname(self): ...

class dictionary:
    list: Incomplete
    glyphs: Incomplete
    index: Incomplete
    unicode: Incomplete
    messages: Incomplete
    def __init__(self, intext=None) -> None: ...
    def parse(self, intext, priority: int = 1): ...
    def glyph(self, n): ...
    def remove(self, n): ...
    def add(self, g, priority: int = 1): ...

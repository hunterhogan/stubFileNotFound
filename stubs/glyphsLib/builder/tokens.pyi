from _typeshed import Incomplete

class TokenExpander:
    number_token_re: str
    glyph_predicate_re: str
    glyph_property_re: str
    bare_number_value_re: str
    font: Incomplete
    master: Incomplete
    def __init__(self, font, master) -> None: ...
    featurecode: Incomplete
    output: str
    position: int
    def expand(self, featurecode): ...
    def parse_token(self) -> None: ...
    def parse_bare_number_value(self, number): ...
    def parse_number_token(self, token, layer=None): ...
    def parse_glyph_property(self, token): ...
    originaltoken: Incomplete
    glyph_predicate: Incomplete
    def parse_glyph_predicate(self, token): ...
    gsglyph_predicate_objects: Incomplete
    gsglyph_predicate_object_re: Incomplete
    comparators_re: str
    apply_comparators: Incomplete

class PassThruExpander:
    def expand(self, token): ...


from babelfont.Font import Font
from fontTools.feaLib import ast
from fontTools.misc.visitor import Visitor
from typing import Set

logger = ...
def drop_unexported_glyphs(font: Font, args=...): # -> None:
    ...

def warn_about_used_glyphs(font: Font, unexported: set[str]): # -> None:
    ...

def fixup_used_glyphs(font: Font, unexported: set[str]): # -> None:
    ...

class FeaAppearsVisitor(Visitor):
    def __init__(self) -> None:
        ...



@FeaAppearsVisitor.register(ast.GlyphName)
def visit(visitor, gn, *args, **kwargs): # -> Literal[False]:
    ...

@FeaAppearsVisitor.register(ast.MarkClassDefinition)
def visit(visitor, mcd, *args, **kwargs): # -> Literal[False]:
    ...

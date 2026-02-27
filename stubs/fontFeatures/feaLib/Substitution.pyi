
from typing import Union
import fontTools.feaLib.ast as feaast

def glyphref(g) -> feaast.GlyphName | feaast.GlyphClass:
    ...

def has_classes(self) -> bool:
    ...

def all_classes_equal(self) -> bool:
    ...

def is_paired(self) -> bool:
    ...

def paired_ligature(self) -> feaast.LigatureSubstStatement:
    ...

def paired_mult(self) -> feaast.MultipleSubstStatement:
    ...

def asFeaAST(self): # -> Comment | MultipleSubstStatement | SingleSubstStatement | AlternateSubstStatement | LigatureSubstStatement | ReverseChainSingleSubstStatement:
    ...

from _typeshed import Incomplete
from fontTools import cffLib as cffLib, ttLib as ttLib
from fontTools.merge.base import add_method as add_method, mergeObjects as mergeObjects

log: Incomplete
headFlagsMergeBitMap: Incomplete
os2FsTypeMergeBitMap: Incomplete

def mergeOs2FsType(lst): ...
def merge(self, m, tables): ...

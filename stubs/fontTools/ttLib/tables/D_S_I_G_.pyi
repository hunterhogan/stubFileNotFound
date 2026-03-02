from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import (
	bytesjoin as bytesjoin, safeEval as safeEval, strjoin as strjoin, tobytes as tobytes, tostr as tostr)
from fontTools.ttLib import TTFont

DSIG_HeaderFormat: str
DSIG_SignatureFormat: str
DSIG_SignatureBlockFormat: str

class table_D_S_I_G_(DefaultTable.DefaultTable):
    """Digital Signature table

    The ``DSIG`` table contains cryptographic signatures for the font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/dsig
    """

    signatureRecords: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, xmlWriter, ttFont: TTFont) -> None: ...
    ulVersion: Incomplete
    usNumSigs: Incomplete
    usFlag: Incomplete
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

pem_spam: Incomplete

def b64encode(b): ...

class SignatureRecord:
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    ulFormat: Incomplete
    usReserved1: Incomplete
    usReserved2: Incomplete
    pkcs7: Incomplete
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

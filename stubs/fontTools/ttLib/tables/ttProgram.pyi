from _typeshed import Incomplete
from fontTools.misc.textTools import (
	binary2num as binary2num, num2binary as num2binary, readHex as readHex, strjoin as strjoin)
from fontTools.ttLib import TTFont

log: Incomplete
streamInstructions: Incomplete
instructions: Incomplete

def bitRepr(value, bits): ...

_mnemonicPat: Incomplete

def _makeDict(instructionList): ...

streamOpcodeDict: Incomplete
streamMnemonicDict: Incomplete
opcodeDict: Incomplete
mnemonicDict: Incomplete

class tt_instructions_error(Exception):
    error: Incomplete
    def __init__(self, error) -> None: ...

_comment: str
_instruction: str
_number: str
_token: Incomplete
_tokenRE: Incomplete
_whiteRE: Incomplete
_pushCountPat: Incomplete
_indentRE: Incomplete
_unindentRE: Incomplete

def _skipWhite(data, pos): ...

class Program:
    def __init__(self) -> None: ...
    bytecode: Incomplete
    def fromBytecode(self, bytecode: bytes) -> None: ...
    assembly: Incomplete
    def fromAssembly(self, assembly: list[str] | str) -> None: ...
    def getBytecode(self) -> bytes: ...
    def getAssembly(self, preserve: bool = True) -> list[str]: ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    def _assemble(self) -> None: ...
    def _disassemble(self, preserve: bool = False) -> None: ...
    def __bool__(self) -> bool:
        """
        >>> p = Program()
        >>> bool(p)
        False
        >>> bc = array.array("B", [0])
        >>> p.fromBytecode(bc)
        >>> bool(p)
        True
        >>> p.bytecode.pop()
        0
        >>> bool(p)
        False

        >>> p = Program()
        >>> asm = [\'SVTCA[0]\']
        >>> p.fromAssembly(asm)
        >>> bool(p)
        True
        >>> p.assembly.pop()
        \'SVTCA[0]\'
        >>> bool(p)
        False
        """
    __nonzero__ = __bool__
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...

def _test() -> None:
    """
    >>> _test()
    True
    """

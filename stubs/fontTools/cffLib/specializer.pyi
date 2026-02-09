from _typeshed import Incomplete
from collections.abc import Generator
from fontTools.cffLib import maxStackLimit as maxStackLimit

def stringToProgram(string): ...
def programToString(program): ...
def programToCommands(program, getNumRegions=None):
    """Takes a T2CharString program list and returns list of commands.
    Each command is a two-tuple of commandname,arg-list.  The commandname might
    be empty string if no commandname shall be emitted (used for glyph width,
    hintmask/cntrmask argument, as well as stray arguments at the end of the
    program (🤷).
    'getNumRegions' may be None, or a callable object. It must return the
    number of regions. 'getNumRegions' takes a single argument, vsindex. It
    returns the numRegions for the vsindex.
    The Charstring may or may not start with a width value. If the first
    non-blend operator has an odd number of arguments, then the first argument is
    a width, and is popped off. This is complicated with blend operators, as
    there may be more than one before the first hint or moveto operator, and each
    one reduces several arguments to just one list argument. We have to sum the
    number of arguments that are not part of the blend arguments, and all the
    'numBlends' values. We could instead have said that by definition, if there
    is a blend operator, there is no width value, since CFF2 Charstrings don't
    have width values. I discussed this with Behdad, and we are allowing for an
    initial width value in this case because developers may assemble a CFF2
    charstring from CFF Charstrings, which could have width values.
    """
def _flattenBlendArgs(args): ...
def commandsToProgram(commands):
    """Takes a commands list as returned by programToCommands() and converts
    it back to a T2CharString program list.
    """
def _everyN(el, n) -> Generator[Incomplete]:
    """Group the list el into groups of size n"""

class _GeneralizerDecombinerCommandsMap:
    @staticmethod
    def rmoveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def hmoveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def vmoveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def rlineto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def hlineto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def vlineto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def rrcurveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def hhcurveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def vvcurveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def hvcurveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def vhcurveto(args) -> Generator[Incomplete]: ...
    @staticmethod
    def rcurveline(args) -> Generator[Incomplete]: ...
    @staticmethod
    def rlinecurve(args) -> Generator[Incomplete]: ...

def _convertBlendOpToArgs(blendList): ...
def generalizeCommands(commands, ignoreErrors: bool = False): ...
def generalizeProgram(program, getNumRegions=None, **kwargs): ...
def _categorizeVector(v):
    """
    Takes X,Y vector v and returns one of r, h, v, or 0 depending on which
    of X and/or Y are zero, plus tuple of nonzero ones.  If both are zero,
    it returns a single zero still.

    >>> _categorizeVector((0,0))
    ('0', (0,))
    >>> _categorizeVector((1,0))
    ('h', (1,))
    >>> _categorizeVector((0,2))
    ('v', (2,))
    >>> _categorizeVector((1,2))
    ('r', (1, 2))
    """
def _mergeCategories(a, b): ...
def _negateCategory(a): ...
def _convertToBlendCmds(args): ...
def _addArgs(a, b): ...
def _argsStackUse(args): ...
def specializeCommands(commands, ignoreErrors: bool = False, generalizeFirst: bool = True, preserveTopology: bool = False, maxstack: int = 48): ...
def specializeProgram(program, getNumRegions=None, **kwargs): ...

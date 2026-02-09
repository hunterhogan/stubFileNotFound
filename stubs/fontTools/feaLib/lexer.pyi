from fontTools.feaLib.error import FeatureLibError as FeatureLibError, IncludedFeaNotFound as IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation as FeatureLibLocation
from typing import Any, ClassVar
import re

__test__: dict

class IncludingLexer:
    """A Lexer that follows include statements.

    The OpenType feature file specification states that due to
    historical reasons, relative imports should be resolved in this
    order:

    1. If the source font is UFO format, then relative to the UFO's
       font directory
    2. relative to the top-level include file
    3. relative to the parent include file

    We only support 1 (via includeDir) and 2.
    """

    def __init__(self, featurefile, includeDir=...) -> Any:
        """IncludingLexer.__init__(self, featurefile, *, includeDir=None)

        Initializes an IncludingLexer.

        Behavior:
            If includeDir is passed, it will be used to determine the top-level
            include directory to use for all encountered include statements. If it is
            not passed, ``os.path.dirname(featurefile)`` will be considered the
            include directory.
        """
    @staticmethod
    def make_lexer_(file_or_path) -> Any:
        """IncludingLexer.make_lexer_(file_or_path)"""
    def next(self) -> Any:
        """IncludingLexer.next(self)"""
    def scan_anonymous_block(self, tag) -> Any:
        """IncludingLexer.scan_anonymous_block(self, tag)"""
    def __iter__(self) -> Any:
        """IncludingLexer.__iter__(self)"""
    def __next__(self) -> Any:
        """IncludingLexer.__next__(self)"""

class Lexer:
    ANONYMOUS_BLOCK: ClassVar[str] = ...
    CHAR_DIGIT_: ClassVar[str] = ...
    CHAR_HEXDIGIT_: ClassVar[str] = ...
    CHAR_LETTER_: ClassVar[str] = ...
    CHAR_NAME_CONTINUATION_: ClassVar[str] = ...
    CHAR_NAME_START_: ClassVar[str] = ...
    CHAR_NEWLINE_: ClassVar[str] = ...
    CHAR_SYMBOL_: ClassVar[str] = ...
    CHAR_WHITESPACE_: ClassVar[str] = ...
    CID: ClassVar[str] = ...
    COMMENT: ClassVar[str] = ...
    FILENAME: ClassVar[str] = ...
    FLOAT: ClassVar[str] = ...
    GLYPHCLASS: ClassVar[str] = ...
    HEXADECIMAL: ClassVar[str] = ...
    MODE_FILENAME_: ClassVar[str] = ...
    MODE_NORMAL_: ClassVar[str] = ...
    NAME: ClassVar[str] = ...
    NEWLINE: ClassVar[str] = ...
    NUMBER: ClassVar[str] = ...
    NUMBERS: ClassVar[tuple] = ...
    OCTAL: ClassVar[str] = ...
    RE_GLYPHCLASS: ClassVar[re.Pattern] = ...
    STRING: ClassVar[str] = ...
    SYMBOL: ClassVar[str] = ...
    def __init__(self, text, filename) -> Any:
        """Lexer.__init__(self, text, filename)"""
    def location_(self) -> Any:
        """Lexer.location_(self)"""
    def next(self) -> Any:
        """Lexer.next(self)"""
    def next_(self) -> Any:
        """Lexer.next_(self)"""
    def scan_anonymous_block(self, tag) -> Any:
        """Lexer.scan_anonymous_block(self, tag)"""
    def scan_over_(self, valid) -> Any:
        """Lexer.scan_over_(self, valid)"""
    def scan_until_(self, stop_at) -> Any:
        """Lexer.scan_until_(self, stop_at)"""
    def __iter__(self) -> Any:
        """Lexer.__iter__(self)"""
    def __next__(self) -> Any:
        """Lexer.__next__(self)"""

class NonIncludingLexer(IncludingLexer):
    """Lexer that does not follow `include` statements, emits them as-is."""

    def __next__(self) -> Any:
        """NonIncludingLexer.__next__(self)"""

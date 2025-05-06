import typing as T
from _typeshed import Incomplete

_log: Incomplete

class _Token:
    """
    A token in a PostScript stream.

    Attributes
    ----------
    pos : int
        Position, i.e. offset from the beginning of the data.
    raw : str
        Raw text of the token.
    kind : str
        Description of the token (for debugging or testing).
    """
    __slots__: Incomplete
    kind: str
    pos: Incomplete
    raw: Incomplete
    def __init__(self, pos, raw) -> None: ...
    def __str__(self) -> str: ...
    def endpos(self):
        """Position one past the end of the token"""
    def is_keyword(self, *names):
        """Is this a name token with one of the names?"""
    def is_slash_name(self):
        """Is this a name token that starts with a slash?"""
    def is_delim(self):
        """Is this a delimiter token?"""
    def is_number(self):
        """Is this a number token?"""
    def value(self): ...

class _NameToken(_Token):
    kind: str
    def is_slash_name(self): ...
    def value(self): ...

class _BooleanToken(_Token):
    kind: str
    def value(self): ...

class _KeywordToken(_Token):
    kind: str
    def is_keyword(self, *names): ...

class _DelimiterToken(_Token):
    kind: str
    def is_delim(self): ...
    def opposite(self): ...

class _WhitespaceToken(_Token):
    kind: str

class _StringToken(_Token):
    kind: str
    _escapes_re: Incomplete
    _replacements: Incomplete
    _ws_re: Incomplete
    @classmethod
    def _escape(cls, match): ...
    def value(self): ...

class _BinaryToken(_Token):
    kind: str
    def value(self): ...

class _NumberToken(_Token):
    kind: str
    def is_number(self): ...
    def value(self): ...

def _tokenize(data: bytes, skip_ws: bool) -> T.Generator[_Token, int, None]:
    """
    A generator that produces _Token instances from Type-1 font code.

    The consumer of the generator may send an integer to the tokenizer to
    indicate that the next token should be _BinaryToken of the given length.

    Parameters
    ----------
    data : bytes
        The data of the font to tokenize.

    skip_ws : bool
        If true, the generator will drop any _WhitespaceTokens from the output.
    """

class _BalancedExpression(_Token): ...

def _expression(initial, tokens, data):
    """
    Consume some number of tokens and return a balanced PostScript expression.

    Parameters
    ----------
    initial : _Token
        The token that triggered parsing a balanced expression.
    tokens : iterator of _Token
        Following tokens.
    data : bytes
        Underlying data that the token positions point to.

    Returns
    -------
    _BalancedExpression
    """

class Type1Font:
    """
    A class representing a Type-1 font, for use by backends.

    Attributes
    ----------
    parts : tuple
        A 3-tuple of the cleartext part, the encrypted part, and the finale of
        zeros.

    decrypted : bytes
        The decrypted form of ``parts[1]``.

    prop : dict[str, Any]
        A dictionary of font properties. Noteworthy keys include:

        - FontName: PostScript name of the font
        - Encoding: dict from numeric codes to glyph names
        - FontMatrix: bytes object encoding a matrix
        - UniqueID: optional font identifier, dropped when modifying the font
        - CharStrings: dict from glyph names to byte code
        - Subrs: array of byte code subroutines
        - OtherSubrs: bytes object encoding some PostScript code
    """
    __slots__: Incomplete
    parts: Incomplete
    decrypted: Incomplete
    _abbr: Incomplete
    def __init__(self, input) -> None:
        """
        Initialize a Type-1 font.

        Parameters
        ----------
        input : str or 3-tuple
            Either a pfb file name, or a 3-tuple of already-decoded Type-1
            font `~.Type1Font.parts`.
        """
    def _read(self, file):
        """Read the font from a file, decoding into usable parts."""
    def _split(self, data):
        """
        Split the Type 1 font into its three main parts.

        The three parts are: (1) the cleartext part, which ends in a
        eexec operator; (2) the encrypted part; (3) the fixed part,
        which contains 512 ASCII zeros possibly divided on various
        lines, a cleartomark operator, and possibly something else.
        """
    @staticmethod
    def _decrypt(ciphertext, key, ndiscard: int = 4):
        '''
        Decrypt ciphertext using the Type-1 font algorithm.

        The algorithm is described in Adobe\'s "Adobe Type 1 Font Format".
        The key argument can be an integer, or one of the strings
        \'eexec\' and \'charstring\', which map to the key specified for the
        corresponding part of Type-1 fonts.

        The ndiscard argument should be an integer, usually 4.
        That number of bytes is discarded from the beginning of plaintext.
        '''
    @staticmethod
    def _encrypt(plaintext, key, ndiscard: int = 4):
        '''
        Encrypt plaintext using the Type-1 font algorithm.

        The algorithm is described in Adobe\'s "Adobe Type 1 Font Format".
        The key argument can be an integer, or one of the strings
        \'eexec\' and \'charstring\', which map to the key specified for the
        corresponding part of Type-1 fonts.

        The ndiscard argument should be an integer, usually 4. That
        number of bytes is prepended to the plaintext before encryption.
        This function prepends NUL bytes for reproducibility, even though
        the original algorithm uses random bytes, presumably to avoid
        cryptanalysis.
        '''
    prop: Incomplete
    _pos: Incomplete
    def _parse(self) -> None:
        '''
        Find the values of various font properties. This limited kind
        of parsing is described in Chapter 10 "Adobe Type Manager
        Compatibility" of the Type-1 spec.
        '''
    def _parse_subrs(self, tokens, _data): ...
    @staticmethod
    def _parse_charstrings(tokens, _data): ...
    @staticmethod
    def _parse_encoding(tokens, _data): ...
    @staticmethod
    def _parse_othersubrs(tokens, data): ...
    def transform(self, effects):
        """
        Return a new font that is slanted and/or extended.

        Parameters
        ----------
        effects : dict
            A dict with optional entries:

            - 'slant' : float, default: 0
                Tangent of the angle that the font is to be slanted to the
                right. Negative values slant to the left.
            - 'extend' : float, default: 1
                Scaling factor for the font width. Values less than 1 condense
                the glyphs.

        Returns
        -------
        `Type1Font`
        """

_StandardEncoding: Incomplete

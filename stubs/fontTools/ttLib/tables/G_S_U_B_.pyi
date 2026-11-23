from .otBase import BaseTTXConverter as BaseTTXConverter
from .otTables import GSUB

class table_G_S_U_B_(BaseTTXConverter):
    """Glyph Substitution table

    The ``GSUB`` table contains glyph-substitution rules used in
    OpenType Layout.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/gsub
    """
    table: GSUB

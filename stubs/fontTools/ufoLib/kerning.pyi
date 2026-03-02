from fontTools.annotations import (
	IntFloat as IntFloat, KerningDict as KerningDict, KerningGroups as KerningGroups, KerningPair as KerningPair)
from typing import TypeAlias

StrDict: TypeAlias = dict[str, str]

def lookupKerningValue(pair: KerningPair, kerning: KerningDict, groups: KerningGroups, fallback: IntFloat = 0, glyphToFirstGroup: StrDict | None = None, glyphToSecondGroup: StrDict | None = None) -> IntFloat:
    """Retrieve the kerning value (if any) between a pair of elements.

    The elments can be either individual glyphs (by name) or kerning
    groups (by name), or any combination of the two.

    Args:
      pair:
          A tuple, in logical order (first, second) with respect
          to the reading direction, to query the font for kerning
          information on. Each element in the tuple can be either
          a glyph name or a kerning group name.
      kerning:
          A dictionary of kerning pairs.
      groups:
          A set of kerning groups.
      fallback:
          The fallback value to return if no kern is found between
          the elements in ``pair``. Defaults to 0.
      glyphToFirstGroup:
          A dictionary mapping glyph names to the first-glyph kerning
          groups to which they belong. Defaults to ``None``.
      glyphToSecondGroup:
          A dictionary mapping glyph names to the second-glyph kerning
          groups to which they belong. Defaults to ``None``.

    Returns
    -------
      The kerning value between the element pair. If no kerning for
      the pair is found, the fallback value is returned.

    Note: This function expects the ``kerning`` argument to be a flat
    dictionary of kerning pairs, not the nested structure used in a
    kerning.plist file.

    Examples::

      >>> groups = {
      ...     "public.kern1.O" : ["O", "D", "Q"],
      ...     "public.kern2.E" : ["E", "F"]
      ... }
      >>> kerning = {
      ...     ("public.kern1.O", "public.kern2.E") : -100,
      ...     ("public.kern1.O", "F") : -200,
      ...     ("D", "F") : -300
      ... }
      >>> lookupKerningValue(("D", "F"), kerning, groups)
      -300
      >>> lookupKerningValue(("O", "F"), kerning, groups)
      -200
      >>> lookupKerningValue(("O", "E"), kerning, groups)
      -100
      >>> lookupKerningValue(("O", "O"), kerning, groups)
      0
      >>> lookupKerningValue(("E", "E"), kerning, groups)
      0
      >>> lookupKerningValue(("E", "O"), kerning, groups)
      0
      >>> lookupKerningValue(("X", "X"), kerning, groups)
      0
      >>> lookupKerningValue(("public.kern1.O", "public.kern2.E"),
      ...     kerning, groups)
      -100
      >>> lookupKerningValue(("public.kern1.O", "F"), kerning, groups)
      -200
      >>> lookupKerningValue(("O", "public.kern2.E"), kerning, groups)
      -100
      >>> lookupKerningValue(("public.kern1.X", "public.kern2.X"), kerning, groups)
      0
    """

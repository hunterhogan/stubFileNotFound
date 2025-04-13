from _typeshed import Incomplete
from typing import overload

latex_to_bakoma: Incomplete
type12uni: Incomplete
uni2type1: Incomplete
tex2uni: Incomplete
_EntryTypeIn = tuple[str, str, str, str | int]
_EntryTypeOut = tuple[int, int, str, int]
_stix_virtual_fonts: dict[str, dict[str, list[_EntryTypeIn]] | list[_EntryTypeIn]]

@overload
def _normalize_stix_fontcodes(d: _EntryTypeIn) -> _EntryTypeOut: ...
@overload
def _normalize_stix_fontcodes(d: list[_EntryTypeIn]) -> list[_EntryTypeOut]: ...
@overload
def _normalize_stix_fontcodes(d: dict[str, list[_EntryTypeIn] | dict[str, list[_EntryTypeIn]]]) -> dict[str, list[_EntryTypeOut] | dict[str, list[_EntryTypeOut]]]: ...

stix_virtual_fonts: dict[str, dict[str, list[_EntryTypeOut]] | list[_EntryTypeOut]]
stix_glyph_fixes: Incomplete


from .Axis import Axis, Tag
from .BaseObject import BaseObject, Number
from .Features import Features
from .Glyph import GlyphList
from .Instance import Instance
from .Master import Master
from .Names import Names
from dataclasses import dataclass
from datetime import datetime
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.varLib.models import VariationModel
from typing import Any, Dict, List, Optional, Tuple
import functools

log = ...
@dataclass
class _FontFields:
    upm: int = ...
    version: tuple[int, int] = ...
    axes: list[Axis] = ...
    instances: list[Instance] = ...
    masters: list[Master] = ...
    glyphs: GlyphList = ...
    note: str = ...
    date: datetime = ...
    names: Names = ...
    custom_opentype_values: dict[tuple[str, str], Any] = ...
    features: Features = ...
    first_kern_groups: dict[str, list[str]] = ...
    second_kern_groups: dict[str, list[str]] = ...


@dataclass
class Font(_FontFields, BaseObject):
    """Represents a font, with one or more masters."""

    def __repr__(self): # -> str:
        ...

    def save(self, filename: str, **kwargs):
        """Save the font to a file. The file type is determined by the extension.
        Any additional keyword arguments are passed to the save method of the
        appropriate converter.
        """

    def master(self, mid: str) -> Master | None:
        """Locates a master by its ID. Returns `None` if not found."""

    def map_forward(self, location: dict[Tag, Number]) -> dict[Tag, Number]:
        """Map a location (dictionary of `tag: number`) from userspace to designspace."""

    def map_backward(self, location: dict[Tag, Number]) -> dict[Tag, Number]:
        """Map a location (dictionary of `tag: number`) from designspace to userspace."""

    def userspace_to_designspace(self, v: dict[Tag, Number]) -> dict[Tag, Number]:
        """Map a location (dictionary of `tag: number`) from userspace to designspace."""

    def designspace_to_userspace(self, v: dict[Tag, Number]) -> dict[Tag, Number]:
        """Map a location (dictionary of `tag: number`) from designspace to userspace."""

    @functools.cached_property
    def default_master(self) -> Master:
        """Return the default master. If there is only one master, return it.
        If there are multiple masters, return the one with the default location.
        If there is no default location, raise an error.
        """

    @functools.cached_property
    def unicode_map(self) -> dict[int, str]:
        """Return a dictionary mapping Unicode codepoints to glyph names."""

    def variation_model(self) -> VariationModel:
        """Return a `fontTools.varLib.models.VariationModel` object representing
        the font's axes and masters. This is used for generating variable fonts.
        """

    def get_variable_anchor(self, glyph, anchorname) -> tuple[VariableScalar, VariableScalar]:
        """Return a tuple of `VariableScalar` objects representing the x and y
        coordinates of the anchor on the given glyph. The `VariableScalar` objects
        are indexed by master location. If the anchor is not found on some master,
        raise an `IncompatibleMastersError`.
        """

    def exported_glyphs(self) -> list[str]:
        """Return a list of glyph names that are marked for export."""

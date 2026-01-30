from _typeshed import Incomplete
from dataclasses import dataclass
from fontTools.designspaceLib import AxisDescriptor as AxisDescriptor, AxisLabelDescriptor as AxisLabelDescriptor, DesignSpaceDocument as DesignSpaceDocument, DiscreteAxisDescriptor as DiscreteAxisDescriptor, SimpleLocationDict as SimpleLocationDict, SourceDescriptor as SourceDescriptor

LOGGER: Incomplete
RibbiStyleName: Incomplete
BOLD_ITALIC_TO_RIBBI_STYLE: Incomplete

@dataclass
class StatNames:
    """Name data generated from the STAT table information."""
    familyNames: dict[str, str]
    styleNames: dict[str, str]
    postScriptFontName: str | None
    styleMapFamilyNames: dict[str, str]
    styleMapStyleName: RibbiStyleName | None

def getStatNames(doc: DesignSpaceDocument, userLocation: SimpleLocationDict) -> StatNames:
    """Compute the family, style, PostScript names of the given ``userLocation``
    using the document's STAT information.

    Also computes localizations.

    If not enough STAT data is available for a given name, either its dict of
    localized names will be empty (family and style names), or the name will be
    None (PostScript name).

    Note: this method does not consider info attached to the instance, like
    family name. The user needs to override all names on an instance that STAT
    information would compute differently than desired.

    .. versionadded:: 5.0
    """
def _getSortedAxisLabels(axes: list[AxisDescriptor | DiscreteAxisDescriptor]) -> dict[str, list[AxisLabelDescriptor]]:
    """Returns axis labels sorted by their ordering, with unordered ones appended as
    they are listed."""
def _getAxisLabelsForUserLocation(axes: list[AxisDescriptor | DiscreteAxisDescriptor], userLocation: SimpleLocationDict) -> list[AxisLabelDescriptor]: ...
def _getRibbiStyle(self, userLocation: SimpleLocationDict) -> tuple[RibbiStyleName, SimpleLocationDict]:
    """Compute the RIBBI style name of the given user location,
    return the location of the matching Regular in the RIBBI group.

    .. versionadded:: 5.0
    """

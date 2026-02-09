from _typeshed import Incomplete
from collections.abc import Callable, Iterator
from fontTools.designspaceLib import (
	AxisDescriptor as AxisDescriptor, AxisMappingDescriptor as AxisMappingDescriptor,
	DesignSpaceDocument as DesignSpaceDocument, DiscreteAxisDescriptor as DiscreteAxisDescriptor,
	InstanceDescriptor as InstanceDescriptor, RuleDescriptor as RuleDescriptor, SimpleLocationDict as SimpleLocationDict,
	SourceDescriptor as SourceDescriptor, VariableFontDescriptor as VariableFontDescriptor)
from fontTools.designspaceLib.statNames import getStatNames as getStatNames, StatNames as StatNames
from fontTools.designspaceLib.types import (
	ConditionSet as ConditionSet, getVFUserRegion as getVFUserRegion, locationInRegion as locationInRegion, Range as Range,
	Region as Region, regionInRegion as regionInRegion, userRegionToDesignRegion as userRegionToDesignRegion)
from typing import Any, TypeAlias

LOGGER: Incomplete
MakeInstanceFilenameCallable: TypeAlias = Callable[[DesignSpaceDocument, InstanceDescriptor, StatNames], str]

def defaultMakeInstanceFilename(doc: DesignSpaceDocument, instance: InstanceDescriptor, statNames: StatNames) -> str:
    """Default callable to synthesize an instance filename
    when makeNames=True, for instances that don't specify an instance name
    in the designspace. This part of the name generation can be overriden
    because it's not specified by the STAT table.
    """
def splitInterpolable(doc: DesignSpaceDocument, makeNames: bool = True, expandLocations: bool = True, makeInstanceFilename: MakeInstanceFilenameCallable = ...) -> Iterator[tuple[SimpleLocationDict, DesignSpaceDocument]]:
    """Split the given DS5 into several interpolable sub-designspaces.
    There are as many interpolable sub-spaces as there are combinations of
    discrete axis values.

    E.g. with axes:
        - italic (discrete) Upright or Italic
        - style (discrete) Sans or Serif
        - weight (continuous) 100 to 900

    There are 4 sub-spaces in which the Weight axis should interpolate:
    (Upright, Sans), (Upright, Serif), (Italic, Sans) and (Italic, Serif).

    The sub-designspaces still include the full axis definitions and STAT data,
    but the rules, sources, variable fonts, instances are trimmed down to only
    keep what falls within the interpolable sub-space.

    Args:
      - ``makeNames``: Whether to compute the instance family and style
        names using the STAT data.
      - ``expandLocations``: Whether to turn all locations into "full"
        locations, including implicit default axis values where missing.
      - ``makeInstanceFilename``: Callable to synthesize an instance filename
        when makeNames=True, for instances that don\'t specify an instance name
        in the designspace. This part of the name generation can be overridden
        because it\'s not specified by the STAT table.

    .. versionadded:: 5.0
    """
def splitVariableFonts(doc: DesignSpaceDocument, makeNames: bool = False, expandLocations: bool = False, makeInstanceFilename: MakeInstanceFilenameCallable = ...) -> Iterator[tuple[str, DesignSpaceDocument]]:
    """Convert each variable font listed in this document into a standalone
    designspace. This can be used to compile all the variable fonts from a
    format 5 designspace using tools that can only deal with 1 VF at a time.

    Args:
      - ``makeNames``: Whether to compute the instance family and style
        names using the STAT data.
      - ``expandLocations``: Whether to turn all locations into "full"
        locations, including implicit default axis values where missing.
      - ``makeInstanceFilename``: Callable to synthesize an instance filename
        when makeNames=True, for instances that don\'t specify an instance name
        in the designspace. This part of the name generation can be overridden
        because it\'s not specified by the STAT table.

    .. versionadded:: 5.0
    """
def convert5to4(doc: DesignSpaceDocument) -> dict[str, DesignSpaceDocument]:
    """Convert each variable font listed in this document into a standalone
    format 4 designspace. This can be used to compile all the variable fonts
    from a format 5 designspace using tools that only know about format 4.

    .. versionadded:: 5.0
    """
def _extractSubSpace(doc: DesignSpaceDocument, userRegion: Region, *, keepVFs: bool, makeNames: bool, expandLocations: bool, makeInstanceFilename: MakeInstanceFilenameCallable) -> DesignSpaceDocument: ...
def _conditionSetFrom(conditionSet: list[dict[str, Any]]) -> ConditionSet: ...
def _subsetRulesBasedOnConditions(rules: list[RuleDescriptor], designRegion: Region) -> list[RuleDescriptor]: ...
def _filterLocation(userRegion: Region, location: dict[str, float]) -> dict[str, float]: ...

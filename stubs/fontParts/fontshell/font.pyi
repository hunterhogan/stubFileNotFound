
from fontParts.base import BaseFont
from fontParts.fontshell.base import RBaseObject
from fontParts.fontshell.features import RFeatures
from fontParts.fontshell.groups import RGroups
from fontParts.fontshell.guideline import RGuideline
from fontParts.fontshell.info import RInfo
from fontParts.fontshell.kerning import RKerning
from fontParts.fontshell.layer import RLayer
from fontParts.fontshell.lib import RLib

class RFont(RBaseObject, BaseFont):
    wrapClass = ...
    infoClass = RInfo
    groupsClass = RGroups
    kerningClass = RKerning
    featuresClass = RFeatures
    libClass = RLib
    layerClass = RLayer
    guidelineClass = RGuideline

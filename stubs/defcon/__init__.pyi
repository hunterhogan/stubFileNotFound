
from ._version import __version__
from defcon.errors import DefconError
from defcon.objects.anchor import Anchor
from defcon.objects.color import Color
from defcon.objects.component import Component
from defcon.objects.contour import Contour
from defcon.objects.features import Features
from defcon.objects.font import Font
from defcon.objects.glyph import addRepresentationFactory, Glyph, removeRepresentationFactory
from defcon.objects.groups import Groups
from defcon.objects.guideline import Guideline
from defcon.objects.image import Image
from defcon.objects.info import Info
from defcon.objects.kerning import Kerning
from defcon.objects.layer import Layer
from defcon.objects.layerSet import LayerSet
from defcon.objects.layoutEngine import LayoutEngine
from defcon.objects.lib import Lib
from defcon.objects.point import Point
from defcon.objects.uniData import UnicodeData

"""
A set of objects that are suited to being the basis
of font development tools. This works on UFO files.
"""
def registerRepresentationFactory(cls, name, factory, destructiveNotifications=...): # -> None:
    """
    Register **factory** as a representation factory
    for all instances of **cls** (a :class:`defcon.objects.base.BaseObject`)
    subclass under **name**.
    """

def unregisterRepresentationFactory(cls, name): # -> None:
    """
    Unregister the representation factory stored under
    **name** in all instances of **cls**.
    """

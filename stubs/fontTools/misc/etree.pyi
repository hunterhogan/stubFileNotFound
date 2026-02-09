from _typeshed import Incomplete
from lxml.etree import *
from xml.etree.ElementTree import *
from xml.etree.ElementTree import Element as _Element, XML as XML
import contextlib

__all__ = ['PI', 'XML', 'Comment', 'Element', 'ElementTree', 'ParseError', 'ProcessingInstruction', 'QName', 'SubElement', 'TreeBuilder', 'XMLParser', 'dump', 'fromstring', 'fromstringlist', 'iselement', 'iterparse', 'parse', 'register_namespace', 'tostring', 'tostringlist']

_Attrib = dict
_Element = Element

class Element(_Element):
    """Element subclass that keeps the order of attributes."""

    attrib: Incomplete
    def __init__(self, tag, attrib=..., **extra) -> None: ...

def SubElement(parent, tag, attrib=..., **extra):
    """Must override SubElement as well otherwise _elementtree.SubElement
    fails if 'parent' is a subclass of Element object.
    """
_ElementTree = ElementTree

class ElementTree(_ElementTree):
    """ElementTree subclass that adds \'pretty_print\' and \'doctype\'
    arguments to the \'write\' method.
    Currently these are only supported for the default XML serialization
    \'method\', and not also for "html" or "text", for these are delegated
    to the base class.
    """

    def write(self, file_or_filename, encoding=None, xml_declaration: bool = False, method=None, doctype=None, pretty_print: bool = False) -> None: ...

def tostring(element, encoding=None, xml_declaration=None, method=None, doctype=None, pretty_print: bool = False):
    """Custom 'tostring' function that uses our ElementTree subclass, with
    pretty_print support.
    """

# Names in __all__ with no definition:
#   Comment
#   PI
#   ParseError
#   ProcessingInstruction
#   QName
#   TreeBuilder
#   XMLParser
#   dump
#   fromstring
#   fromstringlist
#   iselement
#   iterparse
#   parse
#   register_namespace
#   tostringlist

from lxml.etree import *
from xml.etree.cElementTree import *
from xml.etree.ElementTree import *
import contextlib
from _typeshed import Incomplete
from xml.etree.ElementTree import Element as _Element, XML as XML

__all__ = ['Comment', 'dump', 'Element', 'ElementTree', 'fromstring', 'fromstringlist', 'iselement', 'iterparse', 'parse', 'ParseError', 'PI', 'ProcessingInstruction', 'QName', 'SubElement', 'tostring', 'tostringlist', 'TreeBuilder', 'XML', 'XMLParser', 'register_namespace']

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
    '''ElementTree subclass that adds \'pretty_print\' and \'doctype\'
        arguments to the \'write\' method.
        Currently these are only supported for the default XML serialization
        \'method\', and not also for "html" or "text", for these are delegated
        to the base class.
        '''
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

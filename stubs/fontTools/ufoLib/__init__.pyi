from _typeshed import Incomplete
from collections import OrderedDict
from fontTools.annotations import (
	GlyphNameToFileNameFunc, KerningDict, KerningGroups, KerningNested, PathOrFS, PathStr, UFOFormatVersionInput)
from fontTools.misc.filesystem._base import FS
from fontTools.ufoLib.errors import UFOLibError as UFOLibError
from fontTools.ufoLib.glifLib import GlyphSet
from fontTools.ufoLib.utils import BaseFormatVersion
from fontTools.ufoLib.validators import *
from typing import Any, IO, TypeAlias
import enum

__all__ = ['UFOFileStructure', 'UFOLibError', 'UFOReader', 'UFOReaderWriter', 'UFOWriter', ..., ..., 'deprecatedFontInfoAttributesVersion2', 'fontInfoAttributesVersion1', 'fontInfoAttributesVersion2', 'fontInfoAttributesVersion3', 'haveFS', 'makeUFOPath', 'validateFontInfoVersion2ValueForAttribute', 'validateFontInfoVersion3ValueForAttribute']

KerningGroupRenameMaps: TypeAlias = dict[str, dict[str, str]]
LibDict: TypeAlias = dict[str, Any]
LayerOrderList: TypeAlias = list[str | None] | None
AttributeDataDict: TypeAlias = dict[str, Any]
FontInfoAttributes: TypeAlias = dict[str, AttributeDataDict]
haveFS: Incomplete

class UFOFormatVersion(BaseFormatVersion):
    FORMAT_1_0 = (1, 0)
    FORMAT_2_0 = (2, 0)
    FORMAT_3_0 = (3, 0)

class UFOFileStructure(enum.Enum):
    ZIP = 'zip'
    PACKAGE = 'package'

class _UFOBaseIO:
    fs: FS
    _havePreviousFile: bool
    def getFileModificationTime(self, path: PathStr) -> float | None:
        """
        Returns the modification time for the file at the given path, as a
        floating point number giving the number of seconds since the epoch.
        The path must be relative to the UFO path.
        Returns None if the file does not exist.
        """
    def _getPlist(self, fileName: str, default: Any | None = None) -> Any:
        """
        Read a property list relative to the UFO filesystem's root.
        Raises UFOLibError if the file is missing and default is None,
        otherwise default is returned.

        The errors that could be raised during the reading of a plist are
        unpredictable and/or too large to list, so, a blind try: except:
        is done. If an exception occurs, a UFOLibError will be raised.
        """
    def _writePlist(self, fileName: str, obj: Any) -> None:
        """
        Write a property list to a file relative to the UFO filesystem's root.

        Do this sort of atomically, making it harder to corrupt existing files,
        for example when plistlib encounters an error halfway during write.
        This also checks to see if text matches the text that is already in the
        file at path. If so, the file is not rewritten so that the modification
        date is preserved.

        The errors that could be raised during the writing of a plist are
        unpredictable and/or too large to list, so, a blind try: except: is done.
        If an exception occurs, a UFOLibError will be raised.
        """

class UFOReader(_UFOBaseIO):
    """Read the various components of a .ufo.

    Attributes
    ----------
        path: An :class:`os.PathLike` object pointing to the .ufo.
        validate: A boolean indicating if the data read should be
          validated. Defaults to `True`.

    By default read data is validated. Set ``validate`` to
    ``False`` to not validate the data.
    """

    fs: FS
    _shouldClose: bool
    _fileStructure: Incomplete
    _path: str
    _validate: bool
    _upConvertedKerningData: dict[str, Any] | None
    def __init__(self, path: PathOrFS, validate: bool = True) -> None: ...
    def _get_path(self) -> str: ...
    path: property
    def _get_formatVersion(self) -> int: ...
    formatVersion: Incomplete
    @property
    def formatVersionTuple(self) -> tuple[int, int]:
        """The (major, minor) format version of the UFO.
        This is determined by reading metainfo.plist during __init__.
        """
    def _get_fileStructure(self) -> Any: ...
    fileStructure: property
    def _upConvertKerning(self, validate: bool) -> None:
        """
        Up convert kerning and groups in UFO 1 and 2.
        The data will be held internally until each bit of data
        has been retrieved. The conversion of both must be done
        at once, so the raw data is cached and an error is raised
        if one bit of data becomes obsolete before it is called.

        ``validate`` will validate the data.
        """
    def readBytesFromPath(self, path: PathStr) -> bytes | None:
        """
        Returns the bytes in the file at the given path.
        The path must be relative to the UFO's filesystem root.
        Returns None if the file does not exist.
        """
    def getReadFileForPath(self, path: PathStr, encoding: str | None = None) -> IO[bytes] | IO[str] | None:
        """
        Returns a file (or file-like) object for the file at the given path.
        The path must be relative to the UFO path.
        Returns None if the file does not exist.
        By default the file is opened in binary mode (reads bytes).
        If encoding is passed, the file is opened in text mode (reads str).

        Note: The caller is responsible for closing the open file.
        """
    def _readMetaInfo(self, validate: bool | None = None) -> dict[str, Any]:
        """
        Read metainfo.plist and return raw data. Only used for internal operations.

        ``validate`` will validate the read data, by default it is set
        to the class's validate value, can be overridden.
        """
    _formatVersion: Incomplete
    def readMetaInfo(self, validate: bool | None = None) -> None:
        """
        Read metainfo.plist and set formatVersion. Only used for internal operations.

        ``validate`` will validate the read data, by default it is set
        to the class's validate value, can be overridden.
        """
    def _readGroups(self) -> dict[str, list[str]]: ...
    def readGroups(self, validate: bool | None = None) -> dict[str, list[str]]:
        """
        Read groups.plist. Returns a dict.
        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
    def getKerningGroupConversionRenameMaps(self, validate: bool | None = None) -> KerningGroupRenameMaps:
        """
        Get maps defining the renaming that was done during any
        needed kerning group conversion. This method returns a
        dictionary of this form::

                {
                        "side1" : {"old group name" : "new group name"},
                        "side2" : {"old group name" : "new group name"}
                }

        When no conversion has been performed, the side1 and side2
        dictionaries will be empty.

        ``validate`` will validate the groups, by default it is set to the
        class\'s validate value, can be overridden.
        """
    def _readInfo(self, validate: bool) -> dict[str, Any]: ...
    def readInfo(self, info: Any, validate: bool | None = None) -> None:
        """
        Read fontinfo.plist. It requires an object that allows
        setting attributes with names that follow the fontinfo.plist
        version 3 specification. This will write the attributes
        defined in the file into the object.

        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
    def _readKerning(self) -> KerningNested: ...
    def readKerning(self, validate: bool | None = None) -> KerningDict:
        """
        Read kerning.plist. Returns a dict.

        ``validate`` will validate the kerning data, by default it is set to the
        class's validate value, can be overridden.
        """
    def readLib(self, validate: bool | None = None) -> dict[str, Any]:
        """
        Read lib.plist. Returns a dict.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def readFeatures(self) -> str:
        """
        Read features.fea. Return a string.
        The returned string is empty if the file is missing.
        """
    def _readLayerContents(self, validate: bool) -> list[tuple[str, str]]:
        """
        Rebuild the layer contents list by checking what glyphsets
        are available on disk.

        ``validate`` will validate the layer contents.
        """
    def getLayerNames(self, validate: bool | None = None) -> list[str]:
        """
        Get the ordered layer names from layercontents.plist.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def getDefaultLayerName(self, validate: bool | None = None) -> str:
        """
        Get the default layer name from layercontents.plist.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def getGlyphSet(self, layerName: str | None = None, validateRead: bool | None = None, validateWrite: bool | None = None) -> GlyphSet:
        """
        Return the GlyphSet associated with the
        glyphs directory mapped to layerName
        in the UFO. If layerName is not provided,
        the name retrieved with getDefaultLayerName
        will be used.

        ``validateRead`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        ``validateWrite`` will validate the written data, by default it is set to the
        class's validate value, can be overridden.
        """
    def getCharacterMapping(self, layerName: str | None = None, validate: bool | None = None) -> dict[int, list[str]]:
        """
        Return a dictionary that maps unicode values (ints) to
        lists of glyph names.
        """
    _dataFS: Incomplete
    def getDataDirectoryListing(self) -> list[str]:
        """
        Returns a list of all files in the data directory.
        The returned paths will be relative to the UFO.
        This will not list directory names, only file names.
        Thus, empty directories will be skipped.
        """
    _imagesFS: Incomplete
    def getImageDirectoryListing(self, validate: bool | None = None) -> list[str]:
        """
        Returns a list of all image file names in
        the images directory. Each of the images will
        have been verified to have the PNG signature.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def readData(self, fileName: PathStr) -> bytes:
        """
        Return bytes for the file named 'fileName' inside the 'data/' directory.
        """
    def readImage(self, fileName: PathStr, validate: bool | None = None) -> bytes:
        """
        Return image data for the file named fileName.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def close(self) -> None: ...
    def __enter__(self) -> UFOReader: ...
    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None: ...

class UFOWriter(UFOReader):
    """Write the various components of a .ufo.

    Attributes
    ----------
        path: An :class:`os.PathLike` object pointing to the .ufo.
        formatVersion: the UFO format version as a tuple of integers (major, minor),
            or as a single integer for the major digit only (minor is implied to be 0).
            By default, the latest formatVersion will be used; currently it is 3.0,
            which is equivalent to formatVersion=(3, 0).
        fileCreator: The creator of the .ufo file. Defaults to
            `com.github.fonttools.ufoLib`.
        structure: The internal structure of the .ufo file: either `ZIP` or `PACKAGE`.
        validate: A boolean indicating if the data read should be validated. Defaults
            to `True`.

    By default, the written data will be validated before writing. Set ``validate`` to
    ``False`` if you do not want to validate the data. Validation can also be overriden
    on a per-method level if desired.

    Raises
    ------
        UnsupportedUFOFormat: An exception indicating that the requested UFO
            formatVersion is not supported.
    """

    fs: Incomplete
    _fileStructure: Incomplete
    _havePreviousFile: Incomplete
    _shouldClose: bool
    _path: Incomplete
    _formatVersion: Incomplete
    _fileCreator: Incomplete
    _downConversionKerningData: KerningGroupRenameMaps | None
    _validate: Incomplete
    layerContents: dict[str, str] | OrderedDict[str, str]
    def __init__(self, path: PathOrFS, formatVersion: UFOFormatVersionInput = None, fileCreator: str = 'com.github.fonttools.ufoLib', structure: UFOFileStructure | None = None, validate: bool = True) -> None: ...
    def _get_fileCreator(self) -> str: ...
    fileCreator: property
    def copyFromReader(self, reader: UFOReader, sourcePath: PathStr, destPath: PathStr) -> None:
        """
        Copy the sourcePath in the provided UFOReader to destPath
        in this writer. The paths must be relative. This works with
        both individual files and directories.
        """
    def writeBytesToPath(self, path: PathStr, data: bytes) -> None:
        """
        Write bytes to a path relative to the UFO filesystem's root.
        If writing to an existing UFO, check to see if data matches the data
        that is already in the file at path; if so, the file is not rewritten
        so that the modification date is preserved.
        If needed, the directory tree for the given path will be built.
        """
    def getFileObjectForPath(self, path: PathStr, mode: str = 'w', encoding: str | None = None) -> IO[Any] | None:
        """
        Returns a file (or file-like) object for the
        file at the given path. The path must be relative
        to the UFO path. Returns None if the file does
        not exist and the mode is "r" or "rb.
        An encoding may be passed if the file is opened in text mode.

        Note: The caller is responsible for closing the open file.
        """
    def removePath(self, path: PathStr, force: bool = False, removeEmptyParents: bool = True) -> None:
        """
        Remove the file (or directory) at path. The path
        must be relative to the UFO.
        Raises UFOLibError if the path doesn't exist.
        If force=True, ignore non-existent paths.
        If the directory where 'path' is located becomes empty, it will
        be automatically removed, unless 'removeEmptyParents' is False.
        """
    removeFileForPath = removePath
    def setModificationTime(self) -> None:
        """
        Set the UFO modification time to the current time.
        This is never called automatically. It is up to the
        caller to call this when finished working on the UFO.
        """
    def _writeMetaInfo(self) -> None: ...
    def setKerningGroupConversionRenameMaps(self, maps: KerningGroupRenameMaps) -> None:
        """
        Set maps defining the renaming that should be done
        when writing groups and kerning in UFO 1 and UFO 2.
        This will effectively undo the conversion done when
        UFOReader reads this data. The dictionary should have
        this form::

                {
                        "side1" : {"group name to use when writing" : "group name in data"},
                        "side2" : {"group name to use when writing" : "group name in data"}
                }

        This is the same form returned by UFOReader\'s
        getKerningGroupConversionRenameMaps method.
        """
    def writeGroups(self, groups: KerningGroups, validate: bool | None = None) -> None:
        """
        Write groups.plist. This method requires a
        dict of glyph groups as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def writeInfo(self, info: Any, validate: bool | None = None) -> None:
        """
        Write info.plist. This method requires an object
        that supports getting attributes that follow the
        fontinfo.plist version 2 specification. Attributes
        will be taken from the given object and written
        into the file.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def writeKerning(self, kerning: KerningDict, validate: bool | None = None) -> None:
        """
        Write kerning.plist. This method requires a
        dict of kerning pairs as an argument.

        This performs basic structural validation of the kerning,
        but it does not check for compliance with the spec in
        regards to conflicting pairs. The assumption is that the
        kerning data being passed is standards compliant.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def writeLib(self, libDict: LibDict, validate: bool | None = None) -> None:
        """
        Write lib.plist. This method requires a
        lib dict as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    def writeFeatures(self, features: str, validate: bool | None = None) -> None:
        """
        Write features.fea. This method requires a
        features string as an argument.
        """
    def writeLayerContents(self, layerOrder: LayerOrderList = None, validate: bool | None = None) -> None:
        """
        Write the layercontents.plist file. This method  *must* be called
        after all glyph sets have been written.
        """
    def _findDirectoryForLayerName(self, layerName: str | None) -> str: ...
    def getGlyphSet(self, layerName: str | None = None, defaultLayer: bool = True, glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None, validateRead: bool | None = None, validateWrite: bool | None = None, expectContentsFile: bool = False) -> GlyphSet:
        """
        Return the GlyphSet object associated with the
        appropriate glyph directory in the .ufo.
        If layerName is None, the default glyph set
        will be used. The defaultLayer flag indictes
        that the layer should be saved into the default
        glyphs directory.

        ``validateRead`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        ``validateWrte`` will validate the written data, by default it is set to the
        class's validate value, can be overridden.
        ``expectContentsFile`` will raise a GlifLibError if a contents.plist file is
        not found on the glyph set file system. This should be set to ``True`` if you
        are reading an existing UFO and ``False`` if you use ``getGlyphSet`` to create
        a fresh\tglyph set.
        """
    def _getDefaultGlyphSet(self, validateRead: bool, validateWrite: bool, glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None, expectContentsFile: bool = False) -> GlyphSet: ...
    def _getGlyphSetFormatVersion3(self, validateRead: bool, validateWrite: bool, layerName: str | None = None, defaultLayer: bool = True, glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None, expectContentsFile: bool = False) -> GlyphSet: ...
    def renameGlyphSet(self, layerName: str | None, newLayerName: str | None, defaultLayer: bool = False) -> None:
        """
        Rename a glyph set.

        Note: if a GlyphSet object has already been retrieved for
        layerName, it is up to the caller to inform that object that
        the directory it represents has changed.
        """
    def deleteGlyphSet(self, layerName: str | None) -> None:
        """
        Remove the glyph set matching layerName.
        """
    def writeData(self, fileName: PathStr, data: bytes) -> None:
        """
        Write data to fileName in the 'data' directory.
        The data must be a bytes string.
        """
    def removeData(self, fileName: PathStr) -> None:
        """
        Remove the file named fileName from the data directory.
        """
    def writeImage(self, fileName: PathStr, data: bytes, validate: bool | None = None) -> None:
        """
        Write data to fileName in the images directory.
        The data must be a valid PNG.
        """
    def removeImage(self, fileName: PathStr, validate: bool | None = None) -> None:
        """
        Remove the file named fileName from the
        images directory.
        """
    def copyImageFromReader(self, reader: UFOReader, sourceFileName: PathStr, destFileName: PathStr, validate: bool | None = None) -> None:
        """
        Copy the sourceFileName in the provided UFOReader to destFileName
        in this writer. This uses the most memory efficient method possible
        for copying the data possible.
        """
    def close(self) -> None: ...
UFOReaderWriter = UFOWriter

def makeUFOPath(path: PathStr) -> str:
    """
    Return a .ufo pathname.

    >>> makeUFOPath("directory/something.ext") == (
    ... \tos.path.join(\'directory\', \'something.ufo\'))
    True
    >>> makeUFOPath("directory/something.another.thing.ext") == (
    ... \tos.path.join(\'directory\', \'something.another.thing.ufo\'))
    True
    """
def validateFontInfoVersion2ValueForAttribute(attr: str, value: Any) -> bool:
    """
    This performs very basic validation of the value for attribute
    following the UFO 2 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
def validateFontInfoVersion3ValueForAttribute(attr: str, value: Any) -> bool:
    """
    This performs very basic validation of the value for attribute
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """

fontInfoAttributesVersion1: set[str]
fontInfoAttributesVersion2: set[str]
fontInfoAttributesVersion3: set[str]
deprecatedFontInfoAttributesVersion2: Incomplete

def convertFontInfoValueForAttributeFromVersion1ToVersion2(attr: str, value: Any) -> tuple[str, Any]:
    """
    Convert value from version 1 to version 2 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
def convertFontInfoValueForAttributeFromVersion2ToVersion1(attr: str, value: Any) -> tuple[str, Any]:
    """
    Convert value from version 2 to version 1 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """

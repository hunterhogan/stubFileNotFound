import os
from _typeshed import Incomplete
from collections.abc import Mapping, MutableMapping
from fontTools.config import Config as Config
from fontTools.misc import xmlWriter as xmlWriter
from fontTools.misc.configTools import AbstractConfig as AbstractConfig
from fontTools.misc.loggingTools import deprecateArgument as deprecateArgument
from fontTools.misc.textTools import Tag as Tag, byteord as byteord, tostr as tostr
from fontTools.ttLib import TTLibError as TTLibError
from fontTools.ttLib.sfnt import SFNTReader as SFNTReader, SFNTWriter as SFNTWriter
from fontTools.ttLib.tables import B_A_S_E_ as B_A_S_E_, C_B_D_T_ as C_B_D_T_, C_B_L_C_ as C_B_L_C_, C_F_F_ as C_F_F_, C_F_F__2 as C_F_F__2, C_O_L_R_ as C_O_L_R_, C_P_A_L_ as C_P_A_L_, D_S_I_G_ as D_S_I_G_, D__e_b_g as D__e_b_g, E_B_D_T_ as E_B_D_T_, E_B_L_C_ as E_B_L_C_, F_F_T_M_ as F_F_T_M_, F__e_a_t as F__e_a_t, G_D_E_F_ as G_D_E_F_, G_M_A_P_ as G_M_A_P_, G_P_K_G_ as G_P_K_G_, G_P_O_S_ as G_P_O_S_, G_S_U_B_ as G_S_U_B_, G_V_A_R_ as G_V_A_R_, G__l_a_t as G__l_a_t, G__l_o_c as G__l_o_c, H_V_A_R_ as H_V_A_R_, J_S_T_F_ as J_S_T_F_, L_T_S_H_ as L_T_S_H_, M_A_T_H_ as M_A_T_H_, M_E_T_A_ as M_E_T_A_, M_V_A_R_ as M_V_A_R_, O_S_2f_2 as O_S_2f_2, S_I_N_G_ as S_I_N_G_, S_T_A_T_ as S_T_A_T_, S_V_G_ as S_V_G_, S__i_l_f as S__i_l_f, S__i_l_l as S__i_l_l, T_S_I_B_ as T_S_I_B_, T_S_I_C_ as T_S_I_C_, T_S_I_D_ as T_S_I_D_, T_S_I_J_ as T_S_I_J_, T_S_I_P_ as T_S_I_P_, T_S_I_S_ as T_S_I_S_, T_S_I_V_ as T_S_I_V_, T_S_I__0 as T_S_I__0, T_S_I__1 as T_S_I__1, T_S_I__2 as T_S_I__2, T_S_I__3 as T_S_I__3, T_S_I__5 as T_S_I__5, T_T_F_A_ as T_T_F_A_, V_A_R_C_ as V_A_R_C_, V_D_M_X_ as V_D_M_X_, V_O_R_G_ as V_O_R_G_, V_V_A_R_ as V_V_A_R_, _a_n_k_r as _a_n_k_r, _a_v_a_r as _a_v_a_r, _b_s_l_n as _b_s_l_n, _c_i_d_g as _c_i_d_g, _c_m_a_p as _c_m_a_p, _c_v_a_r as _c_v_a_r, _c_v_t as _c_v_t, _f_e_a_t as _f_e_a_t, _f_p_g_m as _f_p_g_m, _f_v_a_r as _f_v_a_r, _g_a_s_p as _g_a_s_p, _g_c_i_d as _g_c_i_d, _g_l_y_f as _g_l_y_f, _g_v_a_r as _g_v_a_r, _h_d_m_x as _h_d_m_x, _h_e_a_d as _h_e_a_d, _h_h_e_a as _h_h_e_a, _h_m_t_x as _h_m_t_x, _k_e_r_n as _k_e_r_n, _l_c_a_r as _l_c_a_r, _l_o_c_a as _l_o_c_a, _l_t_a_g as _l_t_a_g, _m_a_x_p as _m_a_x_p, _m_e_t_a as _m_e_t_a, _m_o_r_t as _m_o_r_t, _m_o_r_x as _m_o_r_x, _n_a_m_e as _n_a_m_e, _o_p_b_d as _o_p_b_d, _p_o_s_t as _p_o_s_t, _p_r_e_p as _p_r_e_p, _p_r_o_p as _p_r_o_p, _s_b_i_x as _s_b_i_x, _t_r_a_k as _t_r_a_k, _v_h_e_a as _v_h_e_a, _v_m_t_x as _v_m_t_x
from fontTools.ttLib.tables.DefaultTable import DefaultTable as DefaultTable
from fontTools.ttLib.ttGlyphSet import _TTGlyph as _TTGlyph, _TTGlyphSet as _TTGlyphSet, _TTGlyphSetCFF as _TTGlyphSetCFF, _TTGlyphSetGlyf as _TTGlyphSetGlyf, _TTGlyphSetVARC as _TTGlyphSetVARC
from types import ModuleType, TracebackType
from typing import Any, BinaryIO, Literal, TextIO, TypeVar, TypedDict, overload
from collections.abc import Sequence
from typing import Self, Unpack

_VT_co = TypeVar('_VT_co', covariant=True)
log: Incomplete
_NumberT = TypeVar('_NumberT', bound=float)

class TTFont:
    r"""Represents a TrueType font.

    The object manages file input and output, and offers a convenient way of
    accessing tables. Tables will be only decompiled when necessary, ie. when
    they\'re actually accessed. This means that simple operations can be extremely fast.

    Example usage:

    .. code-block:: pycon

        >>>
        >> from fontTools import ttLib
        >> tt = ttLib.TTFont("afont.ttf") # Load an existing font file
        >> tt[\'maxp\'].numGlyphs
        242
        >> tt[\'OS/2\'].achVendID
        \'B&H\x00\'
        >> tt[\'head\'].unitsPerEm
        2048

    For details of the objects returned when accessing each table, see the
    :doc:`tables </ttLib/tables>` documentation.
    To add a table to the font, use the :py:func:`newTable` function:

    .. code-block:: pycon

        >>>
        >> os2 = newTable("OS/2")
        >> os2.version = 4
        >> # set other attributes
        >> font["OS/2"] = os2

    TrueType fonts can also be serialized to and from XML format (see also the
    :doc:`ttx </ttx>` binary):

    .. code-block:: pycon

        >>
        >> tt.saveXML("afont.ttx")
        Dumping \'LTSH\' table...
        Dumping \'OS/2\' table...
        [...]

        >> tt2 = ttLib.TTFont() # Create a new font object
        >> tt2.importXML("afont.ttx")
        >> tt2[\'maxp\'].numGlyphs
        242

    The TTFont object may be used as a context manager; this will cause the file
    reader to be closed after the context ``with`` block is exited::

            with TTFont(filename) as f:
                    # Do stuff

    Args:
            file: When reading a font from disk, either a pathname pointing to a file,
                    or a readable file object.
            res_name_or_index: If running on a Macintosh, either a sfnt resource name or
                    an sfnt resource index number. If the index number is zero, TTLib will
                    autodetect whether the file is a flat file or a suitcase. (If it is a suitcase,
                    only the first \'sfnt\' resource will be read.)
            sfntVersion (str): When constructing a font object from scratch, sets the four-byte
                    sfnt magic number to be used. Defaults to ``\x00\x01\x00\x00`` (TrueType). To create
                    an OpenType file, use ``OTTO``.
            flavor (str): Set this to ``woff`` when creating a WOFF file or ``woff2`` for a WOFF2
                    file.
            checkChecksums (int): How checksum data should be treated. Default is 0
                    (no checking). Set to 1 to check and warn on wrong checksums; set to 2 to
                    raise an exception if any wrong checksums are found.
            recalcBBoxes (bool): If true (the default), recalculates ``glyf``, ``CFF ``,
                    ``head`` bounding box values and ``hhea``/``vhea`` min/max values on save.
                    Also compiles the glyphs on importing, which saves memory consumption and
                    time.
            ignoreDecompileErrors (bool): If true, exceptions raised during table decompilation
                    will be ignored, and the binary data will be returned for those tables instead.
            recalcTimestamp (bool): If true (the default), sets the ``modified`` timestamp in
                    the ``head`` table on save.
            fontNumber (int): The index of the font in a TrueType Collection file.
            lazy (bool): If lazy is set to True, many data structures are loaded lazily, upon
                    access only. If it is set to False, many data structures are loaded immediately.
                    The default is ``lazy=None`` which is somewhere in between.
    """

    tables: dict[Tag, DefaultTable | GlyphOrder]
    reader: SFNTReader | None
    sfntVersion: str
    flavor: str | None
    flavorData: Any | None
    lazy: bool | None
    recalcBBoxes: bool
    recalcTimestamp: bool
    ignoreDecompileErrors: bool
    cfg: AbstractConfig
    glyphOrder: list[str]
    _reverseGlyphOrderDict: dict[str, int]
    _tableCache: MutableMapping[tuple[Tag, bytes], DefaultTable] | None
    disassembleInstructions: bool
    bitmapGlyphDataFormat: str
    verbose: bool | None
    quiet: bool | None
    def __init__(self, file: str | os.PathLike[str] | BinaryIO | None = None, res_name_or_index: str | int | None = None, sfntVersion: str = '\x00\x01\x00\x00', flavor: str | None = None, checkChecksums: int = 0, verbose: bool | None = None, recalcBBoxes: bool = True, allowVID: Any = ..., ignoreDecompileErrors: bool = False, recalcTimestamp: bool = True, fontNumber: int = -1, lazy: bool | None = None, quiet: bool | None = None, _tableCache: MutableMapping[tuple[Tag, bytes], DefaultTable] | None = None, cfg: Mapping[str, Any] | AbstractConfig = {}) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...
    def close(self) -> None:
        """If we still have a reader object, close it."""
    def save(self, file: str | os.PathLike[str] | BinaryIO, reorderTables: bool | None = True) -> None:
        """Save the font to disk.

        Args:
                file: Similarly to the constructor, can be either a pathname or a writable
                        binary file object.
                reorderTables (Option[bool]): If true (the default), reorder the tables,
                        sorting them by tag (recommended by the OpenType specification). If
                        false, retain the original font order. If None, reorder by table
                        dependency (fastest).
        """
    def _save(self, file: BinaryIO, tableCache: MutableMapping[tuple[Tag, bytes], Any] | None = None) -> bool:
        """Internal function, to be shared by save() and TTCollection.save()."""
    class XMLSavingOptions(TypedDict):
        writeVersion: bool
        quiet: bool | None
        tables: Sequence[str | bytes] | None
        skipTables: Sequence[str] | None
        splitTables: bool
        splitGlyphs: bool
        disassembleInstructions: bool
        bitmapGlyphDataFormat: str
    def saveXML(self, fileOrPath: str | os.PathLike[str] | BinaryIO | TextIO, newlinestr: str = '\n', **kwargs: Unpack[XMLSavingOptions]) -> None:
        """Export the font as TTX (an XML-based text file), or as a series of text
        files when splitTables is true. In the latter case, the 'fileOrPath'
        argument should be a path to a directory.
        The 'tables' argument must either be false (dump all tables) or a
        list of tables to dump. The 'skipTables' argument may be a list of tables
        to skip, but only when the 'tables' argument is false.
        """
    def _saveXML(self, writer: xmlWriter.XMLWriter, writeVersion: bool = True, quiet: bool | None = None, tables: Sequence[str | bytes] | None = None, skipTables: Sequence[str] | None = None, splitTables: bool = False, splitGlyphs: bool = False, disassembleInstructions: bool = True, bitmapGlyphDataFormat: str = 'raw') -> None: ...
    def _tableToXML(self, writer: xmlWriter.XMLWriter, tag: str | bytes, quiet: bool | None = None, splitGlyphs: bool = False) -> None: ...
    def importXML(self, fileOrPath: str | os.PathLike[str] | BinaryIO, quiet: bool | None = None) -> None:
        """Import a TTX file (an XML-based text format), so as to recreate
        a font object.
        """
    def isLoaded(self, tag: str | bytes) -> bool:
        """Return true if the table identified by ``tag`` has been
        decompiled and loaded into memory.
        """
    def has_key(self, tag: str | bytes) -> bool:
        """Test if the table identified by ``tag`` is present in the font.

        As well as this method, ``tag in font`` can also be used to determine the
        presence of the table.
        """
    __contains__ = has_key
    def keys(self) -> list[str]:
        """Returns the list of tables in the font, along with the ``GlyphOrder`` pseudo-table."""
    def ensureDecompiled(self, recurse: bool | None = None) -> None:
        """Decompile all the tables, even if a TTFont was opened in 'lazy' mode."""
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, tag: Literal['BASE']) -> B_A_S_E_.table_B_A_S_E_: ...
    @overload
    def __getitem__(self, tag: Literal['CBDT']) -> C_B_D_T_.table_C_B_D_T_: ...
    @overload
    def __getitem__(self, tag: Literal['CBLC']) -> C_B_L_C_.table_C_B_L_C_: ...
    @overload
    def __getitem__(self, tag: Literal['CFF']) -> C_F_F_.table_C_F_F_: ...
    @overload
    def __getitem__(self, tag: Literal['CFF2']) -> C_F_F__2.table_C_F_F__2: ...
    @overload
    def __getitem__(self, tag: Literal['COLR']) -> C_O_L_R_.table_C_O_L_R_: ...
    @overload
    def __getitem__(self, tag: Literal['CPAL']) -> C_P_A_L_.table_C_P_A_L_: ...
    @overload
    def __getitem__(self, tag: Literal['DSIG']) -> D_S_I_G_.table_D_S_I_G_: ...
    @overload
    def __getitem__(self, tag: Literal['EBDT']) -> E_B_D_T_.table_E_B_D_T_: ...
    @overload
    def __getitem__(self, tag: Literal['EBLC']) -> E_B_L_C_.table_E_B_L_C_: ...
    @overload
    def __getitem__(self, tag: Literal['FFTM']) -> F_F_T_M_.table_F_F_T_M_: ...
    @overload
    def __getitem__(self, tag: Literal['GDEF']) -> G_D_E_F_.table_G_D_E_F_: ...
    @overload
    def __getitem__(self, tag: Literal['GMAP']) -> G_M_A_P_.table_G_M_A_P_: ...
    @overload
    def __getitem__(self, tag: Literal['GPKG']) -> G_P_K_G_.table_G_P_K_G_: ...
    @overload
    def __getitem__(self, tag: Literal['GPOS']) -> G_P_O_S_.table_G_P_O_S_: ...
    @overload
    def __getitem__(self, tag: Literal['GSUB']) -> G_S_U_B_.table_G_S_U_B_: ...
    @overload
    def __getitem__(self, tag: Literal['GVAR']) -> G_V_A_R_.table_G_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal['HVAR']) -> H_V_A_R_.table_H_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal['JSTF']) -> J_S_T_F_.table_J_S_T_F_: ...
    @overload
    def __getitem__(self, tag: Literal['LTSH']) -> L_T_S_H_.table_L_T_S_H_: ...
    @overload
    def __getitem__(self, tag: Literal['MATH']) -> M_A_T_H_.table_M_A_T_H_: ...
    @overload
    def __getitem__(self, tag: Literal['META']) -> M_E_T_A_.table_M_E_T_A_: ...
    @overload
    def __getitem__(self, tag: Literal['MVAR']) -> M_V_A_R_.table_M_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal['SING']) -> S_I_N_G_.table_S_I_N_G_: ...
    @overload
    def __getitem__(self, tag: Literal['STAT']) -> S_T_A_T_.table_S_T_A_T_: ...
    @overload
    def __getitem__(self, tag: Literal['SVG']) -> S_V_G_.table_S_V_G_: ...
    @overload
    def __getitem__(self, tag: Literal['TSI0']) -> T_S_I__0.table_T_S_I__0: ...
    @overload
    def __getitem__(self, tag: Literal['TSI1']) -> T_S_I__1.table_T_S_I__1: ...
    @overload
    def __getitem__(self, tag: Literal['TSI2']) -> T_S_I__2.table_T_S_I__2: ...
    @overload
    def __getitem__(self, tag: Literal['TSI3']) -> T_S_I__3.table_T_S_I__3: ...
    @overload
    def __getitem__(self, tag: Literal['TSI5']) -> T_S_I__5.table_T_S_I__5: ...
    @overload
    def __getitem__(self, tag: Literal['TSIB']) -> T_S_I_B_.table_T_S_I_B_: ...
    @overload
    def __getitem__(self, tag: Literal['TSIC']) -> T_S_I_C_.table_T_S_I_C_: ...
    @overload
    def __getitem__(self, tag: Literal['TSID']) -> T_S_I_D_.table_T_S_I_D_: ...
    @overload
    def __getitem__(self, tag: Literal['TSIJ']) -> T_S_I_J_.table_T_S_I_J_: ...
    @overload
    def __getitem__(self, tag: Literal['TSIP']) -> T_S_I_P_.table_T_S_I_P_: ...
    @overload
    def __getitem__(self, tag: Literal['TSIS']) -> T_S_I_S_.table_T_S_I_S_: ...
    @overload
    def __getitem__(self, tag: Literal['TSIV']) -> T_S_I_V_.table_T_S_I_V_: ...
    @overload
    def __getitem__(self, tag: Literal['TTFA']) -> T_T_F_A_.table_T_T_F_A_: ...
    @overload
    def __getitem__(self, tag: Literal['VARC']) -> V_A_R_C_.table_V_A_R_C_: ...
    @overload
    def __getitem__(self, tag: Literal['VDMX']) -> V_D_M_X_.table_V_D_M_X_: ...
    @overload
    def __getitem__(self, tag: Literal['VORG']) -> V_O_R_G_.table_V_O_R_G_: ...
    @overload
    def __getitem__(self, tag: Literal['VVAR']) -> V_V_A_R_.table_V_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal['Debg']) -> D__e_b_g.table_D__e_b_g: ...
    @overload
    def __getitem__(self, tag: Literal['Feat']) -> F__e_a_t.table_F__e_a_t: ...
    @overload
    def __getitem__(self, tag: Literal['Glat']) -> G__l_a_t.table_G__l_a_t: ...
    @overload
    def __getitem__(self, tag: Literal['Gloc']) -> G__l_o_c.table_G__l_o_c: ...
    @overload
    def __getitem__(self, tag: Literal['OS/2']) -> O_S_2f_2.table_O_S_2f_2: ...
    @overload
    def __getitem__(self, tag: Literal['Silf']) -> S__i_l_f.table_S__i_l_f: ...
    @overload
    def __getitem__(self, tag: Literal['Sill']) -> S__i_l_l.table_S__i_l_l: ...
    @overload
    def __getitem__(self, tag: Literal['ankr']) -> _a_n_k_r.table__a_n_k_r: ...
    @overload
    def __getitem__(self, tag: Literal['avar']) -> _a_v_a_r.table__a_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal['bsln']) -> _b_s_l_n.table__b_s_l_n: ...
    @overload
    def __getitem__(self, tag: Literal['cidg']) -> _c_i_d_g.table__c_i_d_g: ...
    @overload
    def __getitem__(self, tag: Literal['cmap']) -> _c_m_a_p.table__c_m_a_p: ...
    @overload
    def __getitem__(self, tag: Literal['cvar']) -> _c_v_a_r.table__c_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal['cvt']) -> _c_v_t.table__c_v_t: ...
    @overload
    def __getitem__(self, tag: Literal['feat']) -> _f_e_a_t.table__f_e_a_t: ...
    @overload
    def __getitem__(self, tag: Literal['fpgm']) -> _f_p_g_m.table__f_p_g_m: ...
    @overload
    def __getitem__(self, tag: Literal['fvar']) -> _f_v_a_r.table__f_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal['gasp']) -> _g_a_s_p.table__g_a_s_p: ...
    @overload
    def __getitem__(self, tag: Literal['gcid']) -> _g_c_i_d.table__g_c_i_d: ...
    @overload
    def __getitem__(self, tag: Literal['glyf']) -> _g_l_y_f.table__g_l_y_f: ...
    @overload
    def __getitem__(self, tag: Literal['gvar']) -> _g_v_a_r.table__g_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal['hdmx']) -> _h_d_m_x.table__h_d_m_x: ...
    @overload
    def __getitem__(self, tag: Literal['head']) -> _h_e_a_d.table__h_e_a_d: ...
    @overload
    def __getitem__(self, tag: Literal['hhea']) -> _h_h_e_a.table__h_h_e_a: ...
    @overload
    def __getitem__(self, tag: Literal['hmtx']) -> _h_m_t_x.table__h_m_t_x: ...
    @overload
    def __getitem__(self, tag: Literal['kern']) -> _k_e_r_n.table__k_e_r_n: ...
    @overload
    def __getitem__(self, tag: Literal['lcar']) -> _l_c_a_r.table__l_c_a_r: ...
    @overload
    def __getitem__(self, tag: Literal['loca']) -> _l_o_c_a.table__l_o_c_a: ...
    @overload
    def __getitem__(self, tag: Literal['ltag']) -> _l_t_a_g.table__l_t_a_g: ...
    @overload
    def __getitem__(self, tag: Literal['maxp']) -> _m_a_x_p.table__m_a_x_p: ...
    @overload
    def __getitem__(self, tag: Literal['meta']) -> _m_e_t_a.table__m_e_t_a: ...
    @overload
    def __getitem__(self, tag: Literal['mort']) -> _m_o_r_t.table__m_o_r_t: ...
    @overload
    def __getitem__(self, tag: Literal['morx']) -> _m_o_r_x.table__m_o_r_x: ...
    @overload
    def __getitem__(self, tag: Literal['name']) -> _n_a_m_e.table__n_a_m_e: ...
    @overload
    def __getitem__(self, tag: Literal['opbd']) -> _o_p_b_d.table__o_p_b_d: ...
    @overload
    def __getitem__(self, tag: Literal['post']) -> _p_o_s_t.table__p_o_s_t: ...
    @overload
    def __getitem__(self, tag: Literal['prep']) -> _p_r_e_p.table__p_r_e_p: ...
    @overload
    def __getitem__(self, tag: Literal['prop']) -> _p_r_o_p.table__p_r_o_p: ...
    @overload
    def __getitem__(self, tag: Literal['sbix']) -> _s_b_i_x.table__s_b_i_x: ...
    @overload
    def __getitem__(self, tag: Literal['trak']) -> _t_r_a_k.table__t_r_a_k: ...
    @overload
    def __getitem__(self, tag: Literal['vhea']) -> _v_h_e_a.table__v_h_e_a: ...
    @overload
    def __getitem__(self, tag: Literal['vmtx']) -> _v_m_t_x.table__v_m_t_x: ...
    @overload
    def __getitem__(self, tag: Literal['GlyphOrder']) -> GlyphOrder: ...
    @overload
    def __getitem__(self, tag: str | bytes) -> DefaultTable | GlyphOrder: ...
    def _readTable(self, tag: Tag) -> DefaultTable: ...
    def __setitem__(self, tag: str | bytes, table: DefaultTable) -> None: ...
    def __delitem__(self, tag: str | bytes) -> None: ...
    @overload
    def get(self, tag: Literal['BASE']) -> B_A_S_E_.table_B_A_S_E_ | None: ...
    @overload
    def get(self, tag: Literal['CBDT']) -> C_B_D_T_.table_C_B_D_T_ | None: ...
    @overload
    def get(self, tag: Literal['CBLC']) -> C_B_L_C_.table_C_B_L_C_ | None: ...
    @overload
    def get(self, tag: Literal['CFF']) -> C_F_F_.table_C_F_F_ | None: ...
    @overload
    def get(self, tag: Literal['CFF2']) -> C_F_F__2.table_C_F_F__2 | None: ...
    @overload
    def get(self, tag: Literal['COLR']) -> C_O_L_R_.table_C_O_L_R_ | None: ...
    @overload
    def get(self, tag: Literal['CPAL']) -> C_P_A_L_.table_C_P_A_L_ | None: ...
    @overload
    def get(self, tag: Literal['DSIG']) -> D_S_I_G_.table_D_S_I_G_ | None: ...
    @overload
    def get(self, tag: Literal['EBDT']) -> E_B_D_T_.table_E_B_D_T_ | None: ...
    @overload
    def get(self, tag: Literal['EBLC']) -> E_B_L_C_.table_E_B_L_C_ | None: ...
    @overload
    def get(self, tag: Literal['FFTM']) -> F_F_T_M_.table_F_F_T_M_ | None: ...
    @overload
    def get(self, tag: Literal['GDEF']) -> G_D_E_F_.table_G_D_E_F_ | None: ...
    @overload
    def get(self, tag: Literal['GMAP']) -> G_M_A_P_.table_G_M_A_P_ | None: ...
    @overload
    def get(self, tag: Literal['GPKG']) -> G_P_K_G_.table_G_P_K_G_ | None: ...
    @overload
    def get(self, tag: Literal['GPOS']) -> G_P_O_S_.table_G_P_O_S_ | None: ...
    @overload
    def get(self, tag: Literal['GSUB']) -> G_S_U_B_.table_G_S_U_B_ | None: ...
    @overload
    def get(self, tag: Literal['GVAR']) -> G_V_A_R_.table_G_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal['HVAR']) -> H_V_A_R_.table_H_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal['JSTF']) -> J_S_T_F_.table_J_S_T_F_ | None: ...
    @overload
    def get(self, tag: Literal['LTSH']) -> L_T_S_H_.table_L_T_S_H_ | None: ...
    @overload
    def get(self, tag: Literal['MATH']) -> M_A_T_H_.table_M_A_T_H_ | None: ...
    @overload
    def get(self, tag: Literal['META']) -> M_E_T_A_.table_M_E_T_A_ | None: ...
    @overload
    def get(self, tag: Literal['MVAR']) -> M_V_A_R_.table_M_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal['SING']) -> S_I_N_G_.table_S_I_N_G_ | None: ...
    @overload
    def get(self, tag: Literal['STAT']) -> S_T_A_T_.table_S_T_A_T_ | None: ...
    @overload
    def get(self, tag: Literal['SVG']) -> S_V_G_.table_S_V_G_ | None: ...
    @overload
    def get(self, tag: Literal['TSI0']) -> T_S_I__0.table_T_S_I__0 | None: ...
    @overload
    def get(self, tag: Literal['TSI1']) -> T_S_I__1.table_T_S_I__1 | None: ...
    @overload
    def get(self, tag: Literal['TSI2']) -> T_S_I__2.table_T_S_I__2 | None: ...
    @overload
    def get(self, tag: Literal['TSI3']) -> T_S_I__3.table_T_S_I__3 | None: ...
    @overload
    def get(self, tag: Literal['TSI5']) -> T_S_I__5.table_T_S_I__5 | None: ...
    @overload
    def get(self, tag: Literal['TSIB']) -> T_S_I_B_.table_T_S_I_B_ | None: ...
    @overload
    def get(self, tag: Literal['TSIC']) -> T_S_I_C_.table_T_S_I_C_ | None: ...
    @overload
    def get(self, tag: Literal['TSID']) -> T_S_I_D_.table_T_S_I_D_ | None: ...
    @overload
    def get(self, tag: Literal['TSIJ']) -> T_S_I_J_.table_T_S_I_J_ | None: ...
    @overload
    def get(self, tag: Literal['TSIP']) -> T_S_I_P_.table_T_S_I_P_ | None: ...
    @overload
    def get(self, tag: Literal['TSIS']) -> T_S_I_S_.table_T_S_I_S_ | None: ...
    @overload
    def get(self, tag: Literal['TSIV']) -> T_S_I_V_.table_T_S_I_V_ | None: ...
    @overload
    def get(self, tag: Literal['TTFA']) -> T_T_F_A_.table_T_T_F_A_ | None: ...
    @overload
    def get(self, tag: Literal['VARC']) -> V_A_R_C_.table_V_A_R_C_ | None: ...
    @overload
    def get(self, tag: Literal['VDMX']) -> V_D_M_X_.table_V_D_M_X_ | None: ...
    @overload
    def get(self, tag: Literal['VORG']) -> V_O_R_G_.table_V_O_R_G_ | None: ...
    @overload
    def get(self, tag: Literal['VVAR']) -> V_V_A_R_.table_V_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal['Debg']) -> D__e_b_g.table_D__e_b_g | None: ...
    @overload
    def get(self, tag: Literal['Feat']) -> F__e_a_t.table_F__e_a_t | None: ...
    @overload
    def get(self, tag: Literal['Glat']) -> G__l_a_t.table_G__l_a_t | None: ...
    @overload
    def get(self, tag: Literal['Gloc']) -> G__l_o_c.table_G__l_o_c | None: ...
    @overload
    def get(self, tag: Literal['OS/2']) -> O_S_2f_2.table_O_S_2f_2 | None: ...
    @overload
    def get(self, tag: Literal['Silf']) -> S__i_l_f.table_S__i_l_f | None: ...
    @overload
    def get(self, tag: Literal['Sill']) -> S__i_l_l.table_S__i_l_l | None: ...
    @overload
    def get(self, tag: Literal['ankr']) -> _a_n_k_r.table__a_n_k_r | None: ...
    @overload
    def get(self, tag: Literal['avar']) -> _a_v_a_r.table__a_v_a_r | None: ...
    @overload
    def get(self, tag: Literal['bsln']) -> _b_s_l_n.table__b_s_l_n | None: ...
    @overload
    def get(self, tag: Literal['cidg']) -> _c_i_d_g.table__c_i_d_g | None: ...
    @overload
    def get(self, tag: Literal['cmap']) -> _c_m_a_p.table__c_m_a_p | None: ...
    @overload
    def get(self, tag: Literal['cvar']) -> _c_v_a_r.table__c_v_a_r | None: ...
    @overload
    def get(self, tag: Literal['cvt']) -> _c_v_t.table__c_v_t | None: ...
    @overload
    def get(self, tag: Literal['feat']) -> _f_e_a_t.table__f_e_a_t | None: ...
    @overload
    def get(self, tag: Literal['fpgm']) -> _f_p_g_m.table__f_p_g_m | None: ...
    @overload
    def get(self, tag: Literal['fvar']) -> _f_v_a_r.table__f_v_a_r | None: ...
    @overload
    def get(self, tag: Literal['gasp']) -> _g_a_s_p.table__g_a_s_p | None: ...
    @overload
    def get(self, tag: Literal['gcid']) -> _g_c_i_d.table__g_c_i_d | None: ...
    @overload
    def get(self, tag: Literal['glyf']) -> _g_l_y_f.table__g_l_y_f | None: ...
    @overload
    def get(self, tag: Literal['gvar']) -> _g_v_a_r.table__g_v_a_r | None: ...
    @overload
    def get(self, tag: Literal['hdmx']) -> _h_d_m_x.table__h_d_m_x | None: ...
    @overload
    def get(self, tag: Literal['head']) -> _h_e_a_d.table__h_e_a_d | None: ...
    @overload
    def get(self, tag: Literal['hhea']) -> _h_h_e_a.table__h_h_e_a | None: ...
    @overload
    def get(self, tag: Literal['hmtx']) -> _h_m_t_x.table__h_m_t_x | None: ...
    @overload
    def get(self, tag: Literal['kern']) -> _k_e_r_n.table__k_e_r_n | None: ...
    @overload
    def get(self, tag: Literal['lcar']) -> _l_c_a_r.table__l_c_a_r | None: ...
    @overload
    def get(self, tag: Literal['loca']) -> _l_o_c_a.table__l_o_c_a | None: ...
    @overload
    def get(self, tag: Literal['ltag']) -> _l_t_a_g.table__l_t_a_g | None: ...
    @overload
    def get(self, tag: Literal['maxp']) -> _m_a_x_p.table__m_a_x_p | None: ...
    @overload
    def get(self, tag: Literal['meta']) -> _m_e_t_a.table__m_e_t_a | None: ...
    @overload
    def get(self, tag: Literal['mort']) -> _m_o_r_t.table__m_o_r_t | None: ...
    @overload
    def get(self, tag: Literal['morx']) -> _m_o_r_x.table__m_o_r_x | None: ...
    @overload
    def get(self, tag: Literal['name']) -> _n_a_m_e.table__n_a_m_e | None: ...
    @overload
    def get(self, tag: Literal['opbd']) -> _o_p_b_d.table__o_p_b_d | None: ...
    @overload
    def get(self, tag: Literal['post']) -> _p_o_s_t.table__p_o_s_t | None: ...
    @overload
    def get(self, tag: Literal['prep']) -> _p_r_e_p.table__p_r_e_p | None: ...
    @overload
    def get(self, tag: Literal['prop']) -> _p_r_o_p.table__p_r_o_p | None: ...
    @overload
    def get(self, tag: Literal['sbix']) -> _s_b_i_x.table__s_b_i_x | None: ...
    @overload
    def get(self, tag: Literal['trak']) -> _t_r_a_k.table__t_r_a_k | None: ...
    @overload
    def get(self, tag: Literal['vhea']) -> _v_h_e_a.table__v_h_e_a | None: ...
    @overload
    def get(self, tag: Literal['vmtx']) -> _v_m_t_x.table__v_m_t_x | None: ...
    @overload
    def get(self, tag: Literal['GlyphOrder']) -> GlyphOrder: ...
    @overload
    def get(self, tag: str | bytes) -> DefaultTable | GlyphOrder | Any | None: ...
    @overload
    def get(self, tag: str | bytes, default: _VT_co) -> DefaultTable | GlyphOrder | Any | _VT_co: ...
    def setGlyphOrder(self, glyphOrder: list[str]) -> None:
        """Set the glyph order

        Args:
                glyphOrder ([str]): List of glyph names in order.
        """
    def getGlyphOrder(self) -> list[str]:
        """Returns a list of glyph names ordered by their position in the font."""
    def _getGlyphNamesFromCmap(self) -> None: ...
    @staticmethod
    def _makeGlyphName(codepoint: int) -> str: ...
    def getGlyphNames(self) -> list[str]:
        """Get a list of glyph names, sorted alphabetically."""
    def getGlyphNames2(self) -> list[str]:
        """Get a list of glyph names, sorted alphabetically,
        but not case sensitive.
        """
    def getGlyphName(self, glyphID: int) -> str:
        """Returns the name for the glyph with the given ID.

        If no name is available, synthesises one with the form ``glyphXXXXX``` where
        ```XXXXX`` is the zero-padded glyph ID.
        """
    def getGlyphNameMany(self, lst: Sequence[int]) -> list[str]:
        """Converts a list of glyph IDs into a list of glyph names."""
    def getGlyphID(self, glyphName: str) -> int:
        """Returns the ID of the glyph with the given name."""
    def getGlyphIDMany(self, lst: Sequence[str]) -> list[int]:
        """Converts a list of glyph names into a list of glyph IDs."""
    def getReverseGlyphMap(self, rebuild: bool = False) -> dict[str, int]:
        """Returns a mapping of glyph names to glyph IDs."""
    def _buildReverseGlyphOrderDict(self) -> dict[str, int]: ...
    def _writeTable(self, tag: str | bytes, writer: SFNTWriter, done: list[str | bytes], tableCache: MutableMapping[tuple[Tag, bytes], DefaultTable] | None = None) -> None:
        """Internal helper function for self.save(). Keeps track of inter-table dependencies."""
    def getTableData(self, tag: str | bytes) -> bytes:
        """Returns the binary representation of a table.

        If the table is currently loaded and in memory, the data is compiled to
        binary and returned; if it is not currently loaded, the binary data is
        read from the font file and returned.
        """
    def getGlyphSet(self, preferCFF: bool = True, location: Mapping[str, _NumberT] | None = None, normalized: bool = False, recalcBounds: bool = True) -> _TTGlyphSet:
        """Return a generic GlyphSet, which is a dict-like object
        mapping glyph names to glyph objects. The returned glyph objects
        have a ``.draw()`` method that supports the Pen protocol, and will
        have an attribute named 'width'.

        If the font is CFF-based, the outlines will be taken from the ``CFF ``
        or ``CFF2`` tables. Otherwise the outlines will be taken from the
        ``glyf`` table.

        If the font contains both a ``CFF ``/``CFF2`` and a ``glyf`` table, you
        can use the ``preferCFF`` argument to specify which one should be taken.
        If the font contains both a ``CFF `` and a ``CFF2`` table, the latter is
        taken.

        If the ``location`` parameter is set, it should be a dictionary mapping
        four-letter variation tags to their float values, and the returned
        glyph-set will represent an instance of a variable font at that
        location.

        If the ``normalized`` variable is set to True, that location is
        interpreted as in the normalized (-1..+1) space, otherwise it is in the
        font's defined axes space.
        """
    def normalizeLocation(self, location: Mapping[str, float]) -> dict[str, float]:
        """Normalize a ``location`` from the font's defined axes space (also
        known as user space) into the normalized (-1..+1) space. It applies
        ``avar`` mapping if the font contains an ``avar`` table.

        The ``location`` parameter should be a dictionary mapping four-letter
        variation tags to their float values.

        Raises ``TTLibError`` if the font is not a variable font.
        """
    def getBestCmap(self, cmapPreferences: Sequence[tuple[int, int]] = ...) -> dict[int, str] | None:
        """Returns the 'best' Unicode cmap dictionary available in the font
        or ``None``, if no Unicode cmap subtable is available.

        By default it will search for the following (platformID, platEncID)
        pairs in order::

                        (3, 10), # Windows Unicode full repertoire
                        (0, 6),  # Unicode full repertoire (format 13 subtable)
                        (0, 4),  # Unicode 2.0 full repertoire
                        (3, 1),  # Windows Unicode BMP
                        (0, 3),  # Unicode 2.0 BMP
                        (0, 2),  # Unicode ISO/IEC 10646
                        (0, 1),  # Unicode 1.1
                        (0, 0)   # Unicode 1.0

        This particular order matches what HarfBuzz uses to choose what
        subtable to use by default. This order prefers the largest-repertoire
        subtable, and among those, prefers the Windows-platform over the
        Unicode-platform as the former has wider support.

        This order can be customized via the ``cmapPreferences`` argument.
        """
    def reorderGlyphs(self, new_glyph_order: list[str]) -> None: ...

class GlyphOrder:
    """A pseudo table. The glyph order isn't in the font as a separate
    table, but it's nice to present it as such in the TTX format.
    """

    def __init__(self, tag: str | None = None) -> None: ...
    def toXML(self, writer: xmlWriter.XMLWriter, ttFont: TTFont) -> None: ...
    glyphOrder: Incomplete
    def fromXML(self, name: str, attrs: dict[str, str], content: list[Any], ttFont: TTFont) -> None: ...

def getTableModule(tag: str | bytes) -> ModuleType | None:
    """Fetch the packer/unpacker module for a table.
    Return None when no module is found.
    """

_customTableRegistry: dict[str | bytes, tuple[str, str]]

def registerCustomTableClass(tag: str | bytes, moduleName: str, className: str | None = None) -> None:
    """Register a custom packer/unpacker class for a table.

    The 'moduleName' must be an importable module. If no 'className'
    is given, it is derived from the tag, for example it will be
    ``table_C_U_S_T_`` for a 'CUST' tag.

    The registered table class should be a subclass of
    :py:class:`fontTools.ttLib.tables.DefaultTable.DefaultTable`
    """
def unregisterCustomTableClass(tag: str | bytes) -> None:
    """Unregister the custom packer/unpacker class for a table."""
def getCustomTableClass(tag: str | bytes) -> type[DefaultTable] | None:
    """Return the custom table class for tag, if one has been registered
    with 'registerCustomTableClass()'. Else return None.
    """
def getTableClass(tag: str | bytes) -> type[DefaultTable]:
    """Fetch the packer/unpacker class for a table."""
def getClassTag(klass: type[DefaultTable]) -> str | bytes:
    """Fetch the table tag for a class object."""
def newTable(tag: str | bytes) -> DefaultTable:
    """Return a new instance of a table."""
def _escapechar(c: str) -> str:
    """Helper function for tagToIdentifier()"""
def tagToIdentifier(tag: str | bytes) -> str:
    """Convert a table tag to a valid (but UGLY) python identifier,
    as well as a filename that's guaranteed to be unique even on a
    caseless file system. Each character is mapped to two characters.
    Lowercase letters get an underscore before the letter, uppercase
    letters get an underscore after the letter. Trailing spaces are
    trimmed. Illegal characters are escaped as two hex bytes. If the
    result starts with a number (as the result of a hex escape), an
    extra underscore is prepended. Examples:
    .. code-block:: pycon

        >>>
        >> tagToIdentifier('glyf')
        '_g_l_y_f'
        >> tagToIdentifier('cvt ')
        '_c_v_t'
        >> tagToIdentifier('OS/2')
        'O_S_2f_2'
    """
def identifierToTag(ident: str) -> str:
    """The opposite of tagToIdentifier()"""
def tagToXML(tag: str | bytes) -> str:
    """Similarly to tagToIdentifier(), this converts a TT tag
    to a valid XML element name. Since XML element names are
    case sensitive, this is a fairly simple/readable translation.
    """
def xmlToTag(tag: str) -> str:
    """The opposite of tagToXML()"""

TTFTableOrder: Incomplete
OTFTableOrder: Incomplete

def sortedTagList(tagList: Sequence[str], tableOrder: Sequence[str] | None = None) -> list[str]:
    """Return a sorted copy of tagList, sorted according to the OpenType
    specification, or according to a custom tableOrder. If given and not
    None, tableOrder needs to be a list of tag names.
    """
def reorderFontTables(inFile: BinaryIO, outFile: BinaryIO, tableOrder: Sequence[str] | None = None, checkChecksums: bool = False) -> None:
    """Rewrite a font file, ordering the tables as recommended by the
    OpenType specification 1.4.
    """
def maxPowerOfTwo(x: int) -> int:
    """Return the highest exponent of two, so that
    (2 ** exponent) <= x.  Return 0 if x is 0.
    """
def getSearchRange(n: int, itemSize: int = 16) -> tuple[int, int, int]:
    """Calculate searchRange, entrySelector, rangeShift."""

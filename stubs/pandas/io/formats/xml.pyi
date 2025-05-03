from _typeshed import Incomplete
from pandas._libs.lib import is_list_like as is_list_like
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.dtypes.missing import isna as isna
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.io.common import get_handle as get_handle
from pandas.io.xml import get_data_from_filepath as get_data_from_filepath, preprocess_data as preprocess_data
from pandas.util._decorators import doc as doc
from typing import Any, ClassVar

TYPE_CHECKING: bool
_shared_docs: dict

class _BaseXMLFormatter:
    _docstring_components: ClassVar[list] = ...
    _sub_element_cls: Incomplete
    def __init__(self, frame: DataFrame, path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None, index: bool = ..., root_name: str | None = ..., row_name: str | None = ..., na_rep: str | None, attr_cols: list[str] | None, elem_cols: list[str] | None, namespaces: dict[str | None, str] | None, prefix: str | None, encoding: str = ..., xml_declaration: bool | None = ..., pretty_print: bool | None = ..., stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None) -> None: ...
    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
    def _validate_columns(self) -> None:
        """
        Validate elems_cols and attrs_cols.

        This method will check if columns is list-like.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
    def _validate_encoding(self) -> None:
        """
        Validate encoding.

        This method will check if encoding is among listed under codecs.

        Raises
        ------
        LookupError
            * If encoding is not available in codecs.
        """
    def _process_dataframe(self) -> dict[int | str, dict[str, Any]]:
        """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """
    def _handle_indexes(self) -> None:
        """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """
    def _get_prefix_uri(self) -> str:
        """
        Get uri of namespace prefix.

        This method retrieves corresponding URI to prefix in namespaces.

        Raises
        ------
        KeyError
            *If prefix is not included in namespace dict.
        """
    def _other_namespaces(self) -> dict:
        """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """
    def _build_attribs(self, d: dict[str, Any], elem_row: Any) -> Any:
        """
        Create attributes of row.

        This method adds attributes using attr_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
    def _get_flat_col_name(self, col: str | tuple) -> str: ...
    def _build_elems(self, d: dict[str, Any], elem_row: Any) -> None:
        """
        Create child elements of row.

        This method adds child elements using elem_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
    def write_output(self) -> str | None: ...

class EtreeXMLFormatter(_BaseXMLFormatter):
    _sub_element_cls: Incomplete
    def _build_tree(self) -> bytes: ...
    def _get_prefix_uri(self) -> str: ...
    def _prettify_tree(self) -> bytes:
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """

class LxmlXMLFormatter(_BaseXMLFormatter):
    _sub_element_cls: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
    def _convert_empty_str_key(self) -> None:
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """
    def _get_prefix_uri(self) -> str: ...
    def _transform_doc(self) -> bytes:
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse stylesheet object into tree for parsing
        conditionally by its specific object type, then transforms
        original tree with XSLT script.
        """

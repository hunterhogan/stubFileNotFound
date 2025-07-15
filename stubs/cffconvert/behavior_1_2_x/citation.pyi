from typing import Any
from cffconvert.behavior_1_2_x.apalike_object import ApalikeObject as ApalikeObject
from cffconvert.behavior_1_2_x.bibtex_object import BibtexObject as BibtexObject
from cffconvert.behavior_1_2_x.codemeta_object import CodemetaObject as CodemetaObject
from cffconvert.behavior_1_2_x.endnote_object import EndnoteObject as EndnoteObject
from cffconvert.behavior_1_2_x.ris_object import RisObject as RisObject
from cffconvert.behavior_1_2_x.schemaorg_object import SchemaorgObject as SchemaorgObject
from cffconvert.behavior_1_2_x.zenodo_object import ZenodoObject as ZenodoObject
from cffconvert.contracts.citation import Contract as Contract
from cffconvert.root import get_package_root as get_package_root

class Citation_1_2_x(Contract):
    supported_cff_versions: list[str]
    cffstr: str
    cffversion: str
    cffobj: dict[str, Any]
    schema: dict[str, Any]

    def __init__(self, cffstr: str, cffversion: str) -> None: ...

    def _get_schema(self) -> dict[str, Any]: ...

    def _parse(self) -> dict[str, Any]: ...

    def as_apalike(self) -> str: ...

    def as_bibtex(self, reference: str = 'YourReferenceHere') -> str: ...

    def as_cff(self) -> str: ...

    def as_codemeta(self) -> str: ...

    def as_endnote(self) -> str: ...

    def as_ris(self) -> str: ...

    def as_schemaorg(self) -> str: ...

    def as_zenodo(self) -> str: ...

    def validate(self, verbose: bool = True) -> None: ...

from _typeshed import Incomplete
from cffconvert.behavior_1_2_x.schemaorg_object import SchemaorgObject as SchemaorgObject

class CodemetaObject(SchemaorgObject):
    supported_cff_versions: Incomplete
    supported_codemeta_props: Incomplete
    def __init__(self, cffobj, initialize_empty: bool = False) -> None: ...

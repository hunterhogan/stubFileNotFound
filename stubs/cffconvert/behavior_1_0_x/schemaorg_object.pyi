from _typeshed import Incomplete
from cffconvert.behavior_1_0_x.schemaorg_author import SchemaorgAuthor as SchemaorgAuthor
from cffconvert.behavior_1_0_x.schemaorg_urls import SchemaorgUrls as SchemaorgUrls
from cffconvert.behavior_shared.schemaorg_object_shared import SchemaorgObjectShared as Shared

class SchemaorgObject(Shared):
    supported_cff_versions: Incomplete
    author: Incomplete
    def add_author(self): ...
    date_published: Incomplete
    def add_date_published(self): ...
    identifier: Incomplete
    def add_identifier(self): ...
    def add_urls(self): ...

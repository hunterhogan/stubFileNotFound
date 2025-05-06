from _typeshed import Incomplete
from cffconvert.behavior_1_1_x.apalike_author import ApalikeAuthor as ApalikeAuthor
from cffconvert.behavior_1_1_x.apalike_url import ApalikeUrl as ApalikeUrl
from cffconvert.behavior_shared.apalike_object_shared import ApalikeObjectShared as Shared

class ApalikeObject(Shared):
    supported_cff_versions: Incomplete
    author: Incomplete
    def add_author(self): ...
    year: Incomplete
    def add_year(self): ...
    doi: Incomplete
    def add_doi(self): ...
    url: Incomplete
    def add_url(self): ...

from _typeshed import Incomplete
from cffconvert.behavior_1_2_x.ris_author import RisAuthor as RisAuthor
from cffconvert.behavior_1_2_x.ris_url import RisUrl as RisUrl
from cffconvert.behavior_shared.ris_object_shared import RisObjectShared as Shared

class RisObject(Shared):
    supported_cff_versions: Incomplete
    author: Incomplete
    def add_author(self): ...
    date: Incomplete
    def add_date(self): ...
    doi: Incomplete
    def add_doi(self): ...
    url: Incomplete
    def add_url(self): ...
    year: Incomplete
    def add_year(self): ...

from _typeshed import Incomplete
from cffconvert.behavior_1_2_x.bibtex_author import BibtexAuthor as BibtexAuthor
from cffconvert.behavior_1_2_x.bibtex_url import BibtexUrl as BibtexUrl
from cffconvert.behavior_shared.bibtex_object_shared import BibtexObjectShared as Shared

class BibtexObject(Shared):
    supported_cff_versions: Incomplete
    author: Incomplete
    def add_author(self): ...
    doi: Incomplete
    def add_doi(self): ...
    month: Incomplete
    def add_month(self): ...
    url: Incomplete
    def add_url(self): ...
    year: Incomplete
    def add_year(self): ...

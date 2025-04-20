__all__ = ['merge', 'merge_with']
@cytoolz.curry
def merge(d, *dicts, **kwargs):
    ...

@cytoolz.curry
def merge_with(func, d, *dicts, **kwargs):
    ...

from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.category import CategoricalIndex as CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex as IntervalIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.period import PeriodIndex as PeriodIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex

def get_objs_combined_axis(
    objs, intersect: bool = False, axis=0, sort: bool = True
) -> Index: ...
def union_indexes(indexes, sort=True) -> Index: ...
def all_indexes_same(indexes): ...

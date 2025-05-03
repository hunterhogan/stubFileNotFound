import numpy as np
from pandas._typing import AxisInt as AxisInt, Scalar as Scalar

def shift(values: np.ndarray, periods: int, axis: AxisInt, fill_value: Scalar) -> np.ndarray: ...

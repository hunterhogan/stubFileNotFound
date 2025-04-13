from _typeshed import Incomplete
from matplotlib import _api as _api, cbook as cbook, ticker as ticker, units as units

_log: Incomplete

class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """
        Convert strings in *value* to floats using mapping information stored
        in the *unit* object.

        Parameters
        ----------
        value : str or iterable
            Value or list of values to be converted.
        unit : `.UnitData`
            An object mapping strings to integers.
        axis : `~matplotlib.axis.Axis`
            The axis on which the converted value is plotted.

            .. note:: *axis* is unused.

        Returns
        -------
        float or `~numpy.ndarray` of float
        """
    @staticmethod
    def axisinfo(unit, axis):
        """
        Set the default axis ticks and labels.

        Parameters
        ----------
        unit : `.UnitData`
            object string unit information for value
        axis : `~matplotlib.axis.Axis`
            axis for which information is being set

            .. note:: *axis* is not used

        Returns
        -------
        `~matplotlib.units.AxisInfo`
            Information to support default tick labeling

        """
    @staticmethod
    def default_units(data, axis):
        """
        Set and update the `~matplotlib.axis.Axis` units.

        Parameters
        ----------
        data : str or iterable of str
        axis : `~matplotlib.axis.Axis`
            axis on which the data is plotted

        Returns
        -------
        `.UnitData`
            object storing string to integer mapping
        """
    @staticmethod
    def _validate_unit(unit) -> None: ...

class StrCategoryLocator(ticker.Locator):
    """Tick at every integer mapping of the string data."""
    _units: Incomplete
    def __init__(self, units_mapping) -> None:
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...

class StrCategoryFormatter(ticker.Formatter):
    """String representation of the data at every tick."""
    _units: Incomplete
    def __init__(self, units_mapping) -> None:
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_ticks(self, values): ...
    @staticmethod
    def _text(value):
        """Convert text values into utf-8 or ascii strings."""

class UnitData:
    _mapping: Incomplete
    _counter: Incomplete
    def __init__(self, data: Incomplete | None = None) -> None:
        """
        Create mapping between unique categorical values and integer ids.

        Parameters
        ----------
        data : iterable
            sequence of string values
        """
    @staticmethod
    def _str_is_convertible(val):
        """
        Helper method to check whether a string can be parsed as float or date.
        """
    def update(self, data) -> None:
        """
        Map new values to integer identifiers.

        Parameters
        ----------
        data : iterable of str or bytes

        Raises
        ------
        TypeError
            If elements in *data* are neither str nor bytes.
        """

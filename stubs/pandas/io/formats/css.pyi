from _typeshed import Incomplete
from collections.abc import Generator, Iterable, Iterator
from pandas.errors import CSSWarning as CSSWarning
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Callable

def _side_expander(prop_fmt: str) -> Callable:
    """
    Wrapper to expand shorthand property into top, right, bottom, left properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """
def _border_expander(side: str = '') -> Callable:
    """
    Wrapper to expand 'border' property into border color, style, and width properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """

class CSSResolver:
    """
    A callable for parsing and resolving CSS to atomic properties.
    """
    UNIT_RATIOS: Incomplete
    FONT_SIZE_RATIOS: Incomplete
    MARGIN_RATIOS: Incomplete
    BORDER_WIDTH_RATIOS: Incomplete
    BORDER_STYLES: Incomplete
    SIDE_SHORTHANDS: Incomplete
    SIDES: Incomplete
    CSS_EXPANSIONS: Incomplete
    def __call__(self, declarations: str | Iterable[tuple[str, str]], inherited: dict[str, str] | None = None) -> dict[str, str]:
        '''
        The given declarations to atomic properties.

        Parameters
        ----------
        declarations_str : str | Iterable[tuple[str, str]]
            A CSS string or set of CSS declaration tuples
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}
        inherited : dict, optional
            Atomic properties indicating the inherited style context in which
            declarations_str is to be resolved. ``inherited`` should already
            be resolved, i.e. valid output of this method.

        Returns
        -------
        dict
            Atomic CSS 2.2 properties.

        Examples
        --------
        >>> resolve = CSSResolver()
        >>> inherited = {\'font-family\': \'serif\', \'font-weight\': \'bold\'}
        >>> out = resolve(\'\'\'
        ...               border-color: BLUE RED;
        ...               font-size: 1em;
        ...               font-size: 2em;
        ...               font-weight: normal;
        ...               font-weight: inherit;
        ...               \'\'\', inherited)
        >>> sorted(out.items())  # doctest: +NORMALIZE_WHITESPACE
        [(\'border-bottom-color\', \'blue\'),
         (\'border-left-color\', \'red\'),
         (\'border-right-color\', \'red\'),
         (\'border-top-color\', \'blue\'),
         (\'font-family\', \'serif\'),
         (\'font-size\', \'24pt\'),
         (\'font-weight\', \'bold\')]
        '''
    def _update_initial(self, props: dict[str, str], inherited: dict[str, str]) -> dict[str, str]: ...
    def _update_font_size(self, props: dict[str, str], inherited: dict[str, str]) -> dict[str, str]: ...
    def _get_font_size(self, props: dict[str, str]) -> float | None: ...
    def _get_float_font_size_from_pt(self, font_size_string: str) -> float: ...
    def _update_other_units(self, props: dict[str, str]) -> dict[str, str]: ...
    def size_to_pt(self, in_val, em_pt: Incomplete | None = None, conversions=...) -> str: ...
    def atomize(self, declarations: Iterable) -> Generator[tuple[str, str], None, None]: ...
    def parse(self, declarations_str: str) -> Iterator[tuple[str, str]]:
        """
        Generates (prop, value) pairs from declarations.

        In a future version may generate parsed tokens from tinycss/tinycss2

        Parameters
        ----------
        declarations_str : str
        """

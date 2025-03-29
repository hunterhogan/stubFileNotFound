from _typeshed import Incomplete
from numba import guvectorize

def _resample_loop(x, t_out, interp_win, interp_delta, num_table, scale, y) -> None: ...

_resample_loop_p: Incomplete
_resample_loop_s: Incomplete
@guvectorize("(n),(m),(p),(p),(),()->(m)", nopython=True)

def resample_f_p(x, t_out, interp_win, interp_delta, num_table, scale, y) -> None: ...
@guvectorize("(n),(m),(p),(p),(),()->(m)", nopython=True)
def resample_f_s(x, t_out, interp_win, interp_delta, num_table, scale, y) -> None: ...

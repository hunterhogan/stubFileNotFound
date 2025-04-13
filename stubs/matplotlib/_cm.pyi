from _typeshed import Incomplete

_binary_data: Incomplete
_autumn_data: Incomplete
_bone_data: Incomplete
_cool_data: Incomplete
_copper_data: Incomplete

def _flag_red(x): ...
def _flag_green(x): ...
def _flag_blue(x): ...

_flag_data: Incomplete

def _prism_red(x): ...
def _prism_green(x): ...
def _prism_blue(x): ...

_prism_data: Incomplete

def _ch_helper(gamma, s, r, h, p0, p1, x):
    """Helper function for generating picklable cubehelix colormaps."""
def cubehelix(gamma: float = 1.0, s: float = 0.5, r: float = -1.5, h: float = 1.0):
    """
    Return custom data dictionary of (r, g, b) conversion functions, which can
    be used with `.ColormapRegistry.register`, for the cubehelix color scheme.

    Unlike most other color schemes cubehelix was designed by D.A. Green to
    be monotonically increasing in terms of perceived brightness.
    Also, when printed on a black and white postscript printer, the scheme
    results in a greyscale with monotonically increasing brightness.
    This color scheme is named cubehelix because the (r, g, b) values produced
    can be visualised as a squashed helix around the diagonal in the
    (r, g, b) color cube.

    For a unit color cube (i.e. 3D coordinates for (r, g, b) each in the
    range 0 to 1) the color scheme starts at (r, g, b) = (0, 0, 0), i.e. black,
    and finishes at (r, g, b) = (1, 1, 1), i.e. white. For some fraction *x*,
    between 0 and 1, the color is the corresponding grey value at that
    fraction along the black to white diagonal (x, x, x) plus a color
    element. This color element is calculated in a plane of constant
    perceived intensity and controlled by the following parameters.

    Parameters
    ----------
    gamma : float, default: 1
        Gamma factor emphasizing either low intensity values (gamma < 1), or
        high intensity values (gamma > 1).
    s : float, default: 0.5 (purple)
        The starting color.
    r : float, default: -1.5
        The number of r, g, b rotations in color that are made from the start
        to the end of the color scheme.  The default of -1.5 corresponds to ->
        B -> G -> R -> B.
    h : float, default: 1
        The hue, i.e. how saturated the colors are. If this parameter is zero
        then the color scheme is purely a greyscale.
    """

_cubehelix_data: Incomplete
_bwr_data: Incomplete
_brg_data: Incomplete

def _g0(x): ...
def _g1(x): ...
def _g2(x): ...
def _g3(x): ...
def _g4(x): ...
def _g5(x): ...
def _g6(x): ...
def _g7(x): ...
def _g8(x): ...
def _g9(x): ...
def _g10(x): ...
def _g11(x): ...
def _g12(x): ...
def _g13(x): ...
def _g14(x): ...
def _g15(x): ...
def _g16(x): ...
def _g17(x): ...
def _g18(x): ...
def _g19(x): ...
def _g20(x): ...
def _g21(x): ...
def _g22(x): ...
def _g23(x): ...
def _g24(x): ...
def _g25(x): ...
def _g26(x): ...
def _g27(x): ...
def _g28(x): ...
def _g29(x): ...
def _g30(x): ...
def _g31(x): ...
def _g32(x): ...
def _g33(x): ...
def _g34(x): ...
def _g35(x): ...
def _g36(x): ...

gfunc: Incomplete
_gnuplot_data: Incomplete
_gnuplot2_data: Incomplete
_ocean_data: Incomplete
_afmhot_data: Incomplete
_rainbow_data: Incomplete
_seismic_data: Incomplete
_terrain_data: Incomplete
_gray_data: Incomplete
_hot_data: Incomplete
_hsv_data: Incomplete
_jet_data: Incomplete
_pink_data: Incomplete
_spring_data: Incomplete
_summer_data: Incomplete
_winter_data: Incomplete
_nipy_spectral_data: Incomplete
_Blues_data: Incomplete
_BrBG_data: Incomplete
_BuGn_data: Incomplete
_BuPu_data: Incomplete
_GnBu_data: Incomplete
_Greens_data: Incomplete
_Greys_data: Incomplete
_Oranges_data: Incomplete
_OrRd_data: Incomplete
_PiYG_data: Incomplete
_PRGn_data: Incomplete
_PuBu_data: Incomplete
_PuBuGn_data: Incomplete
_PuOr_data: Incomplete
_PuRd_data: Incomplete
_Purples_data: Incomplete
_RdBu_data: Incomplete
_RdGy_data: Incomplete
_RdPu_data: Incomplete
_RdYlBu_data: Incomplete
_RdYlGn_data: Incomplete
_Reds_data: Incomplete
_Spectral_data: Incomplete
_YlGn_data: Incomplete
_YlGnBu_data: Incomplete
_YlOrBr_data: Incomplete
_YlOrRd_data: Incomplete
_Accent_data: Incomplete
_Dark2_data: Incomplete
_Paired_data: Incomplete
_Pastel1_data: Incomplete
_Pastel2_data: Incomplete
_Set1_data: Incomplete
_Set2_data: Incomplete
_Set3_data: Incomplete
_gist_earth_data: Incomplete
_gist_gray_data: Incomplete

def _gist_heat_red(x): ...
def _gist_heat_green(x): ...
def _gist_heat_blue(x): ...

_gist_heat_data: Incomplete
_gist_ncar_data: Incomplete
_gist_rainbow_data: Incomplete
_gist_stern_data: Incomplete

def _gist_yarg(x): ...

_gist_yarg_data: Incomplete
_coolwarm_data: Incomplete
_CMRmap_data: Incomplete
_wistia_data: Incomplete
_tab10_data: Incomplete
_tab20_data: Incomplete
_tab20b_data: Incomplete
_tab20c_data: Incomplete
_petroff10_data: Incomplete
datad: Incomplete

from sympy.core.numbers import I as I, pi as pi
from sympy.functions.elementary.complexes import Abs as Abs, im as im, re as re
from sympy.functions.elementary.exponential import exp as exp
from sympy.functions.elementary.miscellaneous import sqrt as sqrt
from sympy.functions.elementary.trigonometric import cos as cos, sin as sin
from sympy.matrices.dense import Matrix as Matrix
from sympy.physics.quantum import TensorProduct as TensorProduct
from sympy.simplify.simplify import simplify as simplify

def jones_vector(psi, chi):
    """A Jones vector corresponding to a polarization ellipse with `psi` tilt,
    and `chi` circularity.

    Parameters
    ----------
    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the `x` axis.

    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.


    Returns
    -------
    Matrix :
        A Jones vector.

    Examples
    --------
    The axes on the PoincarГ© sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> psi, chi = symbols("psi, chi", real=True)

    A general Jones vector.

    >>> pprint(jones_vector(psi, chi), use_unicode=True)
    вЋЎ-в…€в‹…sin(П‡)в‹…sin(П€) + cos(П‡)в‹…cos(П€)вЋ¤
    вЋў                                вЋҐ
    вЋЈв…€в‹…sin(П‡)в‹…cos(П€) + sin(П€)в‹…cos(П‡) вЋ¦

    Horizontal polarization.

    >>> pprint(jones_vector(0, 0), use_unicode=True)
    вЋЎ1вЋ¤
    вЋў вЋҐ
    вЋЈ0вЋ¦

    Vertical polarization.

    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)
    вЋЎ0вЋ¤
    вЋў вЋҐ
    вЋЈ1вЋ¦

    Diagonal polarization.

    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)
    вЋЎв€љ2вЋ¤
    вЋўв”Ђв”ЂвЋҐ
    вЋў2 вЋҐ
    вЋў  вЋҐ
    вЋўв€љ2вЋҐ
    вЋўв”Ђв”ЂвЋҐ
    вЋЈ2 вЋ¦

    Anti-diagonal polarization.

    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)
    вЋЎ в€љ2 вЋ¤
    вЋў в”Ђв”Ђ вЋҐ
    вЋў 2  вЋҐ
    вЋў    вЋҐ
    вЋў-в€љ2 вЋҐ
    вЋўв”Ђв”Ђв”Ђв”ЂвЋҐ
    вЋЈ 2  вЋ¦

    Right-hand circular polarization.

    >>> pprint(jones_vector(0, pi/4), use_unicode=True)
    вЋЎ в€љ2 вЋ¤
    вЋў в”Ђв”Ђ вЋҐ
    вЋў 2  вЋҐ
    вЋў    вЋҐ
    вЋўв€љ2в‹…в…€вЋҐ
    вЋўв”Ђв”Ђв”Ђв”ЂвЋҐ
    вЋЈ 2  вЋ¦

    Left-hand circular polarization.

    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)
    вЋЎ  в€љ2  вЋ¤
    вЋў  в”Ђв”Ђ  вЋҐ
    вЋў  2   вЋҐ
    вЋў      вЋҐ
    вЋў-в€љ2в‹…в…€ вЋҐ
    вЋўв”Ђв”Ђв”Ђв”Ђв”Ђв”ЂвЋҐ
    вЋЈ  2   вЋ¦

    """
def stokes_vector(psi, chi, p: int = 1, I: int = 1):
    """A Stokes vector corresponding to a polarization ellipse with ``psi``
    tilt, and ``chi`` circularity.

    Parameters
    ----------
    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the ``x`` axis.
    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.
    p : numeric type or SymPy Symbol
        The degree of polarization.
    I : numeric type or SymPy Symbol
        The intensity of the field.


    Returns
    -------
    Matrix :
        A Stokes vector.

    Examples
    --------
    The axes on the PoincarГ© sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import stokes_vector
    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)
    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)
    вЋЎ          I          вЋ¤
    вЋў                     вЋҐ
    вЋўIв‹…pв‹…cos(2в‹…П‡)в‹…cos(2в‹…П€)вЋҐ
    вЋў                     вЋҐ
    вЋўIв‹…pв‹…sin(2в‹…П€)в‹…cos(2в‹…П‡)вЋҐ
    вЋў                     вЋҐ
    вЋЈ    Iв‹…pв‹…sin(2в‹…П‡)     вЋ¦


    Horizontal polarization

    >>> pprint(stokes_vector(0, 0), use_unicode=True)
    вЋЎ1вЋ¤
    вЋў вЋҐ
    вЋў1вЋҐ
    вЋў вЋҐ
    вЋў0вЋҐ
    вЋў вЋҐ
    вЋЈ0вЋ¦

    Vertical polarization

    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)
    вЋЎ1 вЋ¤
    вЋў  вЋҐ
    вЋў-1вЋҐ
    вЋў  вЋҐ
    вЋў0 вЋҐ
    вЋў  вЋҐ
    вЋЈ0 вЋ¦

    Diagonal polarization

    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)
    вЋЎ1вЋ¤
    вЋў вЋҐ
    вЋў0вЋҐ
    вЋў вЋҐ
    вЋў1вЋҐ
    вЋў вЋҐ
    вЋЈ0вЋ¦

    Anti-diagonal polarization

    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)
    вЋЎ1 вЋ¤
    вЋў  вЋҐ
    вЋў0 вЋҐ
    вЋў  вЋҐ
    вЋў-1вЋҐ
    вЋў  вЋҐ
    вЋЈ0 вЋ¦

    Right-hand circular polarization

    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)
    вЋЎ1вЋ¤
    вЋў вЋҐ
    вЋў0вЋҐ
    вЋў вЋҐ
    вЋў0вЋҐ
    вЋў вЋҐ
    вЋЈ1вЋ¦

    Left-hand circular polarization

    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)
    вЋЎ1 вЋ¤
    вЋў  вЋҐ
    вЋў0 вЋҐ
    вЋў  вЋҐ
    вЋў0 вЋҐ
    вЋў  вЋҐ
    вЋЈ-1вЋ¦

    Unpolarized light

    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)
    вЋЎ1вЋ¤
    вЋў вЋҐ
    вЋў0вЋҐ
    вЋў вЋҐ
    вЋў0вЋҐ
    вЋў вЋҐ
    вЋЈ0вЋ¦

    """
def jones_2_stokes(e):
    """Return the Stokes vector for a Jones vector ``e``.

    Parameters
    ----------
    e : SymPy Matrix
        A Jones vector.

    Returns
    -------
    SymPy Matrix
        A Jones vector.

    Examples
    --------
    The axes on the PoincarГ© sphere.

    >>> from sympy import pprint, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> from sympy.physics.optics.polarization import jones_2_stokes
    >>> H = jones_vector(0, 0)
    >>> V = jones_vector(pi/2, 0)
    >>> D = jones_vector(pi/4, 0)
    >>> A = jones_vector(-pi/4, 0)
    >>> R = jones_vector(0, pi/4)
    >>> L = jones_vector(0, -pi/4)
    >>> pprint([jones_2_stokes(e) for e in [H, V, D, A, R, L]],
    ...         use_unicode=True)
    вЋЎвЋЎ1вЋ¤  вЋЎ1 вЋ¤  вЋЎ1вЋ¤  вЋЎ1 вЋ¤  вЋЎ1вЋ¤  вЋЎ1 вЋ¤вЋ¤
    вЋўвЋў вЋҐ  вЋў  вЋҐ  вЋў вЋҐ  вЋў  вЋҐ  вЋў вЋҐ  вЋў  вЋҐвЋҐ
    вЋўвЋў1вЋҐ  вЋў-1вЋҐ  вЋў0вЋҐ  вЋў0 вЋҐ  вЋў0вЋҐ  вЋў0 вЋҐвЋҐ
    вЋўвЋў вЋҐ, вЋў  вЋҐ, вЋў вЋҐ, вЋў  вЋҐ, вЋў вЋҐ, вЋў  вЋҐвЋҐ
    вЋўвЋў0вЋҐ  вЋў0 вЋҐ  вЋў1вЋҐ  вЋў-1вЋҐ  вЋў0вЋҐ  вЋў0 вЋҐвЋҐ
    вЋўвЋў вЋҐ  вЋў  вЋҐ  вЋў вЋҐ  вЋў  вЋҐ  вЋў вЋҐ  вЋў  вЋҐвЋҐ
    вЋЈвЋЈ0вЋ¦  вЋЈ0 вЋ¦  вЋЈ0вЋ¦  вЋЈ0 вЋ¦  вЋЈ1вЋ¦  вЋЈ-1вЋ¦вЋ¦

    """
def linear_polarizer(theta: int = 0):
    """A linear polarizer Jones matrix with transmission axis at
    an angle ``theta``.

    Parameters
    ----------
    theta : numeric type or SymPy Symbol
        The angle of the transmission axis relative to the horizontal plane.

    Returns
    -------
    SymPy Matrix
        A Jones matrix representing the polarizer.

    Examples
    --------
    A generic polarizer.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import linear_polarizer
    >>> theta = symbols("theta", real=True)
    >>> J = linear_polarizer(theta)
    >>> pprint(J, use_unicode=True)
    вЋЎ      2                     вЋ¤
    вЋў   cos (Оё)     sin(Оё)в‹…cos(Оё)вЋҐ
    вЋў                            вЋҐ
    вЋў                     2      вЋҐ
    вЋЈsin(Оё)в‹…cos(Оё)     sin (Оё)   вЋ¦


    """
def phase_retarder(theta: int = 0, delta: int = 0):
    """A phase retarder Jones matrix with retardance ``delta`` at angle ``theta``.

    Parameters
    ----------
    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.
    delta : numeric type or SymPy Symbol
        The phase difference between the fast and slow axes of the
        transmitted light.

    Returns
    -------
    SymPy Matrix :
        A Jones matrix representing the retarder.

    Examples
    --------
    A generic retarder.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import phase_retarder
    >>> theta, delta = symbols("theta, delta", real=True)
    >>> R = phase_retarder(theta, delta)
    >>> pprint(R, use_unicode=True)
    вЋЎ                          -в…€в‹…Оґ               -в…€в‹…Оґ               вЋ¤
    вЋў                          в”Ђв”Ђв”Ђв”Ђв”Ђ              в”Ђв”Ђв”Ђв”Ђв”Ђ              вЋҐ
    вЋўвЋ› в…€в‹…Оґ    2         2   вЋћ    2    вЋ›     в…€в‹…ОґвЋћ    2                вЋҐ
    вЋўвЋќв„Ї   в‹…sin (Оё) + cos (Оё)вЋ в‹…в„Ї       вЋќ1 - в„Ї   вЋ в‹…в„Ї     в‹…sin(Оё)в‹…cos(Оё)вЋҐ
    вЋў                                                                вЋҐ
    вЋў            -в…€в‹…Оґ                                           -в…€в‹…Оґ вЋҐ
    вЋў            в”Ђв”Ђв”Ђв”Ђв”Ђ                                          в”Ђв”Ђв”Ђв”Ђв”ЂвЋҐ
    вЋўвЋ›     в…€в‹…ОґвЋћ    2                  вЋ› в…€в‹…Оґ    2         2   вЋћ    2  вЋҐ
    вЋЈвЋќ1 - в„Ї   вЋ в‹…в„Ї     в‹…sin(Оё)в‹…cos(Оё)  вЋќв„Ї   в‹…cos (Оё) + sin (Оё)вЋ в‹…в„Ї     вЋ¦

    """
def half_wave_retarder(theta):
    """A half-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ----------
    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    -------
    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    --------
    A generic half-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import half_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> HWP = half_wave_retarder(theta)
    >>> pprint(HWP, use_unicode=True)
    вЋЎ   вЋ›     2         2   вЋћ                        вЋ¤
    вЋў-в…€в‹…вЋќ- sin (Оё) + cos (Оё)вЋ     -2в‹…в…€в‹…sin(Оё)в‹…cos(Оё)  вЋҐ
    вЋў                                                вЋҐ
    вЋў                             вЋ›   2         2   вЋћвЋҐ
    вЋЈ   -2в‹…в…€в‹…sin(Оё)в‹…cos(Оё)     -в…€в‹…вЋќsin (Оё) - cos (Оё)вЋ вЋ¦

    """
def quarter_wave_retarder(theta):
    """A quarter-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ----------
    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    -------
    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    --------
    A generic quarter-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import quarter_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> QWP = quarter_wave_retarder(theta)
    >>> pprint(QWP, use_unicode=True)
    вЋЎ                       -в…€в‹…ПЂ            -в…€в‹…ПЂ               вЋ¤
    вЋў                       в”Ђв”Ђв”Ђв”Ђв”Ђ           в”Ђв”Ђв”Ђв”Ђв”Ђ              вЋҐ
    вЋўвЋ›     2         2   вЋћ    4               4                вЋҐ
    вЋўвЋќв…€в‹…sin (Оё) + cos (Оё)вЋ в‹…в„Ї       (1 - в…€)в‹…в„Ї     в‹…sin(Оё)в‹…cos(Оё)вЋҐ
    вЋў                                                          вЋҐ
    вЋў         -в…€в‹…ПЂ                                        -в…€в‹…ПЂ вЋҐ
    вЋў         в”Ђв”Ђв”Ђв”Ђв”Ђ                                       в”Ђв”Ђв”Ђв”Ђв”ЂвЋҐ
    вЋў           4                  вЋ›   2           2   вЋћ    4  вЋҐ
    вЋЈ(1 - в…€)в‹…в„Ї     в‹…sin(Оё)в‹…cos(Оё)  вЋќsin (Оё) + в…€в‹…cos (Оё)вЋ в‹…в„Ї     вЋ¦

    """
def transmissive_filter(T):
    """An attenuator Jones matrix with transmittance ``T``.

    Parameters
    ----------
    T : numeric type or SymPy Symbol
        The transmittance of the attenuator.

    Returns
    -------
    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    --------
    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import transmissive_filter
    >>> T = symbols("T", real=True)
    >>> NDF = transmissive_filter(T)
    >>> pprint(NDF, use_unicode=True)
    вЋЎв€љT  0 вЋ¤
    вЋў      вЋҐ
    вЋЈ0   в€љTвЋ¦

    """
def reflective_filter(R):
    """A reflective filter Jones matrix with reflectance ``R``.

    Parameters
    ----------
    R : numeric type or SymPy Symbol
        The reflectance of the filter.

    Returns
    -------
    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    --------
    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import reflective_filter
    >>> R = symbols("R", real=True)
    >>> pprint(reflective_filter(R), use_unicode=True)
    вЋЎв€љR   0 вЋ¤
    вЋў       вЋҐ
    вЋЈ0   -в€љRвЋ¦

    """
def mueller_matrix(J):
    """The Mueller matrix corresponding to Jones matrix `J`.

    Parameters
    ----------
    J : SymPy Matrix
        A Jones matrix.

    Returns
    -------
    SymPy Matrix
        The corresponding Mueller matrix.

    Examples
    --------
    Generic optical components.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import (mueller_matrix,
    ...     linear_polarizer, half_wave_retarder, quarter_wave_retarder)
    >>> theta = symbols("theta", real=True)

    A linear_polarizer

    >>> pprint(mueller_matrix(linear_polarizer(theta)), use_unicode=True)
    вЋЎ            cos(2в‹…Оё)      sin(2в‹…Оё)     вЋ¤
    вЋў  1/2       в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ      в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ    0вЋҐ
    вЋў               2             2         вЋҐ
    вЋў                                       вЋҐ
    вЋўcos(2в‹…Оё)  cos(4в‹…Оё)   1    sin(4в‹…Оё)     вЋҐ
    вЋўв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ  в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ + в”Ђ    в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ    0вЋҐ
    вЋў   2         4       4       4         вЋҐ
    вЋў                                       вЋҐ
    вЋўsin(2в‹…Оё)    sin(4в‹…Оё)    1   cos(4в‹…Оё)   вЋҐ
    вЋўв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ    в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ    в”Ђ - в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ  0вЋҐ
    вЋў   2           4        4      4       вЋҐ
    вЋў                                       вЋҐ
    вЋЈ   0           0             0        0вЋ¦

    A half-wave plate

    >>> pprint(mueller_matrix(half_wave_retarder(theta)), use_unicode=True)
    вЋЎ1              0                           0               0 вЋ¤
    вЋў                                                             вЋҐ
    вЋў        4           2                                        вЋҐ
    вЋў0  8в‹…sin (Оё) - 8в‹…sin (Оё) + 1           sin(4в‹…Оё)            0 вЋҐ
    вЋў                                                             вЋҐ
    вЋў                                     4           2           вЋҐ
    вЋў0          sin(4в‹…Оё)           - 8в‹…sin (Оё) + 8в‹…sin (Оё) - 1  0 вЋҐ
    вЋў                                                             вЋҐ
    вЋЈ0              0                           0               -1вЋ¦

    A quarter-wave plate

    >>> pprint(mueller_matrix(quarter_wave_retarder(theta)), use_unicode=True)
    вЋЎ1       0             0            0    вЋ¤
    вЋў                                        вЋҐ
    вЋў   cos(4в‹…Оё)   1    sin(4в‹…Оё)             вЋҐ
    вЋў0  в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ + в”Ђ    в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ    -sin(2в‹…Оё)вЋҐ
    вЋў      2       2       2                 вЋҐ
    вЋў                                        вЋҐ
    вЋў     sin(4в‹…Оё)    1   cos(4в‹…Оё)           вЋҐ
    вЋў0    в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ    в”Ђ - в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ  cos(2в‹…Оё) вЋҐ
    вЋў        2        2      2               вЋҐ
    вЋў                                        вЋҐ
    вЋЈ0    sin(2в‹…Оё)     -cos(2в‹…Оё)        0    вЋ¦

    """
def polarizing_beam_splitter(Tp: int = 1, Rs: int = 1, Ts: int = 0, Rp: int = 0, phia: int = 0, phib: int = 0):
    """A polarizing beam splitter Jones matrix at angle `theta`.

    Parameters
    ----------
    J : SymPy Matrix
        A Jones matrix.
    Tp : numeric type or SymPy Symbol
        The transmissivity of the P-polarized component.
    Rs : numeric type or SymPy Symbol
        The reflectivity of the S-polarized component.
    Ts : numeric type or SymPy Symbol
        The transmissivity of the S-polarized component.
    Rp : numeric type or SymPy Symbol
        The reflectivity of the P-polarized component.
    phia : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode a.
    phib : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode b.


    Returns
    -------
    SymPy Matrix
        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector
        whose first two entries are the Jones vector on one of the PBS ports,
        and the last two entries the Jones vector on the other port.

    Examples
    --------
    Generic polarizing beam-splitter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter
    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)
    >>> phia, phib = symbols("phi_a, phi_b", real=True)
    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)
    >>> pprint(PBS, use_unicode=False)
    [   ____                           ____                    ]
    [ \\/ Tp            0           I*\\/ Rp           0         ]
    [                                                          ]
    [                  ____                       ____  I*phi_a]
    [   0            \\/ Ts            0      -I*\\/ Rs *e       ]
    [                                                          ]
    [    ____                         ____                     ]
    [I*\\/ Rp           0            \\/ Tp            0         ]
    [                                                          ]
    [               ____  I*phi_b                    ____      ]
    [   0      -I*\\/ Rs *e            0            \\/ Ts       ]

    """

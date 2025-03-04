from .functions import defun as defun

class RSCache:
    def __init__(ctx) -> None: ...

def _coef(ctx, J, eps):
    """
    Computes the coefficients  `c_n`  for `0\\le n\\le 2J` with error less than eps

    **Definition**

    The coefficients c_n are defined by

    .. math ::

        \\begin{equation}
        F(z)=\\frac{e^{\\pi i
        \\bigl(\\frac{z^2}{2}+\\frac38\\bigr)}-i\\sqrt{2}\\cos\\frac{\\pi}{2}z}{2\\cos\\pi
        z}=\\sum_{n=0}^\\infty c_{2n} z^{2n}
        \\end{equation}

    they are computed applying the relation

    .. math ::

        \\begin{multline}
        c_{2n}=-\\frac{i}{\\sqrt{2}}\\Bigl(\\frac{\\pi}{2}\\Bigr)^{2n}
        \\sum_{k=0}^n\\frac{(-1)^k}{(2k)!}
        2^{2n-2k}\\frac{(-1)^{n-k}E_{2n-2k}}{(2n-2k)!}+\\\\\n        +e^{3\\pi i/8}\\sum_{j=0}^n(-1)^j\\frac{
        E_{2j}}{(2j)!}\\frac{i^{n-j}\\pi^{n+j}}{(n-j)!2^{n-j+1}}.
        \\end{multline}
    """
def coef(ctx, J, eps): ...
def aux_M_Fp(ctx, xA, xeps4, a, xB1, xL): ...
def aux_J_needed(ctx, xA, xeps4, a, xB1, xM): ...
def Rzeta_simul(ctx, s, der: int = 0): ...
def Rzeta_set(ctx, s, derivatives=[0]):
    """
    Computes several derivatives of the auxiliary function of Riemann `R(s)`.

    **Definition**

    The function is defined by

    .. math ::

        \\begin{equation}
        {\\mathop{\\mathcal R }\\nolimits}(s)=
        \\int_{0\\swarrow1}\\frac{x^{-s} e^{\\pi i x^2}}{e^{\\pi i x}-
        e^{-\\pi i x}}\\,dx
        \\end{equation}

    To this function we apply the Riemann-Siegel expansion.
    """
def z_half(ctx, t, der: int = 0):
    """
    z_half(t,der=0) Computes Z^(der)(t)
    """
def zeta_half(ctx, s, k: int = 0):
    """
    zeta_half(s,k=0) Computes zeta^(k)(s) when Re s = 0.5
    """
def zeta_offline(ctx, s, k: int = 0):
    """
    Computes zeta^(k)(s) off the line
    """
def z_offline(ctx, w, k: int = 0):
    """
    Computes Z(w) and its derivatives off the line
    """
def rs_zeta(ctx, s, derivative: int = 0, **kwargs): ...
def rs_z(ctx, w, derivative: int = 0): ...

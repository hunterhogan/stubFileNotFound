from sympy.core import Add as Add, Basic as Basic, Mul as Mul

def sub_pre(e):
    """ Replace y - x with -(x - y) if -1 can be extracted from y - x.
    """
def sub_post(e):
    """ Replace 1*-1*x with -x.
    """

from _typeshed import Incomplete

_message_dict_support: Incomplete

def dict_constructor(context, builder, sig, args): ...
def impl_dict(context, builder, sig, args):
    """
    The `dict()` implementation simply forwards the work to `Dict.empty()`.
    """

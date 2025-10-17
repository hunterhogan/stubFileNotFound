from _typeshed import Incomplete
from python_toolbox import caching as caching, context_management as context_management, misc_tools as misc_tools

class Freezer(context_management.DelegatingContextManager):
    r"""
    A freezer is used as a context manager to "freeze" and "thaw" an object.

    Different kinds of objects have different concepts of "freezing" and
    "thawing": A GUI widget could be graphically frozen, preventing the OS from
    drawing any changes to it, and then when its thawed have all the changes
    drawn at once. As another example, an ORM could be frozen to have it not
    write to the database while a suite it being executed, and then have it
    write all the data at once when thawed.

    This class only implements the abstract behavior of a freezer: It is a
    reentrant context manager which has handlers for freezing and thawing, and
    its level of frozenness can be checked by accessing the attribute
    `.frozen`. It\'s up to subclasses to override `freeze_handler` and
    `thaw_handler` to do whatever they should do on freeze and thaw. Note that
    you can override either of these methods to be a no-op, sometimes even both
    methods, and still have a useful freezer by checking the property `.frozen`
    in the logic of the parent object.
    """

    delegatee_context_manager: Incomplete
    frozen: Incomplete
    def freeze_handler(self) -> None:
        """Do something when the object gets frozen."""
    def thaw_handler(self) -> None:
        """Do something when the object gets thawed."""




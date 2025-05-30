from _typeshed import Incomplete

class Gcf:
    '''
    Singleton to maintain the relation between figures and their managers, and
    keep track of and "active" figure and manager.

    The canvas of a figure created through pyplot is associated with a figure
    manager, which handles the interaction between the figure and the backend.
    pyplot keeps track of figure managers using an identifier, the "figure
    number" or "manager number" (which can actually be any hashable value);
    this number is available as the :attr:`number` attribute of the manager.

    This class is never instantiated; it consists of an `OrderedDict` mapping
    figure/manager numbers to managers, and a set of class methods that
    manipulate this `OrderedDict`.

    Attributes
    ----------
    figs : OrderedDict
        `OrderedDict` mapping numbers to managers; the active manager is at the
        end.
    '''
    figs: Incomplete
    @classmethod
    def get_fig_manager(cls, num):
        """
        If manager number *num* exists, make it the active one and return it;
        otherwise return *None*.
        """
    @classmethod
    def destroy(cls, num) -> None:
        '''
        Destroy manager *num* -- either a manager instance or a manager number.

        In the interactive backends, this is bound to the window "destroy" and
        "delete" events.

        It is recommended to pass a manager instance, to avoid confusion when
        two managers share the same number.
        '''
    @classmethod
    def destroy_fig(cls, fig) -> None:
        """Destroy figure *fig*."""
    @classmethod
    def destroy_all(cls) -> None:
        """Destroy all figures."""
    @classmethod
    def has_fignum(cls, num):
        """Return whether figure number *num* exists."""
    @classmethod
    def get_all_fig_managers(cls):
        """Return a list of figure managers."""
    @classmethod
    def get_num_fig_managers(cls):
        """Return the number of figures being managed."""
    @classmethod
    def get_active(cls):
        """Return the active manager, or *None* if there is no manager."""
    @classmethod
    def _set_new_active_manager(cls, manager):
        """Adopt *manager* into pyplot and make it the active manager."""
    @classmethod
    def set_active(cls, manager) -> None:
        """Make *manager* the active manager."""
    @classmethod
    def draw_all(cls, force: bool = False) -> None:
        """
        Redraw all stale managed figures, or, if *force* is True, all managed
        figures.
        """

from _typeshed import Incomplete

__all__ = ['_dispatchable']

class _dispatchable:
    _is_testing: bool
    class _fallback_to_nx:
        """Class property that returns ``nx.config.fallback_to_nx``."""
        def __get__(self, instance, owner: Incomplete | None = None): ...
    __name__: Incomplete
    __defaults__: Incomplete
    __kwdefaults__: Incomplete
    __module__: Incomplete
    __qualname__: Incomplete
    __wrapped__: Incomplete
    _orig_doc: Incomplete
    _cached_doc: Incomplete
    orig_func: Incomplete
    name: Incomplete
    edge_attrs: Incomplete
    node_attrs: Incomplete
    preserve_edge_attrs: Incomplete
    preserve_node_attrs: Incomplete
    preserve_graph_attrs: Incomplete
    mutates_input: Incomplete
    _returns_graph: Incomplete
    optional_graphs: Incomplete
    list_graphs: Incomplete
    graphs: Incomplete
    _sig: Incomplete
    backends: Incomplete
    def __new__(cls, func: Incomplete | None = None, *, name: Incomplete | None = None, graphs: str = 'G', edge_attrs: Incomplete | None = None, node_attrs: Incomplete | None = None, preserve_edge_attrs: bool = False, preserve_node_attrs: bool = False, preserve_graph_attrs: bool = False, preserve_all_attrs: bool = False, mutates_input: bool = False, returns_graph: bool = False):
        '''A decorator function that is used to redirect the execution of ``func``
        function to its backend implementation.

        This decorator function dispatches to
        a different backend implementation based on the input graph types, and it also
        manages all the ``backend_kwargs``. Usage can be any of the following decorator
        forms:

        - ``@_dispatchable``
        - ``@_dispatchable()``
        - ``@_dispatchable(name="override_name")``
        - ``@_dispatchable(graphs="graph_var_name")``
        - ``@_dispatchable(edge_attrs="weight")``
        - ``@_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})``
            with 0 and 1 giving the position in the signature function for graph
            objects. When ``edge_attrs`` is a dict, keys are keyword names and values
            are defaults.

        Parameters
        ----------
        func : callable, optional
            The function to be decorated. If ``func`` is not provided, returns a
            partial object that can be used to decorate a function later. If ``func``
            is provided, returns a new callable object that dispatches to a backend
            algorithm based on input graph types.

        name : str, optional
            The name of the algorithm to use for dispatching. If not provided,
            the name of ``func`` will be used. ``name`` is useful to avoid name
            conflicts, as all dispatched algorithms live in a single namespace.
            For example, ``tournament.is_strongly_connected`` had a name conflict
            with the standard ``nx.is_strongly_connected``, so we used
            ``@_dispatchable(name="tournament_is_strongly_connected")``.

        graphs : str or dict or None, default "G"
            If a string, the parameter name of the graph, which must be the first
            argument of the wrapped function. If more than one graph is required
            for the algorithm (or if the graph is not the first argument), provide
            a dict keyed to argument names with argument position as values for each
            graph argument. For example, ``@_dispatchable(graphs={"G": 0, "auxiliary?": 4})``
            indicates the 0th parameter ``G`` of the function is a required graph,
            and the 4th parameter ``auxiliary?`` is an optional graph.
            To indicate that an argument is a list of graphs, do ``"[graphs]"``.
            Use ``graphs=None``, if *no* arguments are NetworkX graphs such as for
            graph generators, readers, and conversion functions.

        edge_attrs : str or dict, optional
            ``edge_attrs`` holds information about edge attribute arguments
            and default values for those edge attributes.
            If a string, ``edge_attrs`` holds the function argument name that
            indicates a single edge attribute to include in the converted graph.
            The default value for this attribute is 1. To indicate that an argument
            is a list of attributes (all with default value 1), use e.g. ``"[attrs]"``.
            If a dict, ``edge_attrs`` holds a dict keyed by argument names, with
            values that are either the default value or, if a string, the argument
            name that indicates the default value.

        node_attrs : str or dict, optional
            Like ``edge_attrs``, but for node attributes.

        preserve_edge_attrs : bool or str or dict, optional
            For bool, whether to preserve all edge attributes.
            For str, the parameter name that may indicate (with ``True`` or a
            callable argument) whether all edge attributes should be preserved
            when converting.
            For dict of ``{graph_name: {attr: default}}``, indicate pre-determined
            edge attributes (and defaults) to preserve for input graphs.

        preserve_node_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for node attributes.

        preserve_graph_attrs : bool or set
            For bool, whether to preserve all graph attributes.
            For set, which input graph arguments to preserve graph attributes.

        preserve_all_attrs : bool
            Whether to preserve all edge, node and graph attributes.
            This overrides all the other preserve_*_attrs.

        mutates_input : bool or dict, default False
            For bool, whether the function mutates an input graph argument.
            For dict of ``{arg_name: arg_pos}``, arguments that indicate whether an
            input graph will be mutated, and ``arg_name`` may begin with ``"not "``
            to negate the logic (for example, this is used by ``copy=`` arguments).
            By default, dispatching doesn\'t convert input graphs to a different
            backend for functions that mutate input graphs.

        returns_graph : bool, default False
            Whether the function can return or yield a graph object. By default,
            dispatching doesn\'t convert input graphs to a different backend for
            functions that return graphs.
        '''
    @property
    def __doc__(self):
        """If the cached documentation exists, it is returned.
        Otherwise, the documentation is generated using _make_doc() method,
        cached, and then returned."""
    @__doc__.setter
    def __doc__(self, val) -> None:
        """Sets the original documentation to the given value and resets the
        cached documentation."""
    @property
    def __signature__(self):
        """Return the signature of the original function, with the addition of
        the `backend` and `backend_kwargs` parameters."""
    def __call__(self, /, *args, backend: Incomplete | None = None, **kwargs):
        """Returns the result of the original function, or the backend function if
        the backend is specified and that backend implements `func`."""
    def _will_call_mutate_input(self, args, kwargs): ...
    def _can_convert(self, backend_name, graph_backend_names): ...
    def _does_backend_have(self, backend_name):
        """Does the specified backend have this algorithm?"""
    def _can_backend_run(self, backend_name, args, kwargs):
        """Can the specified backend run this algorithm with these arguments?"""
    def _should_backend_run(self, backend_name, args, kwargs):
        """Should the specified backend run this algorithm with these arguments?

        Note that this does not check ``backend.can_run``.
        """
    def _convert_arguments(self, backend_name, args, kwargs, *, use_cache, mutations):
        """Convert graph arguments to the specified backend.

        Returns
        -------
        args tuple and kwargs dict
        """
    def _convert_graph(self, backend_name, graph, *, edge_attrs, node_attrs, preserve_edge_attrs, preserve_node_attrs, preserve_graph_attrs, graph_name, use_cache, mutations): ...
    def _call_with_backend(self, backend_name, args, kwargs, *, extra_message: Incomplete | None = None):
        """Call this dispatchable function with a backend without converting inputs."""
    def _convert_and_call(self, backend_name, input_backend_names, args, kwargs, *, extra_message: Incomplete | None = None, mutations: Incomplete | None = None):
        """Call this dispatchable function with a backend after converting inputs.

        Parameters
        ----------
        backend_name : str
        input_backend_names : set[str]
        args : arguments tuple
        kwargs : keywords dict
        extra_message : str, optional
            Additional message to log if NotImplementedError is raised by backend.
        mutations : list, optional
            Used to clear objects gotten from cache if inputs will be mutated.
        """
    def _convert_and_call_for_tests(self, backend_name, args, kwargs, *, fallback_to_nx: bool = False):
        """Call this dispatchable function with a backend; for use with testing."""
    def _make_doc(self):
        """Generate the backends section at the end for functions having an alternate
        backend implementation(s) using the `backend_info` entry-point."""
    def __reduce__(self):
        """Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        """

class _LazyArgsRepr:
    """Simple wrapper to display arguments of dispatchable functions in logging calls."""
    func: Incomplete
    args: Incomplete
    kwargs: Incomplete
    value: Incomplete
    def __init__(self, func, args, kwargs) -> None: ...
    def __repr__(self) -> str: ...
_orig_dispatchable = _dispatchable

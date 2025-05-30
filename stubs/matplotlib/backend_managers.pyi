from _typeshed import Incomplete
from matplotlib import _api as _api, backend_tools as backend_tools, cbook as cbook, widgets as widgets

class ToolEvent:
    """Event for tool manipulation (add/remove)."""
    name: Incomplete
    sender: Incomplete
    tool: Incomplete
    data: Incomplete
    def __init__(self, name, sender, tool, data: Incomplete | None = None) -> None: ...

class ToolTriggerEvent(ToolEvent):
    """Event to inform that a tool has been triggered."""
    canvasevent: Incomplete
    def __init__(self, name, sender, tool, canvasevent: Incomplete | None = None, data: Incomplete | None = None) -> None: ...

class ToolManagerMessageEvent:
    """
    Event carrying messages from toolmanager.

    Messages usually get displayed to the user by the toolbar.
    """
    name: Incomplete
    sender: Incomplete
    message: Incomplete
    def __init__(self, name, sender, message) -> None: ...

class ToolManager:
    """
    Manager for actions triggered by user interactions (key press, toolbar
    clicks, ...) on a Figure.

    Attributes
    ----------
    figure : `.Figure`
    keypresslock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the `canvas` key_press_event is locked.
    messagelock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the message is available to write.
    """
    _key_press_handler_id: Incomplete
    _tools: Incomplete
    _keys: Incomplete
    _toggled: Incomplete
    _callbacks: Incomplete
    keypresslock: Incomplete
    messagelock: Incomplete
    _figure: Incomplete
    def __init__(self, figure: Incomplete | None = None) -> None: ...
    @property
    def canvas(self):
        """Canvas managed by FigureManager."""
    @property
    def figure(self):
        """Figure that holds the canvas."""
    @figure.setter
    def figure(self, figure) -> None: ...
    def set_figure(self, figure, update_tools: bool = True) -> None:
        """
        Bind the given figure to the tools.

        Parameters
        ----------
        figure : `.Figure`
        update_tools : bool, default: True
            Force tools to update figure.
        """
    def toolmanager_connect(self, s, func):
        """
        Connect event with string *s* to *func*.

        Parameters
        ----------
        s : str
            The name of the event. The following events are recognized:

            - 'tool_message_event'
            - 'tool_removed_event'
            - 'tool_added_event'

            For every tool added a new event is created

            - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.

        func : callable
            Callback function for the toolmanager event with signature::

                def func(event: ToolEvent) -> Any

        Returns
        -------
        cid
            The callback id for the connection. This can be used in
            `.toolmanager_disconnect`.
        """
    def toolmanager_disconnect(self, cid):
        """
        Disconnect callback id *cid*.

        Example usage::

            cid = toolmanager.toolmanager_connect('tool_trigger_zoom', onpress)
            #...later
            toolmanager.toolmanager_disconnect(cid)
        """
    def message_event(self, message, sender: Incomplete | None = None) -> None:
        """Emit a `ToolManagerMessageEvent`."""
    @property
    def active_toggle(self):
        """Currently toggled tools."""
    def get_tool_keymap(self, name):
        """
        Return the keymap associated with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.

        Returns
        -------
        list of str
            List of keys associated with the tool.
        """
    def _remove_keys(self, name) -> None: ...
    def update_keymap(self, name, key) -> None:
        """
        Set the keymap to associate with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.
        key : str or list of str
            Keys to associate with the tool.
        """
    def remove_tool(self, name) -> None:
        """
        Remove tool named *name*.

        Parameters
        ----------
        name : str
            Name of the tool.
        """
    def add_tool(self, name, tool, *args, **kwargs):
        """
        Add *tool* to `ToolManager`.

        If successful, adds a new event ``tool_trigger_{name}`` where
        ``{name}`` is the *name* of the tool; the event is fired every time the
        tool is triggered.

        Parameters
        ----------
        name : str
            Name of the tool, treated as the ID, has to be unique.
        tool : type
            Class of the tool to be added.  A subclass will be used
            instead if one was registered for the current canvas class.
        *args, **kwargs
            Passed to the *tool*'s constructor.

        See Also
        --------
        matplotlib.backend_tools.ToolBase : The base class for tools.
        """
    def _handle_toggle(self, tool, canvasevent, data) -> None:
        """
        Toggle tools, need to untoggle prior to using other Toggle tool.
        Called from trigger_tool.

        Parameters
        ----------
        tool : `.ToolBase`
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """
    def trigger_tool(self, name, sender: Incomplete | None = None, canvasevent: Incomplete | None = None, data: Incomplete | None = None) -> None:
        """
        Trigger a tool and emit the ``tool_trigger_{name}`` event.

        Parameters
        ----------
        name : str
            Name of the tool.
        sender : object
            Object that wishes to trigger the tool.
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """
    def _key_press(self, event) -> None: ...
    @property
    def tools(self):
        """A dict mapping tool name -> controlled tool."""
    def get_tool(self, name, warn: bool = True):
        """
        Return the tool object with the given name.

        For convenience, this passes tool objects through.

        Parameters
        ----------
        name : str or `.ToolBase`
            Name of the tool, or the tool itself.
        warn : bool, default: True
            Whether a warning should be emitted it no tool with the given name
            exists.

        Returns
        -------
        `.ToolBase` or None
            The tool or None if no tool with the given name exists.
        """

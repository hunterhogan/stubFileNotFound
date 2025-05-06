import enum
from _typeshed import Incomplete
from matplotlib import _api as _api, cbook as cbook

class Cursors(enum.IntEnum):
    """Backend-independent cursor types."""
    POINTER = ...
    HAND = ...
    SELECT_REGION = ...
    MOVE = ...
    WAIT = ...
    RESIZE_HORIZONTAL = ...
    RESIZE_VERTICAL = ...
cursors = Cursors
_tool_registry: Incomplete

def _register_tool_class(canvas_cls, tool_cls: Incomplete | None = None):
    """Decorator registering *tool_cls* as a tool class for *canvas_cls*."""
def _find_tool_class(canvas_cls, tool_cls):
    """Find a subclass of *tool_cls* registered for *canvas_cls*."""

_views_positions: str

class ToolBase:
    """
    Base tool class.

    A base tool, only implements `trigger` method or no method at all.
    The tool is instantiated by `matplotlib.backend_managers.ToolManager`.
    """
    default_keymap: Incomplete
    description: Incomplete
    image: Incomplete
    _name: Incomplete
    _toolmanager: Incomplete
    _figure: Incomplete
    def __init__(self, toolmanager, name) -> None: ...
    name: Incomplete
    toolmanager: Incomplete
    canvas: Incomplete
    def set_figure(self, figure) -> None: ...
    figure: Incomplete
    def _make_classic_style_pseudo_toolbar(self):
        """
        Return a placeholder object with a single `canvas` attribute.

        This is useful to reuse the implementations of tools already provided
        by the classic Toolbars.
        """
    def trigger(self, sender, event, data: Incomplete | None = None) -> None:
        """
        Called when this tool gets used.

        This method is called by `.ToolManager.trigger_tool`.

        Parameters
        ----------
        event : `.Event`
            The canvas event that caused this tool to be called.
        sender : object
            Object that requested the tool to be triggered.
        data : object
            Extra data.
        """

class ToolToggleBase(ToolBase):
    """
    Toggleable tool.

    Every time it is triggered, it switches between enable and disable.

    Parameters
    ----------
    ``*args``
        Variable length argument to be used by the Tool.
    ``**kwargs``
        `toggled` if present and True, sets the initial state of the Tool
        Arbitrary keyword arguments to be consumed by the Tool
    """
    radio_group: Incomplete
    cursor: Incomplete
    default_toggled: bool
    _toggled: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def trigger(self, sender, event, data: Incomplete | None = None) -> None:
        """Calls `enable` or `disable` based on `toggled` value."""
    def enable(self, event: Incomplete | None = None) -> None:
        """
        Enable the toggle tool.

        `trigger` calls this method when `toggled` is False.
        """
    def disable(self, event: Incomplete | None = None) -> None:
        """
        Disable the toggle tool.

        `trigger` call this method when `toggled` is True.

        This can happen in different circumstances.

        * Click on the toolbar tool button.
        * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`.
        * Another `ToolToggleBase` derived tool is triggered
          (from the same `.ToolManager`).
        """
    @property
    def toggled(self):
        """State of the toggled tool."""
    def set_figure(self, figure) -> None: ...

class ToolSetCursor(ToolBase):
    """
    Change to the current cursor while inaxes.

    This tool, keeps track of all `ToolToggleBase` derived tools, and updates
    the cursor when a tool gets triggered.
    """
    _id_drag: Incomplete
    _current_tool: Incomplete
    _default_cursor: Incomplete
    _last_cursor: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_figure(self, figure) -> None: ...
    def _add_tool_cbk(self, event) -> None:
        """Process every newly added tool."""
    def _tool_trigger_cbk(self, event) -> None: ...
    def _set_cursor_cbk(self, event) -> None: ...

class ToolCursorPosition(ToolBase):
    """
    Send message with the current pointer position.

    This tool runs in the background reporting the position of the cursor.
    """
    _id_drag: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_figure(self, figure) -> None: ...
    def send_message(self, event) -> None:
        """Call `matplotlib.backend_managers.ToolManager.message_event`."""

class RubberbandBase(ToolBase):
    """Draw and remove a rubberband."""
    def trigger(self, sender, event, data: Incomplete | None = None) -> None:
        """Call `draw_rubberband` or `remove_rubberband` based on data."""
    def draw_rubberband(self, *data) -> None:
        """
        Draw rubberband.

        This method must get implemented per backend.
        """
    def remove_rubberband(self) -> None:
        """
        Remove rubberband.

        This method should get implemented per backend.
        """

class ToolQuit(ToolBase):
    """Tool to call the figure manager destroy method."""
    description: str
    default_keymap: Incomplete
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...

class ToolQuitAll(ToolBase):
    """Tool to call the figure manager destroy method."""
    description: str
    default_keymap: Incomplete
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...

class ToolGrid(ToolBase):
    """Tool to toggle the major grids of the figure."""
    description: str
    default_keymap: Incomplete
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...

class ToolMinorGrid(ToolBase):
    """Tool to toggle the major and minor grids of the figure."""
    description: str
    default_keymap: Incomplete
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...

class ToolFullScreen(ToolBase):
    """Tool to toggle full screen."""
    description: str
    default_keymap: Incomplete
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...

class AxisScaleBase(ToolToggleBase):
    """Base Tool to toggle between linear and logarithmic."""
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...
    def enable(self, event: Incomplete | None = None) -> None: ...
    def disable(self, event: Incomplete | None = None) -> None: ...

class ToolYScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the Y axis."""
    description: str
    default_keymap: Incomplete
    def set_scale(self, ax, scale) -> None: ...

class ToolXScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the X axis."""
    description: str
    default_keymap: Incomplete
    def set_scale(self, ax, scale) -> None: ...

class ToolViewsPositions(ToolBase):
    """
    Auxiliary Tool to handle changes in views and positions.

    Runs in the background and should get used by all the tools that
    need to access the figure's history of views and positions, e.g.

    * `ToolZoom`
    * `ToolPan`
    * `ToolHome`
    * `ToolBack`
    * `ToolForward`
    """
    views: Incomplete
    positions: Incomplete
    home_views: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def add_figure(self, figure):
        """Add the current figure to the stack of views and positions."""
    def clear(self, figure) -> None:
        """Reset the Axes stack."""
    def update_view(self) -> None:
        """
        Update the view limits and position for each Axes from the current
        stack position. If any Axes are present in the figure that aren't in
        the current stack position, use the home view limits for those Axes and
        don't update *any* positions.
        """
    def push_current(self, figure: Incomplete | None = None) -> None:
        """
        Push the current view limits and position onto their respective stacks.
        """
    def _axes_pos(self, ax):
        """
        Return the original and modified positions for the specified Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The `.Axes` to get the positions for.

        Returns
        -------
        original_position, modified_position
            A tuple of the original and modified positions.
        """
    def update_home_views(self, figure: Incomplete | None = None) -> None:
        """
        Make sure that ``self.home_views`` has an entry for all Axes present
        in the figure.
        """
    def home(self) -> None:
        """Recall the first view and position from the stack."""
    def back(self) -> None:
        """Back one step in the stack of views and positions."""
    def forward(self) -> None:
        """Forward one step in the stack of views and positions."""

class ViewsPositionsBase(ToolBase):
    """Base class for `ToolHome`, `ToolBack` and `ToolForward`."""
    _on_trigger: Incomplete
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...

class ToolHome(ViewsPositionsBase):
    """Restore the original view limits."""
    description: str
    image: str
    default_keymap: Incomplete
    _on_trigger: str

class ToolBack(ViewsPositionsBase):
    """Move back up the view limits stack."""
    description: str
    image: str
    default_keymap: Incomplete
    _on_trigger: str

class ToolForward(ViewsPositionsBase):
    """Move forward in the view lim stack."""
    description: str
    image: str
    default_keymap: Incomplete
    _on_trigger: str

class ConfigureSubplotsBase(ToolBase):
    """Base tool for the configuration of subplots."""
    description: str
    image: str

class SaveFigureBase(ToolBase):
    """Base tool for figure saving."""
    description: str
    image: str
    default_keymap: Incomplete

class ZoomPanBase(ToolToggleBase):
    """Base class for `ToolZoom` and `ToolPan`."""
    _button_pressed: Incomplete
    _xypress: Incomplete
    _idPress: Incomplete
    _idRelease: Incomplete
    _idScroll: Incomplete
    base_scale: float
    scrollthresh: float
    lastscroll: Incomplete
    def __init__(self, *args) -> None: ...
    def enable(self, event: Incomplete | None = None) -> None:
        """Connect press/release events and lock the canvas."""
    def disable(self, event: Incomplete | None = None) -> None:
        """Release the canvas and disconnect press/release events."""
    def trigger(self, sender, event, data: Incomplete | None = None) -> None: ...
    def scroll_zoom(self, event) -> None: ...

class ToolZoom(ZoomPanBase):
    """A Tool for zooming using a rectangle selector."""
    description: str
    image: str
    default_keymap: Incomplete
    cursor: Incomplete
    radio_group: str
    _ids_zoom: Incomplete
    def __init__(self, *args) -> None: ...
    _xypress: Incomplete
    _button_pressed: Incomplete
    def _cancel_action(self) -> None: ...
    _zoom_mode: Incomplete
    def _press(self, event) -> None:
        """Callback for mouse button presses in zoom-to-rectangle mode."""
    def _switch_on_zoom_mode(self, event) -> None: ...
    def _switch_off_zoom_mode(self, event) -> None: ...
    def _mouse_move(self, event) -> None:
        """Callback for mouse moves in zoom-to-rectangle mode."""
    def _release(self, event) -> None:
        """Callback for mouse button releases in zoom-to-rectangle mode."""

class ToolPan(ZoomPanBase):
    """Pan Axes with left mouse, zoom with right."""
    default_keymap: Incomplete
    description: str
    image: str
    cursor: Incomplete
    radio_group: str
    _id_drag: Incomplete
    def __init__(self, *args) -> None: ...
    _button_pressed: Incomplete
    _xypress: Incomplete
    def _cancel_action(self) -> None: ...
    def _press(self, event) -> None: ...
    def _release(self, event) -> None: ...
    def _mouse_move(self, event) -> None: ...

class ToolHelpBase(ToolBase):
    description: str
    default_keymap: Incomplete
    image: str
    @staticmethod
    def format_shortcut(key_sequence):
        """
        Convert a shortcut string from the notation used in rc config to the
        standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.
        """
    def _format_tool_keymap(self, name): ...
    def _get_help_entries(self): ...
    def _get_help_text(self): ...
    def _get_help_html(self): ...

class ToolCopyToClipboardBase(ToolBase):
    """Tool to copy the figure to the clipboard."""
    description: str
    default_keymap: Incomplete
    def trigger(self, *args, **kwargs) -> None: ...

default_tools: Incomplete
default_toolbar_tools: Incomplete

def add_tools_to_manager(toolmanager, tools=...) -> None:
    """
    Add multiple tools to a `.ToolManager`.

    Parameters
    ----------
    toolmanager : `.backend_managers.ToolManager`
        Manager to which the tools are added.
    tools : {str: class_like}, optional
        The tools to add in a {name: tool} dict, see
        `.backend_managers.ToolManager.add_tool` for more info.
    """
def add_tools_to_container(container, tools=...) -> None:
    """
    Add multiple tools to the container.

    Parameters
    ----------
    container : Container
        `.backend_bases.ToolContainerBase` object that will get the tools
        added.
    tools : list, optional
        List in the form ``[[group1, [tool1, tool2 ...]], [group2, [...]]]``
        where the tools ``[tool1, tool2, ...]`` will display in group1.
        See `.backend_bases.ToolContainerBase.add_tool` for details.
    """

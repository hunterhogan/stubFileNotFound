from . import _api as _api, _docstring as _docstring, backend_tools as backend_tools, cbook as cbook, collections as collections, colors as colors, ticker as ticker, transforms as transforms
from .patches import Ellipse as Ellipse, Polygon as Polygon, Rectangle as Rectangle
from .transforms import Affine2D as Affine2D, TransformedPatchPath as TransformedPatchPath
from _typeshed import Incomplete

class LockDraw:
    """
    Some widgets, like the cursor, draw onto the canvas, and this is not
    desirable under all circumstances, like when the toolbar is in zoom-to-rect
    mode and drawing a rectangle.  To avoid this, a widget can acquire a
    canvas' lock with ``canvas.widgetlock(widget)`` before drawing on the
    canvas; this will prevent other widgets from doing so at the same time (if
    they also try to acquire the lock first).
    """
    _owner: Incomplete
    def __init__(self) -> None: ...
    def __call__(self, o) -> None:
        """Reserve the lock for *o*."""
    def release(self, o) -> None:
        """Release the lock from *o*."""
    def available(self, o):
        """Return whether drawing is available to *o*."""
    def isowner(self, o):
        """Return whether *o* owns this lock."""
    def locked(self):
        """Return whether the lock is currently held by an owner."""

class Widget:
    """
    Abstract base class for GUI neutral widgets.
    """
    drawon: bool
    eventson: bool
    _active: bool
    def set_active(self, active) -> None:
        """Set whether the widget is active."""
    def get_active(self):
        """Get whether the widget is active."""
    active: Incomplete
    def ignore(self, event):
        """
        Return whether *event* should be ignored.

        This method should be called at the beginning of any event callback.
        """

class AxesWidget(Widget):
    """
    Widget connected to a single `~matplotlib.axes.Axes`.

    To guarantee that the widget remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the widget.  If there are no references to the widget
    object it may be garbage collected which will disconnect the callbacks.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    canvas : `~matplotlib.backend_bases.FigureCanvasBase`
        The parent figure canvas for the widget.
    active : bool
        If False, the widget does not respond to events.
    """
    ax: Incomplete
    _cids: Incomplete
    def __init__(self, ax) -> None: ...
    canvas: Incomplete
    def connect_event(self, event, callback) -> None:
        """
        Connect a callback function with an event.

        This should be used in lieu of ``figure.canvas.mpl_connect`` since this
        function stores callback ids for later clean up.
        """
    def disconnect_events(self) -> None:
        """Disconnect all events created by this widget."""
    def _get_data_coords(self, event):
        """Return *event*'s data coordinates in this widget's Axes."""

class Button(AxesWidget):
    """
    A GUI neutral button.

    For the button to remain responsive you must keep a reference to it.
    Call `.on_clicked` to connect to the button.

    Attributes
    ----------
    ax
        The `~.axes.Axes` the button renders into.
    label
        A `.Text` instance.
    color
        The color of the button when not hovering.
    hovercolor
        The color of the button when hovering.
    """
    label: Incomplete
    _useblit: Incomplete
    _observers: Incomplete
    color: Incomplete
    hovercolor: Incomplete
    def __init__(self, ax, label, image: Incomplete | None = None, color: str = '0.85', hovercolor: str = '0.95', *, useblit: bool = True) -> None:
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            The button text.
        image : array-like or PIL Image
            The image to place in the button, if not *None*.  The parameter is
            directly forwarded to `~.axes.Axes.imshow`.
        color : :mpltype:`color`
            The color of the button when not activated.
        hovercolor : :mpltype:`color`
            The color of the button when the mouse is over it.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :ref:`blitting` for details.

            .. versionadded:: 3.7
        """
    def _click(self, event) -> None: ...
    def _release(self, event) -> None: ...
    def _motion(self, event) -> None: ...
    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Returns a connection id, which can be used to disconnect the callback.
        """
    def disconnect(self, cid) -> None:
        """Remove the callback function with connection id *cid*."""

class SliderBase(AxesWidget):
    """
    The base class for constructing Slider widgets. Not intended for direct
    usage.

    For the slider to remain responsive you must maintain a reference to it.
    """
    orientation: Incomplete
    closedmin: Incomplete
    closedmax: Incomplete
    valmin: Incomplete
    valmax: Incomplete
    valstep: Incomplete
    drag_active: bool
    valfmt: Incomplete
    _fmt: Incomplete
    _observers: Incomplete
    def __init__(self, ax, orientation, closedmin, closedmax, valmin, valmax, valfmt, dragging, valstep) -> None: ...
    def _stepped_value(self, val):
        """Return *val* coerced to closest number in the ``valstep`` grid."""
    def disconnect(self, cid) -> None:
        """
        Remove the observer with connection id *cid*.

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed.
        """
    def reset(self) -> None:
        """Reset the slider to the initial value."""

class Slider(SliderBase):
    """
    A slider representing a floating point range.

    Create a slider from *valmin* to *valmax* in Axes *ax*. For the slider to
    remain responsive you must maintain a reference to it. Call
    :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : float
        Slider value.
    """
    slidermin: Incomplete
    slidermax: Incomplete
    val: Incomplete
    valinit: Incomplete
    track: Incomplete
    poly: Incomplete
    hline: Incomplete
    vline: Incomplete
    label: Incomplete
    valtext: Incomplete
    def __init__(self, ax, label, valmin, valmax, *, valinit: float = 0.5, valfmt: Incomplete | None = None, closedmin: bool = True, closedmax: bool = True, slidermin: Incomplete | None = None, slidermax: Incomplete | None = None, dragging: bool = True, valstep: Incomplete | None = None, orientation: str = 'horizontal', initcolor: str = 'r', track_color: str = 'lightgrey', handle_style: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : float, default: 0.5
            The slider initial position.

        valfmt : str, default: None
            %-format string used to format the slider value.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        slidermin : Slider, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider *slidermin*.

        slidermax : Slider, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider *slidermax*.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float or array-like, default: None
            If a float, the slider will snap to multiples of *valstep*.
            If an array the slider will snap to the values in the array.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        initcolor : :mpltype:`color`, default: 'r'
            The color of the line at the *valinit* position. Set to ``'none'``
            for no line.

        track_color : :mpltype:`color`, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handle. Default values are

            ========= ===== ======= ========================================
            Key       Value Default Description
            ========= ===== ======= ========================================
            facecolor color 'white' The facecolor of the slider handle.
            edgecolor color '.75'   The edgecolor of the slider handle.
            size      int   10      The size of the slider handle in points.
            ========= ===== ======= ========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Rectangle` that draws the slider knob.  See the
        `.Rectangle` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
    def _value_in_bounds(self, val):
        """Makes sure *val* is with given bounds."""
    drag_active: bool
    def _update(self, event) -> None:
        """Update the slider position."""
    def _format(self, val):
        """Pretty-print *val*."""
    def set_val(self, val) -> None:
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : float
        """
    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """

class RangeSlider(SliderBase):
    """
    A slider representing a range of floating point values. Defines the min and
    max of the range via the *val* attribute as a tuple of (min, max).

    Create a slider that defines a range contained within [*valmin*, *valmax*]
    in Axes *ax*. For the slider to remain responsive you must maintain a
    reference to it. Call :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : tuple of float
        Slider value.
    """
    val: Incomplete
    valinit: Incomplete
    track: Incomplete
    poly: Incomplete
    _handles: Incomplete
    label: Incomplete
    valtext: Incomplete
    _active_handle: Incomplete
    def __init__(self, ax, label, valmin, valmax, *, valinit: Incomplete | None = None, valfmt: Incomplete | None = None, closedmin: bool = True, closedmax: bool = True, dragging: bool = True, valstep: Incomplete | None = None, orientation: str = 'horizontal', track_color: str = 'lightgrey', handle_style: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : tuple of float or None, default: None
            The initial positions of the slider. If None the initial positions
            will be at the 25th and 75th percentiles of the range.

        valfmt : str, default: None
            %-format string used to format the slider values.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float, default: None
            If given, the slider will snap to multiples of *valstep*.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        track_color : :mpltype:`color`, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handles. Default values are

            ========= ===== ======= =========================================
            Key       Value Default Description
            ========= ===== ======= =========================================
            facecolor color 'white' The facecolor of the slider handles.
            edgecolor color '.75'   The edgecolor of the slider handles.
            size      int   10      The size of the slider handles in points.
            ========= ===== ======= =========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Polygon` that draws the slider knob.  See the
        `.Polygon` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
    def _update_selection_poly(self, vmin, vmax) -> None:
        """
        Update the vertices of the *self.poly* slider in-place
        to cover the data range *vmin*, *vmax*.
        """
    def _min_in_bounds(self, min):
        """Ensure the new min value is between valmin and self.val[1]."""
    def _max_in_bounds(self, max):
        """Ensure the new max value is between valmax and self.val[0]."""
    def _value_in_bounds(self, vals):
        """Clip min, max values to the bounds."""
    def _update_val_from_pos(self, pos) -> None:
        """Update the slider value based on a given position."""
    drag_active: bool
    def _update(self, event) -> None:
        """Update the slider position."""
    def _format(self, val):
        """Pretty-print *val*."""
    def set_min(self, min) -> None:
        """
        Set the lower value of the slider to *min*.

        Parameters
        ----------
        min : float
        """
    def set_max(self, max) -> None:
        """
        Set the lower value of the slider to *max*.

        Parameters
        ----------
        max : float
        """
    def set_val(self, val) -> None:
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed. The function
            must accept a 2-tuple of floats as its argument.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """

def _expand_text_props(props): ...

class CheckButtons(AxesWidget):
    """
    A GUI neutral set of check buttons.

    For the check buttons to remain responsive you must keep a
    reference to this object.

    Connect to the CheckButtons with the `.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    labels : list of `~matplotlib.text.Text`
        The text label objects of the check buttons.
    """
    _useblit: Incomplete
    _background: Incomplete
    labels: Incomplete
    _frames: Incomplete
    _checks: Incomplete
    _observers: Incomplete
    def __init__(self, ax, labels, actives: Incomplete | None = None, *, useblit: bool = True, label_props: Incomplete | None = None, frame_props: Incomplete | None = None, check_props: Incomplete | None = None) -> None:
        """
        Add check buttons to `~.axes.Axes` instance *ax*.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The parent Axes for the widget.
        labels : list of str
            The labels of the check buttons.
        actives : list of bool, optional
            The initial check states of the buttons. The list must have the
            same length as *labels*. If not given, all buttons are unchecked.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :ref:`blitting` for details.

            .. versionadded:: 3.7

        label_props : dict, optional
            Dictionary of `.Text` properties to be used for the labels.

            .. versionadded:: 3.7
        frame_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            check button frame. Defaults (label font size / 2)**2 size, black
            edgecolor, no facecolor, and 1.0 linewidth.

            .. versionadded:: 3.7
        check_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            check button check. Defaults to (label font size / 2)**2 size,
            black color, and 1.0 linewidth.

            .. versionadded:: 3.7
        """
    def _clear(self, event) -> None:
        """Internal event handler to clear the buttons."""
    def _clicked(self, event) -> None: ...
    def set_label_props(self, props) -> None:
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels.
        """
    def set_frame_props(self, props) -> None:
        """
        Set properties of the check button frames.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button frames.
        """
    def set_check_props(self, props) -> None:
        """
        Set properties of the check button checks.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button check.
        """
    def set_active(self, index, state: Incomplete | None = None) -> None:
        """
        Modify the state of a check button by index.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            Index of the check button to toggle.

        state : bool, optional
            If a boolean value, set the state explicitly. If no value is
            provided, the state is toggled.

        Raises
        ------
        ValueError
            If *index* is invalid.
        TypeError
            If *state* is not boolean.
        """
    _active_check_colors: Incomplete
    def _init_status(self, actives) -> None:
        """
        Initialize properties to match active status.

        The user may have passed custom colours in *check_props* to the
        constructor, or to `.set_check_props`, so we need to modify the
        visibility after getting whatever the user set.
        """
    def clear(self) -> None:
        """Uncheck all checkboxes."""
    def get_status(self):
        """
        Return a list of the status (True/False) of all of the check buttons.
        """
    def get_checked_labels(self):
        """Return a list of labels currently checked by user."""
    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Parameters
        ----------
        func : callable
            When the button is clicked, call *func* with button label.
            When all buttons are cleared, call *func* with None.
            The callback func must have the signature::

                def func(label: str | None) -> Any

            Return values may exist, but are ignored.

        Returns
        -------
        A connection id, which can be used to disconnect the callback.
        """
    def disconnect(self, cid) -> None:
        """Remove the observer with connection id *cid*."""

class TextBox(AxesWidget):
    """
    A GUI neutral text input box.

    For the text box to remain responsive you must keep a reference to it.

    Call `.on_text_change` to be updated whenever the text changes.

    Call `.on_submit` to be updated whenever the user hits enter or
    leaves the text entry field.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    label : `~matplotlib.text.Text`

    color : :mpltype:`color`
        The color of the text box when not hovering.
    hovercolor : :mpltype:`color`
        The color of the text box when hovering.
    """
    _text_position: Incomplete
    label: Incomplete
    text_disp: Incomplete
    _observers: Incomplete
    cursor_index: int
    cursor: Incomplete
    color: Incomplete
    hovercolor: Incomplete
    capturekeystrokes: bool
    def __init__(self, ax, label, initial: str = '', *, color: str = '.95', hovercolor: str = '1', label_pad: float = 0.01, textalignment: str = 'left') -> None:
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            Label for this text box.
        initial : str
            Initial value in the text box.
        color : :mpltype:`color`
            The color of the box.
        hovercolor : :mpltype:`color`
            The color of the box when the mouse is over it.
        label_pad : float
            The distance between the label and the right side of the textbox.
        textalignment : {'left', 'center', 'right'}
            The horizontal location of the text.
        """
    @property
    def text(self): ...
    def _rendercursor(self) -> None: ...
    def _release(self, event) -> None: ...
    def _keypress(self, event) -> None: ...
    def set_val(self, val) -> None: ...
    _on_stop_typing: Incomplete
    def begin_typing(self) -> None: ...
    def stop_typing(self) -> None: ...
    def _click(self, event) -> None: ...
    def _resize(self, event) -> None: ...
    def _motion(self, event) -> None: ...
    def on_text_change(self, func):
        """
        When the text changes, call this *func* with event.

        A connection id is returned which can be used to disconnect.
        """
    def on_submit(self, func):
        """
        When the user hits enter or leaves the submission box, call this
        *func* with event.

        A connection id is returned which can be used to disconnect.
        """
    def disconnect(self, cid) -> None:
        """Remove the observer with connection id *cid*."""

class RadioButtons(AxesWidget):
    """
    A GUI neutral radio button.

    For the buttons to remain responsive you must keep a reference to this
    object.

    Connect to the RadioButtons with the `.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    activecolor : :mpltype:`color`
        The color of the selected button.
    labels : list of `.Text`
        The button labels.
    value_selected : str
        The label text of the currently selected button.
    index_selected : int
        The index of the selected button.
    """
    _activecolor: Incomplete
    _initial_active: Incomplete
    value_selected: Incomplete
    index_selected: Incomplete
    _useblit: Incomplete
    _background: Incomplete
    labels: Incomplete
    _buttons: Incomplete
    _active_colors: Incomplete
    _observers: Incomplete
    def __init__(self, ax, labels, active: int = 0, activecolor: Incomplete | None = None, *, useblit: bool = True, label_props: Incomplete | None = None, radio_props: Incomplete | None = None) -> None:
        """
        Add radio buttons to an `~.axes.Axes`.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : :mpltype:`color`
            The color of the selected button. The default is ``'blue'`` if not
            specified here or in *radio_props*.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :ref:`blitting` for details.

            .. versionadded:: 3.7

        label_props : dict or list of dict, optional
            Dictionary of `.Text` properties to be used for the labels.

            .. versionadded:: 3.7
        radio_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            radio buttons. Defaults to (label font size / 2)**2 size, black
            edgecolor, and *activecolor* facecolor (when active).

            .. note::
                If a facecolor is supplied in *radio_props*, it will override
                *activecolor*. This may be used to provide an active color per
                button.

            .. versionadded:: 3.7
        """
    def _clear(self, event) -> None:
        """Internal event handler to clear the buttons."""
    def _clicked(self, event) -> None: ...
    def set_label_props(self, props) -> None:
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels.
        """
    def set_radio_props(self, props) -> None:
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the radio
            buttons.
        """
    @property
    def activecolor(self): ...
    @activecolor.setter
    def activecolor(self, activecolor) -> None: ...
    def set_active(self, index) -> None:
        """
        Select button with number *index*.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            The index of the button to activate.

        Raises
        ------
        ValueError
            If the index is invalid.
        """
    def clear(self) -> None:
        """Reset the active button to the initially active one."""
    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Parameters
        ----------
        func : callable
            When the button is clicked, call *func* with button label.
            When all buttons are cleared, call *func* with None.
            The callback func must have the signature::

                def func(label: str | None) -> Any

            Return values may exist, but are ignored.

        Returns
        -------
        A connection id, which can be used to disconnect the callback.
        """
    def disconnect(self, cid) -> None:
        """Remove the observer with connection id *cid*."""

class SubplotTool(Widget):
    """
    A tool to adjust the subplot params of a `.Figure`.
    """
    figure: Incomplete
    targetfig: Incomplete
    _sliders: Incomplete
    buttonreset: Incomplete
    def __init__(self, targetfig, toolfig) -> None:
        """
        Parameters
        ----------
        targetfig : `~matplotlib.figure.Figure`
            The figure instance to adjust.
        toolfig : `~matplotlib.figure.Figure`
            The figure instance to embed the subplot tool into.
        """
    def _on_slider_changed(self, _) -> None: ...
    def _on_reset(self, event) -> None: ...

class Cursor(AxesWidget):
    """
    A crosshair cursor that spans the Axes and moves with mouse cursor.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.
        See the tutorial :ref:`blitting` for details.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    Examples
    --------
    See :doc:`/gallery/widgets/cursor`.
    """
    visible: bool
    horizOn: Incomplete
    vertOn: Incomplete
    useblit: Incomplete
    lineh: Incomplete
    linev: Incomplete
    background: Incomplete
    needclear: bool
    def __init__(self, ax, *, horizOn: bool = True, vertOn: bool = True, useblit: bool = False, **lineprops) -> None: ...
    def clear(self, event) -> None:
        """Internal event handler to clear the cursor."""
    def onmove(self, event) -> None:
        """Internal event handler to draw the cursor when the mouse moves."""

class MultiCursor(Widget):
    """
    Provide a vertical (default) and/or horizontal line cursor shared between
    multiple Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    canvas : object
        This parameter is entirely unused and only kept for back-compatibility.

    axes : list of `~matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.

    useblit : bool, default: True
        Use blitting for faster drawing if supported by the backend.
        See the tutorial :ref:`blitting`
        for details.

    horizOn : bool, default: False
        Whether to draw the horizontal line.

    vertOn : bool, default: True
        Whether to draw the vertical line.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    Examples
    --------
    See :doc:`/gallery/widgets/multicursor`.
    """
    _canvas: Incomplete
    axes: Incomplete
    horizOn: Incomplete
    vertOn: Incomplete
    _canvas_infos: Incomplete
    visible: bool
    useblit: Incomplete
    vlines: Incomplete
    hlines: Incomplete
    def __init__(self, canvas, axes, *, useblit: bool = True, horizOn: bool = False, vertOn: bool = True, **lineprops) -> None: ...
    def connect(self) -> None:
        """Connect events."""
    def disconnect(self) -> None:
        """Disconnect events."""
    def clear(self, event) -> None:
        """Clear the cursor."""
    def onmove(self, event) -> None: ...

class _SelectorWidget(AxesWidget):
    _visible: bool
    onselect: Incomplete
    useblit: Incomplete
    _state_modifier_keys: Incomplete
    _use_data_coordinates: Incomplete
    background: Incomplete
    validButtons: Incomplete
    _selection_completed: bool
    _eventpress: Incomplete
    _eventrelease: Incomplete
    _prev_event: Incomplete
    _state: Incomplete
    def __init__(self, ax, onselect: Incomplete | None = None, useblit: bool = False, button: Incomplete | None = None, state_modifier_keys: Incomplete | None = None, use_data_coordinates: bool = False) -> None: ...
    def set_active(self, active) -> None: ...
    def _get_animated_artists(self):
        """
        Convenience method to get all animated artists of the figure containing
        this widget, excluding those already present in self.artists.
        The returned tuple is not sorted by 'z_order': z_order sorting is
        valid only when considering all artists and not only a subset of all
        artists.
        """
    def update_background(self, event):
        """Force an update of the background."""
    def connect_default_events(self) -> None:
        """Connect the major canvas events to methods."""
    def ignore(self, event): ...
    def update(self):
        """Draw using blit() or draw_idle(), depending on ``self.useblit``."""
    def _get_data(self, event):
        """Get the xdata and ydata for event, with limits."""
    def _clean_event(self, event):
        """
        Preprocess an event:

        - Replace *event* by the previous event if *event* has no ``xdata``.
        - Get ``xdata`` and ``ydata`` from this widget's Axes, and clip them to the axes
          limits.
        - Update the previous event.
        """
    def press(self, event):
        """Button press handler and validator."""
    def _press(self, event) -> None:
        """Button press event handler."""
    def release(self, event):
        """Button release event handler and validator."""
    def _release(self, event) -> None:
        """Button release event handler."""
    def onmove(self, event):
        """Cursor move event handler and validator."""
    def _onmove(self, event) -> None:
        """Cursor move event handler."""
    def on_scroll(self, event) -> None:
        """Mouse scroll event handler and validator."""
    def _on_scroll(self, event) -> None:
        """Mouse scroll event handler."""
    def on_key_press(self, event) -> None:
        """Key press event handler and validator for all selection widgets."""
    def _on_key_press(self, event) -> None:
        """Key press event handler - for widget-specific key press actions."""
    def on_key_release(self, event) -> None:
        """Key release event handler and validator."""
    def _on_key_release(self, event) -> None:
        """Key release event handler."""
    def set_visible(self, visible) -> None:
        """Set the visibility of the selector artists."""
    def get_visible(self):
        """Get the visibility of the selector artists."""
    def clear(self) -> None:
        """Clear the selection and set the selector ready to make a new one."""
    def _clear_without_update(self) -> None: ...
    @property
    def artists(self):
        """Tuple of the artists of the selector."""
    def set_props(self, **props) -> None:
        """
        Set the properties of the selector artist.

        See the *props* argument in the selector docstring to know which properties are
        supported.
        """
    def set_handle_props(self, **handle_props) -> None:
        """
        Set the properties of the handles selector artist. See the
        `handle_props` argument in the selector docstring to know which
        properties are supported.
        """
    def _validate_state(self, state) -> None: ...
    def add_state(self, state) -> None:
        """
        Add a state to define the widget's behavior. See the
        `state_modifier_keys` parameters for details.

        Parameters
        ----------
        state : str
            Must be a supported state of the selector. See the
            `state_modifier_keys` parameters for details.

        Raises
        ------
        ValueError
            When the state is not supported by the selector.

        """
    def remove_state(self, state) -> None:
        """
        Remove a state to define the widget's behavior. See the
        `state_modifier_keys` parameters for details.

        Parameters
        ----------
        state : str
            Must be a supported state of the selector. See the
            `state_modifier_keys` parameters for details.

        Raises
        ------
        ValueError
            When the state is not supported by the selector.

        """

class SpanSelector(_SelectorWidget):
    '''
    Visually select a min/max range on a single axis and call a function with
    those values.

    To guarantee that the selector remains responsive, keep a reference to it.

    In order to turn off the SpanSelector, set ``span_selector.active`` to
    False.  To turn it back on, set it to True.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`

    onselect : callable with signature ``func(min: float, max: float)``
        A callback function that is called after a release event and the
        selection is created, changed or removed.

    direction : {"horizontal", "vertical"}
        The direction along which to draw the span selector.

    minspan : float, default: 0
        If selection is less than or equal to *minspan*, the selection is
        removed (when already existing) or cancelled.

    useblit : bool, default: False
        If True, use the backend-dependent blitting features for faster
        canvas updates. See the tutorial :ref:`blitting` for details.

    props : dict, default: {\'facecolor\': \'red\', \'alpha\': 0.5}
        Dictionary of `.Patch` properties.

    onmove_callback : callable with signature ``func(min: float, max: float)``, optional
        Called on mouse move while the span is being selected.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    button : `.MouseButton` or list of `.MouseButton`, default: all buttons
        The mouse buttons which activate the span selector.

    handle_props : dict, default: None
        Properties of the handle lines at the edges of the span. Only used
        when *interactive* is True. See `.Line2D` for valid properties.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be activated.

    state_modifier_keys : dict, optional
        Keyboard modifiers which affect the widget\'s behavior.  Values
        amend the defaults, which are:

        - "clear": Clear the current shape, default: "escape".

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be ignored.

    snap_values : 1D array-like, optional
        Snap the selector edges to the given values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(vmin, vmax):
    ...     print(vmin, vmax)
    >>> span = mwidgets.SpanSelector(ax, onselect, \'horizontal\',
    ...                              props=dict(facecolor=\'blue\', alpha=0.5))
    >>> fig.show()

    See also: :doc:`/gallery/widgets/span_selector`
    '''
    _extents_on_press: Incomplete
    snap_values: Incomplete
    onmove_callback: Incomplete
    minspan: Incomplete
    grab_range: Incomplete
    _interactive: Incomplete
    _edge_handles: Incomplete
    drag_from_anywhere: Incomplete
    ignore_event_outside: Incomplete
    _handle_props: Incomplete
    _edge_order: Incomplete
    _active_handle: Incomplete
    def __init__(self, ax, onselect, direction, *, minspan: int = 0, useblit: bool = False, props: Incomplete | None = None, onmove_callback: Incomplete | None = None, interactive: bool = False, button: Incomplete | None = None, handle_props: Incomplete | None = None, grab_range: int = 10, state_modifier_keys: Incomplete | None = None, drag_from_anywhere: bool = False, ignore_event_outside: bool = False, snap_values: Incomplete | None = None) -> None: ...
    ax: Incomplete
    _selection_completed: bool
    _selection_artist: Incomplete
    def new_axes(self, ax, *, _props: Incomplete | None = None, _init: bool = False) -> None:
        """Set SpanSelector to operate on a new Axes."""
    def _setup_edge_handles(self, props) -> None: ...
    @property
    def _handles_artists(self): ...
    def _set_cursor(self, enabled) -> None:
        """Update the canvas cursor based on direction of the selector."""
    def connect_default_events(self) -> None: ...
    _visible: bool
    def _press(self, event):
        """Button press event handler."""
    @property
    def direction(self):
        """Direction of the span selector: 'vertical' or 'horizontal'."""
    _direction: Incomplete
    @direction.setter
    def direction(self, direction) -> None:
        """Set the direction of the span selector."""
    def _release(self, event):
        """Button release event handler."""
    def _hover(self, event) -> None:
        """Update the canvas cursor if it's over a handle."""
    def _onmove(self, event):
        """Motion notify event handler."""
    def _draw_shape(self, vmin, vmax) -> None: ...
    def _set_active_handle(self, event) -> None:
        """Set active handle based on the location of the mouse event."""
    def _contains(self, event):
        """Return True if event is within the patch."""
    @staticmethod
    def _snap(values, snap_values):
        """Snap values to a given array values (snap_values)."""
    @property
    def extents(self):
        """
        (float, float)
            The values, in data coordinates, for the start and end points of the current
            selection. If there is no selection then the start and end values will be
            the same.
        """
    @extents.setter
    def extents(self, extents) -> None: ...
    def _set_extents(self, extents) -> None: ...

class ToolLineHandles:
    '''
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    positions : 1D array
        Positions of handles in data coordinates.
    direction : {"horizontal", "vertical"}
        Direction of handles, either \'vertical\' or \'horizontal\'
    line_props : dict, optional
        Additional line properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    '''
    ax: Incomplete
    _direction: Incomplete
    _artists: Incomplete
    def __init__(self, ax, positions, direction, *, line_props: Incomplete | None = None, useblit: bool = True) -> None: ...
    @property
    def artists(self): ...
    @property
    def positions(self):
        """Positions of the handle in data coordinates."""
    @property
    def direction(self):
        """Direction of the handle: 'vertical' or 'horizontal'."""
    def set_data(self, positions) -> None:
        """
        Set x- or y-positions of handles, depending on if the lines are
        vertical or horizontal.

        Parameters
        ----------
        positions : tuple of length 2
            Set the positions of the handle in data coordinates
        """
    def set_visible(self, value) -> None:
        """Set the visibility state of the handles artist."""
    def set_animated(self, value) -> None:
        """Set the animated state of the handles artist."""
    def remove(self) -> None:
        """Remove the handles artist from the figure."""
    def closest(self, x, y):
        """
        Return index and pixel distance to closest handle.

        Parameters
        ----------
        x, y : float
            x, y position from which the distance will be calculated to
            determinate the closest handle

        Returns
        -------
        index, distance : index of the handle and its distance from
            position x, y
        """

class ToolHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str, default: 'o'
        Shape of marker used to display handle. See `~.pyplot.plot`.
    marker_props : dict, optional
        Additional marker properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """
    ax: Incomplete
    _markers: Incomplete
    def __init__(self, ax, x, y, *, marker: str = 'o', marker_props: Incomplete | None = None, useblit: bool = True) -> None: ...
    @property
    def x(self): ...
    @property
    def y(self): ...
    @property
    def artists(self): ...
    def set_data(self, pts, y: Incomplete | None = None) -> None:
        """Set x and y positions of handles."""
    def set_visible(self, val) -> None: ...
    def set_animated(self, val) -> None: ...
    def closest(self, x, y):
        """Return index and pixel distance to closest index."""

_RECTANGLESELECTOR_PARAMETERS_DOCSTRING: str

class RectangleSelector(_SelectorWidget):
    """
    Select a rectangular region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(eclick, erelease):
    ...     print(eclick.xdata, eclick.ydata)
    ...     print(erelease.xdata, erelease.ydata)
    >>> props = dict(facecolor='blue', alpha=0.5)
    >>> rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,
    ...                                   props=props)
    >>> fig.show()
    >>> rect.add_state('square')

    See also: :doc:`/gallery/widgets/rectangle_selector`
    """
    _interactive: Incomplete
    drag_from_anywhere: Incomplete
    ignore_event_outside: Incomplete
    _rotation: float
    _aspect_ratio_correction: float
    _allow_creation: bool
    _visible: Incomplete
    _selection_artist: Incomplete
    minspanx: Incomplete
    minspany: Incomplete
    spancoords: Incomplete
    grab_range: Incomplete
    _handle_props: Incomplete
    _corner_order: Incomplete
    _corner_handles: Incomplete
    _edge_order: Incomplete
    _edge_handles: Incomplete
    _center_handle: Incomplete
    _active_handle: Incomplete
    _extents_on_press: Incomplete
    def __init__(self, ax, onselect: Incomplete | None = None, *, minspanx: int = 0, minspany: int = 0, useblit: bool = False, props: Incomplete | None = None, spancoords: str = 'data', button: Incomplete | None = None, grab_range: int = 10, handle_props: Incomplete | None = None, interactive: bool = False, state_modifier_keys: Incomplete | None = None, drag_from_anywhere: bool = False, ignore_event_outside: bool = False, use_data_coordinates: bool = False) -> None: ...
    @property
    def _handles_artists(self): ...
    def _init_shape(self, **props): ...
    _rotation_on_press: Incomplete
    def _press(self, event):
        """Button press event handler."""
    _selection_completed: bool
    def _release(self, event):
        """Button release event handler."""
    def _onmove(self, event) -> None:
        """
        Motion notify event handler.

        This can do one of four things:
        - Translate
        - Rotate
        - Re-size
        - Continue the creation of a new shape
        """
    @property
    def _rect_bbox(self): ...
    def _set_aspect_ratio_correction(self) -> None: ...
    def _get_rotation_transform(self): ...
    @property
    def corners(self):
        """
        Corners of rectangle in data coordinates from lower left,
        moving clockwise.
        """
    @property
    def edge_centers(self):
        """
        Midpoint of rectangle edges in data coordinates from left,
        moving anti-clockwise.
        """
    @property
    def center(self):
        """Center of rectangle in data coordinates."""
    @property
    def extents(self):
        """
        Return (xmin, xmax, ymin, ymax) in data coordinates as defined by the
        bounding box before rotation.
        """
    @extents.setter
    def extents(self, extents) -> None: ...
    @property
    def rotation(self):
        """
        Rotation in degree in interval [-45°, 45°]. The rotation is limited in
        range to keep the implementation simple.
        """
    @rotation.setter
    def rotation(self, value) -> None: ...
    def _draw_shape(self, extents) -> None: ...
    def _set_active_handle(self, event) -> None:
        """Set active handle based on the location of the mouse event."""
    def _contains(self, event):
        """Return True if event is within the patch."""
    @property
    def geometry(self):
        """
        Return an array of shape (2, 5) containing the
        x (``RectangleSelector.geometry[1, :]``) and
        y (``RectangleSelector.geometry[0, :]``) data coordinates of the four
        corners of the rectangle starting and ending in the top left corner.
        """

class EllipseSelector(RectangleSelector):
    """
    Select an elliptical region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    :doc:`/gallery/widgets/rectangle_selector`
    """
    def _init_shape(self, **props): ...
    def _draw_shape(self, extents) -> None: ...
    @property
    def _rect_bbox(self): ...

class LassoSelector(_SelectorWidget):
    """
    Selection curve of an arbitrary shape.

    For the selector to remain responsive you must keep a reference to it.

    The selected path can be used in conjunction with `~.Path.contains_point`
    to select data points from an image.

    In contrast to `Lasso`, `LassoSelector` is written with an interface
    similar to `RectangleSelector` and `SpanSelector`, and will continue to
    interact with the Axes until disconnected.

    Example usage::

        ax = plt.subplot()
        ax.plot(x, y)

        def onselect(verts):
            print(verts)
        lasso = LassoSelector(ax, onselect)

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    onselect : function, optional
        Whenever the lasso is released, the *onselect* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    props : dict, optional
        Properties with which the line is drawn, see `.Line2D`
        for valid properties. Default values are defined in ``mpl.rcParams``.
    button : `.MouseButton` or list of `.MouseButton`, optional
        The mouse buttons used for rectangle selection.  Default is ``None``,
        which corresponds to all buttons.
    """
    verts: Incomplete
    _selection_artist: Incomplete
    def __init__(self, ax, onselect: Incomplete | None = None, *, useblit: bool = True, props: Incomplete | None = None, button: Incomplete | None = None) -> None: ...
    def _press(self, event) -> None: ...
    def _release(self, event) -> None: ...
    def _onmove(self, event) -> None: ...

class PolygonSelector(_SelectorWidget):
    """
    Select a polygon region of an Axes.

    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Once drawn individual vertices
    can be moved by clicking and dragging with the left mouse button, or
    removed by clicking the right mouse button.

    In addition, the following modifier keys can be used:

    - Hold *ctrl* and click and drag a vertex to reposition it before the
      polygon has been completed.
    - Hold the *shift* key and click and drag anywhere in the Axes to move
      all vertices.
    - Press the *esc* key to start a new polygon.

    For the selector to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    onselect : function, optional
        When a polygon is completed or modified after completion,
        the *onselect* function is called and passed a list of the vertices as
        ``(xdata, ydata)`` tuples.

    useblit : bool, default: False
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.

    props : dict, optional
        Properties with which the line is drawn, see `.Line2D` for valid properties.
        Default::

            dict(color='k', linestyle='-', linewidth=2, alpha=0.5)

    handle_props : dict, optional
        Artist properties for the markers drawn at the vertices of the polygon.
        See the marker arguments in `.Line2D` for valid
        properties.  Default values are defined in ``mpl.rcParams`` except for
        the default value of ``markeredgecolor`` which will be the same as the
        ``color`` property in *props*.

    grab_range : float, default: 10
        A vertex is selected (to complete the polygon or to move a vertex) if
        the mouse click is within *grab_range* pixels of the vertex.

    draw_bounding_box : bool, optional
        If `True`, a bounding box will be drawn around the polygon selector
        once it is complete. This box can be used to move and resize the
        selector.

    box_handle_props : dict, optional
        Properties to set for the box handles. See the documentation for the
        *handle_props* argument to `RectangleSelector` for more info.

    box_props : dict, optional
        Properties to set for the box. See the documentation for the *props*
        argument to `RectangleSelector` for more info.

    Examples
    --------
    :doc:`/gallery/widgets/polygon_selector_simple`
    :doc:`/gallery/widgets/polygon_selector_demo`

    Notes
    -----
    If only one point remains after removing points, the selector reverts to an
    incomplete state and you can start drawing a new polygon from the existing
    point.
    """
    _xys: Incomplete
    _selection_artist: Incomplete
    _handle_props: Incomplete
    _polygon_handles: Incomplete
    _active_handle_idx: int
    grab_range: Incomplete
    _draw_box: Incomplete
    _box: Incomplete
    _box_handle_props: Incomplete
    _box_props: Incomplete
    def __init__(self, ax, onselect: Incomplete | None = None, *, useblit: bool = False, props: Incomplete | None = None, handle_props: Incomplete | None = None, grab_range: int = 10, draw_bounding_box: bool = False, box_handle_props: Incomplete | None = None, box_props: Incomplete | None = None) -> None: ...
    def _get_bbox(self): ...
    def _add_box(self) -> None: ...
    def _remove_box(self) -> None: ...
    _old_box_extents: Incomplete
    def _update_box(self) -> None: ...
    def _scale_polygon(self, event) -> None:
        """
        Scale the polygon selector points when the bounding box is moved or
        scaled.

        This is set as a callback on the bounding box RectangleSelector.
        """
    @property
    def _handles_artists(self): ...
    _selection_completed: bool
    def _remove_vertex(self, i) -> None:
        """Remove vertex with index i."""
    _xys_at_press: Incomplete
    def _press(self, event) -> None:
        """Button press event handler."""
    def _release(self, event) -> None:
        """Button release event handler."""
    def onmove(self, event):
        """Cursor move event handler and validator."""
    def _onmove(self, event) -> None:
        """Cursor move event handler."""
    def _on_key_press(self, event) -> None:
        """Key press event handler."""
    def _on_key_release(self, event) -> None:
        """Key release event handler."""
    def _draw_polygon_without_update(self) -> None:
        """Redraw the polygon based on new vertex positions, no update()."""
    def _draw_polygon(self) -> None:
        """Redraw the polygon based on the new vertex positions."""
    @property
    def verts(self):
        """The polygon vertices, as a list of ``(x, y)`` pairs."""
    @verts.setter
    def verts(self, xys) -> None:
        """
        Set the polygon vertices.

        This will remove any preexisting vertices, creating a complete polygon
        with the new vertices.
        """
    def _clear_without_update(self) -> None: ...

class Lasso(AxesWidget):
    """
    Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    `~matplotlib.path.Path.contains_point` to select data points from an image.

    Unlike `LassoSelector`, this must be initialized with a starting
    point *xy*, and the `Lasso` events are destroyed upon release.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    xy : (float, float)
        Coordinates of the start of the lasso.
    callback : callable
        Whenever the lasso is released, the *callback* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    props: dict, optional
        Lasso line properties. See `.Line2D` for valid properties.
        Default *props* are::

            {'linestyle' : '-', 'color' : 'black', 'lw' : 2}

        .. versionadded:: 3.9
    """
    useblit: Incomplete
    background: Incomplete
    verts: Incomplete
    line: Incomplete
    callback: Incomplete
    def __init__(self, ax, xy, callback, *, useblit: bool = True, props: Incomplete | None = None) -> None: ...
    def onrelease(self, event) -> None: ...
    def onmove(self, event) -> None: ...

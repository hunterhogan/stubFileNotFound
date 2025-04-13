def Win32_GetCurrentProcessExplicitAppUserModelID() -> object:
    """Win32_GetCurrentProcessExplicitAppUserModelID() -> object

            --
            Wrapper for Windows's GetCurrentProcessExplicitAppUserModelID.

            On non-Windows platforms, always returns None.
    """
def Win32_GetForegroundWindow() -> object:
    """Win32_GetForegroundWindow() -> object

            --
            Wrapper for Windows' GetForegroundWindow.

            On non-Windows platforms, always returns None.
    """
def Win32_SetCurrentProcessExplicitAppUserModelID(*args, **kwargs):
    """Win32_SetCurrentProcessExplicitAppUserModelID(appid: str, /) -> None

            --
            Wrapper for Windows's SetCurrentProcessExplicitAppUserModelID.

            On non-Windows platforms, does nothing.
    """
def Win32_SetForegroundWindow(hwnd: capsule) -> None:
    """Win32_SetForegroundWindow(hwnd: capsule) -> None

            --
            Wrapper for Windows' SetForegroundWindow.

            On non-Windows platforms, does nothing.
    """
def Win32_SetProcessDpiAwareness_max() -> None:
    """Win32_SetProcessDpiAwareness_max() -> None

            --
            Set Windows' process DPI awareness to best option available.

            On non-Windows platforms, does nothing.
    """
def display_is_valid() -> bool:
    """display_is_valid() -> bool

            --
            Check whether the current X11 or Wayland display is valid.

            On Linux, returns True if either $DISPLAY is set and XOpenDisplay(NULL)
            succeeds, or $WAYLAND_DISPLAY is set and wl_display_connect(NULL)
            succeeds.

            On other platforms, always returns True.
    """
def xdisplay_is_valid() -> bool:
    """xdisplay_is_valid() -> bool

            --
            Check whether the current X11 display is valid.

            On Linux, returns True if either $DISPLAY is set and XOpenDisplay(NULL)
            succeeds. Use this function if you need to specifically check for X11
            only (e.g., for Tkinter).

            On other platforms, always returns True.
    """

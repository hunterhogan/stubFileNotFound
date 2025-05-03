import pandas.errors

__all__ = ['copy', 'paste', 'waitForPaste', 'waitForNewPaste', 'set_clipboard', 'determine_clipboard']

class PyperclipTimeoutException(pandas.errors.PyperclipException): ...

class CheckedCall:
    def __init__(self, f) -> None: ...
    def __call__(self, *args): ...
    def __setattr__(self, key, value): ...
def determine_clipboard():
    """
    Determine the OS/platform and set the copy() and paste() functions
    accordingly.
    """
def set_clipboard(clipboard):
    '''
    Explicitly sets the clipboard mechanism. The "clipboard mechanism" is how
    the copy() and paste() functions interact with the operating system to
    implement the copy/paste feature. The clipboard parameter must be one of:
        - pbcopy
        - pyobjc (default on macOS)
        - qt
        - xclip
        - xsel
        - klipper
        - windows (default on Windows)
        - no (this is what is set when no clipboard mechanism can be found)
    '''
def copy(text):
    """
    A stub function for copy(), which will load the real copy() function when
    called so that the real copy() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
def paste():
    """
    A stub function for paste(), which will load the real paste() function when
    called so that the real paste() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
def waitForPaste(timeout):
    """This function call blocks until a non-empty text string exists on the
    clipboard. It returns this text.

    This function raises PyperclipTimeoutException if timeout was set to
    a number of seconds that has elapsed without non-empty text being put on
    the clipboard."""
def waitForNewPaste(timeout):
    """This function call blocks until a new text string exists on the
    clipboard that is different from the text that was there when the function
    was first called. It returns this text.

    This function raises PyperclipTimeoutException if timeout was set to
    a number of seconds that has elapsed without non-empty text being put on
    the clipboard."""

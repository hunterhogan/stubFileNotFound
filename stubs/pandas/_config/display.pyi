import pandas._config.config as cf

_initial_defencoding: str
def detect_console_encoding() -> str:
    """
    Try to find the most capable encoding supported by the console.
    slightly modified from the way IPython handles the same issue.
    """

pc_encoding_doc: str

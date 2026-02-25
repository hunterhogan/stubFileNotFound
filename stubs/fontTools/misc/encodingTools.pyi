def getEncoding(platformID: int, platEncID: int, langID: int, default: str | None = None) -> str | None:
    """Returns the Python encoding name for OpenType platformID/encodingID/langID triplet.

    If encoding for these values is not known, by default None is
    returned.  That can be overriden by passing a value to the default argument.
    """

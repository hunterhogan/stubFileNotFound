from ._internal_utils import to_native_string as to_native_string
from .compat import Morsel as Morsel, MutableMapping as MutableMapping, cookielib as cookielib, urlparse as urlparse, urlunparse as urlunparse
from _typeshed import Incomplete
from collections.abc import Generator

class MockRequest:
    """Wraps a `requests.Request` to mimic a `urllib2.Request`.

    The code in `http.cookiejar.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """
    _r: Incomplete
    _new_headers: Incomplete
    type: Incomplete
    def __init__(self, request) -> None: ...
    def get_type(self): ...
    def get_host(self): ...
    def get_origin_req_host(self): ...
    def get_full_url(self): ...
    def is_unverifiable(self): ...
    def has_header(self, name): ...
    def get_header(self, name, default: Incomplete | None = None): ...
    def add_header(self, key, val) -> None:
        """cookiejar has no legitimate use for this method; add it back if you find one."""
    def add_unredirected_header(self, name, value) -> None: ...
    def get_new_headers(self): ...
    @property
    def unverifiable(self): ...
    @property
    def origin_req_host(self): ...
    @property
    def host(self): ...

class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `http.cookiejar` expects to see them.
    """
    _headers: Incomplete
    def __init__(self, headers) -> None:
        """Make a MockResponse for `cookiejar` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
    def info(self): ...
    def getheaders(self, name) -> None: ...

def extract_cookies_to_jar(jar, request, response) -> None:
    """Extract the cookies from the response into a CookieJar.

    :param jar: http.cookiejar.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
def get_cookie_header(jar, request):
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
def remove_cookie_by_name(cookiejar, name, domain: Incomplete | None = None, path: Incomplete | None = None) -> None:
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """

class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """

class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a http.cookiejar.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """
    def get(self, name, default: Incomplete | None = None, domain: Incomplete | None = None, path: Incomplete | None = None):
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
    def set(self, name, value, **kwargs):
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
    def iterkeys(self) -> Generator[Incomplete]:
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
    def keys(self):
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
    def itervalues(self) -> Generator[Incomplete]:
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
    def values(self):
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
    def iteritems(self) -> Generator[Incomplete]:
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
    def items(self):
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
    def list_domains(self):
        """Utility method to list all the domains in the jar."""
    def list_paths(self):
        """Utility method to list all the paths in the jar."""
    def multiple_domains(self):
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
    def get_dict(self, domain: Incomplete | None = None, path: Incomplete | None = None):
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
    def __contains__(self, name) -> bool: ...
    def __getitem__(self, name):
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

        .. warning:: operation is O(n), not O(1).
        """
    def __setitem__(self, name, value) -> None:
        """Dict-like __setitem__ for compatibility with client code. Throws
        exception if there is already a cookie of that name in the jar. In that
        case, use the more explicit set() method instead.
        """
    def __delitem__(self, name) -> None:
        """Deletes a cookie given a name. Wraps ``http.cookiejar.CookieJar``'s
        ``remove_cookie_by_name()``.
        """
    def set_cookie(self, cookie, *args, **kwargs): ...
    def update(self, other) -> None:
        """Updates this jar with cookies from another CookieJar or dict-like"""
    def _find(self, name, domain: Incomplete | None = None, path: Incomplete | None = None):
        """Requests uses this method internally to get cookie values.

        If there are conflicting cookies, _find arbitrarily chooses one.
        See _find_no_duplicates if you want an exception thrown if there are
        conflicting cookies.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :return: cookie.value
        """
    def _find_no_duplicates(self, name, domain: Incomplete | None = None, path: Incomplete | None = None):
        """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
    def __getstate__(self):
        """Unlike a normal CookieJar, this class is pickleable."""
    _cookies_lock: Incomplete
    def __setstate__(self, state) -> None:
        """Unlike a normal CookieJar, this class is pickleable."""
    def copy(self):
        """Return a copy of this RequestsCookieJar."""
    def get_policy(self):
        """Return the CookiePolicy instance used."""

def _copy_cookie_jar(jar): ...
def create_cookie(name, value, **kwargs):
    '''Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain \'\'
    and sent on every request (this is sometimes called a "supercookie").
    '''
def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""
def cookiejar_from_dict(cookie_dict, cookiejar: Incomplete | None = None, overwrite: bool = True):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    :rtype: CookieJar
    """
def merge_cookies(cookiejar, cookies):
    """Add cookies to cookiejar and returns a merged CookieJar.

    :param cookiejar: CookieJar object to add the cookies to.
    :param cookies: Dictionary or CookieJar object to be added.
    :rtype: CookieJar
    """

"""
requests.adapters
~~~~~~~~~~~~~~~~~

This module contains the transport adapters that Requests uses to define
and maintain connections.
"""
from .auth import _basic_auth_str as _basic_auth_str
from .compat import basestring as basestring, urlparse as urlparse
from .cookies import extract_cookies_to_jar as extract_cookies_to_jar
from .exceptions import ConnectTimeout as ConnectTimeout, ConnectionError as ConnectionError, InvalidHeader as InvalidHeader, InvalidProxyURL as InvalidProxyURL, InvalidSchema as InvalidSchema, InvalidURL as InvalidURL, ProxyError as ProxyError, ReadTimeout as ReadTimeout, RetryError as RetryError, SSLError as SSLError
from .models import PreparedRequest as PreparedRequest, Response as Response
from .structures import CaseInsensitiveDict as CaseInsensitiveDict
from .utils import DEFAULT_CA_BUNDLE_PATH as DEFAULT_CA_BUNDLE_PATH, extract_zipped_paths as extract_zipped_paths, get_auth_from_url as get_auth_from_url, get_encoding_from_headers as get_encoding_from_headers, prepend_scheme_if_needed as prepend_scheme_if_needed, select_proxy as select_proxy, urldefragauth as urldefragauth
from _typeshed import Incomplete
from urllib3.poolmanager import PoolManager

DEFAULT_POOLBLOCK: bool
DEFAULT_POOLSIZE: int
DEFAULT_RETRIES: int
DEFAULT_POOL_TIMEOUT: Incomplete
_preloaded_ssl_context: Incomplete

def _urllib3_request_context(request: PreparedRequest, verify: bool | str | None, client_cert: tuple[str, str] | str | None, poolmanager: PoolManager) -> '(typing.Dict[str, typing.Any], typing.Dict[str, typing.Any])': ...

class BaseAdapter:
    """The Base Transport Adapter"""
    def __init__(self) -> None: ...
    def send(self, request, stream: bool = False, timeout: Incomplete | None = None, verify: bool = True, cert: Incomplete | None = None, proxies: Incomplete | None = None) -> None:
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        """
    def close(self) -> None:
        """Cleans up adapter specific items."""

class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3.

    Provides a general-case interface for Requests sessions to contact HTTP and
    HTTPS urls by implementing the Transport Adapter interface. This class will
    usually be created by the :class:`Session <Session>` class under the
    covers.

    :param pool_connections: The number of urllib3 connection pools to cache.
    :param pool_maxsize: The maximum number of connections to save in the pool.
    :param max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups, socket
        connections and connection timeouts, never to requests where data has
        made it to the server. By default, Requests does not retry failed
        connections. If you need granular control over the conditions under
        which we retry a request, import urllib3's ``Retry`` class and pass
        that instead.
    :param pool_block: Whether the connection pool should block for connections.

    Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> a = requests.adapters.HTTPAdapter(max_retries=3)
      >>> s.mount('http://', a)
    """
    __attrs__: Incomplete
    max_retries: Incomplete
    config: Incomplete
    proxy_manager: Incomplete
    _pool_connections: Incomplete
    _pool_maxsize: Incomplete
    _pool_block: Incomplete
    def __init__(self, pool_connections=10, pool_maxsize=10, max_retries=0, pool_block=False) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    poolmanager: Incomplete
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs) -> None:
        """Initializes a urllib3 PoolManager.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param connections: The number of urllib3 connection pools to cache.
        :param maxsize: The maximum number of connections to save in the pool.
        :param block: Block when no free connections are available.
        :param pool_kwargs: Extra keyword arguments used to initialize the Pool Manager.
        """
    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Return urllib3 ProxyManager for the given proxy.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The proxy to return a urllib3 ProxyManager for.
        :param proxy_kwargs: Extra keyword arguments used to configure the Proxy Manager.
        :returns: ProxyManager
        :rtype: urllib3.ProxyManager
        """
    def cert_verify(self, conn, url, verify, cert) -> None:
        """Verify a SSL certificate. This method should not be called from user
        code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param conn: The urllib3 connection object associated with the cert.
        :param url: The requested URL.
        :param verify: Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: The SSL certificate to verify.
        """
    def build_response(self, req, resp):
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response. This should not be called from user code, and is only exposed
        for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`

        :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
        :param resp: The urllib3 response object.
        :rtype: requests.Response
        """
    def build_connection_pool_key_attributes(self, request, verify, cert: Incomplete | None = None):
        '''Build the PoolKey attributes used by urllib3 to return a connection.

        This looks at the PreparedRequest, the user-specified verify value,
        and the value of the cert parameter to determine what PoolKey values
        to use to select a connection from a given urllib3 Connection Pool.

        The SSL related pool key arguments are not consistently set. As of
        this writing, use the following to determine what keys may be in that
        dictionary:

        * If ``verify`` is ``True``, ``"ssl_context"`` will be set and will be the
          default Requests SSL Context
        * If ``verify`` is ``False``, ``"ssl_context"`` will not be set but
          ``"cert_reqs"`` will be set
        * If ``verify`` is a string, (i.e., it is a user-specified trust bundle)
          ``"ca_certs"`` will be set if the string is not a directory recognized
          by :py:func:`os.path.isdir`, otherwise ``"ca_certs_dir"`` will be
          set.
        * If ``"cert"`` is specified, ``"cert_file"`` will always be set. If
          ``"cert"`` is a tuple with a second item, ``"key_file"`` will also
          be present

        To override these settings, one may subclass this class, call this
        method and use the above logic to change parameters as desired. For
        example, if one wishes to use a custom :py:class:`ssl.SSLContext` one
        must both set ``"ssl_context"`` and based on what else they require,
        alter the other keys to ensure the desired behaviour.

        :param request:
            The PreparedReqest being sent over the connection.
        :type request:
            :class:`~requests.models.PreparedRequest`
        :param verify:
            Either a boolean, in which case it controls whether
            we verify the server\'s TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use.
        :param cert:
            (optional) Any user-provided SSL certificate for client
            authentication (a.k.a., mTLS). This may be a string (i.e., just
            the path to a file which holds both certificate and key) or a
            tuple of length 2 with the certificate file path and key file
            path.
        :returns:
            A tuple of two dictionaries. The first is the "host parameters"
            portion of the Pool Key including scheme, hostname, and port. The
            second is a dictionary of SSLContext related parameters.
        '''
    def get_connection_with_tls_context(self, request, verify, proxies: Incomplete | None = None, cert: Incomplete | None = None):
        """Returns a urllib3 connection for the given request and TLS settings.
        This should not be called from user code, and is only exposed for use
        when subclassing the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request:
            The :class:`PreparedRequest <PreparedRequest>` object to be sent
            over the connection.
        :param verify:
            Either a boolean, in which case it controls whether we verify the
            server's TLS certificate, or a string, in which case it must be a
            path to a CA bundle to use.
        :param proxies:
            (optional) The proxies dictionary to apply to the request.
        :param cert:
            (optional) Any user-provided SSL certificate to be used for client
            authentication (a.k.a., mTLS).
        :rtype:
            urllib3.ConnectionPool
        """
    def get_connection(self, url, proxies: Incomplete | None = None):
        """DEPRECATED: Users should move to `get_connection_with_tls_context`
        for all subclasses of HTTPAdapter using Requests>=2.32.2.

        Returns a urllib3 connection for the given URL. This should not be
        called from user code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param url: The URL to connect to.
        :param proxies: (optional) A Requests-style dictionary of proxies used on this request.
        :rtype: urllib3.ConnectionPool
        """
    def close(self) -> None:
        """Disposes of any internal state.

        Currently, this closes the PoolManager and any active ProxyManager,
        which closes any pooled connections.
        """
    def request_url(self, request, proxies):
        """Obtain the url to use when making the final request.

        If the message is being sent through a HTTP proxy, the full URL has to
        be used. Otherwise, we should only use the path portion of the URL.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs.
        :rtype: str
        """
    def add_headers(self, request, **kwargs) -> None:
        """Add any headers needed by the connection. As of v2.0 this does
        nothing by default, but is left for overriding by users that subclass
        the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` to add headers to.
        :param kwargs: The keyword arguments from the call to send().
        """
    def proxy_headers(self, proxy):
        """Returns a dictionary of the headers to add to any request sent
        through a proxy. This works with urllib3 magic to ensure that they are
        correctly sent to the proxy, rather than in a tunnelled request if
        CONNECT is being used.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The url of the proxy being used for this request.
        :rtype: dict
        """
    def send(self, request, stream: bool = False, timeout: Incomplete | None = None, verify: bool = True, cert: Incomplete | None = None, proxies: Incomplete | None = None):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple or urllib3 Timeout object
        :param verify: (optional) Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        :rtype: requests.Response
        """

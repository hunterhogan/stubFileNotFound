from _typeshed import Incomplete
from typing import Any

basestring = str

def unicode(_str: Any, _charset: Any) -> Any: ...

class MessageEncodeError(Exception): ...

class Envelope:
    """
    The Envelope class.

    **Address formats**

    The following formats are supported for e-mail addresses:

    * ``"user@server.com"`` - just the e-mail address part as a string,
    * ``"Some User <user@server.com>"`` - name and e-mail address parts as a string,
    * ``("user@server.com", "Some User")`` - e-mail address and name parts as a tuple.

    Whenever you come to manipulate addresses feel free to use any (or all) of
    the formats above.

    :param to_addr: ``To`` address or list of ``To`` addresses
    :param from_addr: ``From`` address
    :param subject: message subject
    :param html_body: optional HTML part of the message
    :param text_body: optional plain text part of the message
    :param cc_addr: optional single CC address or list of CC addresses
    :param bcc_addr: optional single BCC address or list of BCC addresses
    :param headers: optional dictionary of headers
    :param charset: message charset
    """

    ADDR_FORMAT: str
    ADDR_REGEXP: Incomplete
    _to: Incomplete
    _from: Incomplete
    _subject: Incomplete
    _parts: Incomplete
    _cc: Incomplete
    _bcc: Incomplete
    _headers: Incomplete
    _charset: Incomplete
    _addr_format: Incomplete
    def __init__(self, to_addr: Any=None, from_addr: Any=None, subject: Any=None, html_body: Any=None, text_body: Any=None, cc_addr: Any=None, bcc_addr: Any=None, headers: Any=None, charset: str = 'utf-8') -> None: ...
    @property
    def to_addr(self) -> Any:
        """List of ``To`` addresses."""
    def add_to_addr(self, to_addr: Any) -> None:
        """Adds a ``To`` address."""
    def clear_to_addr(self) -> None:
        """Clears list of ``To`` addresses."""
    @property
    def from_addr(self) -> Any: ...
    @from_addr.setter
    def from_addr(self, from_addr: Any) -> None: ...
    @property
    def cc_addr(self) -> Any:
        """List of CC addresses."""
    def add_cc_addr(self, cc_addr: Any) -> None:
        """Adds a CC address."""
    def clear_cc_addr(self) -> None:
        """Clears list of CC addresses."""
    @property
    def bcc_addr(self) -> Any:
        """List of BCC addresses."""
    def add_bcc_addr(self, bcc_addr: Any) -> None:
        """Adds a BCC address."""
    def clear_bcc_addr(self) -> None:
        """Clears list of BCC addresses."""
    @property
    def charset(self) -> Any:
        """Message charset."""
    @charset.setter
    def charset(self, charset: Any) -> None: ...
    def _addr_tuple_to_addr(self, addr_tuple: Any) -> Any: ...
    @property
    def headers(self) -> Any:
        """Dictionary of custom headers."""
    def add_header(self, key: Any, value: Any) -> None:
        """Adds a custom header."""
    def clear_headers(self) -> None:
        """Clears custom headers."""
    def _addrs_to_header(self, addrs: Any) -> Any: ...
    def _raise(self, exc_class: Any, message: Any) -> None: ...
    def _header(self, _str: Any) -> Any: ...
    def _is_ascii(self, _str: Any) -> Any: ...
    def _encoded(self, _str: Any) -> Any: ...
    def to_mime_message(self) -> Any:
        """Returns the envelope as
        :py:class:`email.mime.multipart.MIMEMultipart`.
        """
    def add_attachment(self, file_path: Any, mimetype: Any=None) -> None:
        """Attaches a file located at *file_path* to the envelope. If
        *mimetype* is not specified an attempt to guess it is made. If nothing
        is guessed then `application/octet-stream` is used.
        """
    def send(self, *args: Any, **kwargs: Any) -> Any:
        """Sends the envelope using a freshly created SMTP connection. *args*
        and *kwargs* are passed directly to :py:class:`envelopes.conn.SMTP`
        constructor.

        Returns a tuple of SMTP object and whatever its send method returns.
        """




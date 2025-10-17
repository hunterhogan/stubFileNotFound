from _typeshed import Incomplete
from typing import Any

__all__ = ['SMTP', 'GMailSMTP', 'MailcatcherSMTP', 'SendGridSMTP', 'TimeoutException']

TimeoutException: Incomplete

class SMTP:
    """Wrapper around :py:class:`smtplib.SMTP` class."""

    _conn: Incomplete
    _host: Incomplete
    _port: Incomplete
    _login: Incomplete
    _password: Incomplete
    _tls: Incomplete
    _timeout: Incomplete
    def __init__(self, host: Any=None, port: int = 25, login: Any=None, password: Any=None, tls: bool = False, timeout: Any=None) -> None: ...
    @property
    def is_connected(self) -> Any:
        """Returns *True* if the SMTP connection is initialized and
        connected. Otherwise returns *False*.
        """
    def _connect(self, replace_current: bool = False) -> None: ...
    def send(self, envelope: Any) -> Any:
        """Sends an *envelope*."""

class GMailSMTP(SMTP):
    """Subclass of :py:class:`SMTP` preconfigured for GMail SMTP."""

    GMAIL_SMTP_HOST: str
    GMAIL_SMTP_TLS: bool
    def __init__(self, login: Any=None, password: Any=None) -> None: ...

class SendGridSMTP(SMTP):
    """Subclass of :py:class:`SMTP` preconfigured for SendGrid SMTP."""

    SENDGRID_SMTP_HOST: str
    SENDGRID_SMTP_PORT: int
    SENDGRID_SMTP_TLS: bool
    def __init__(self, login: Any=None, password: Any=None) -> None: ...

class MailcatcherSMTP(SMTP):
    """Subclass of :py:class:`SMTP` preconfigured for local Mailcatcher
    SMTP.
    """

    MAILCATCHER_SMTP_HOST: str
    def __init__(self, port: int = 1025) -> None: ...




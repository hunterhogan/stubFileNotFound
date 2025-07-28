from _typeshed import Incomplete
from redis.exceptions import AuthorizationError as AuthorizationError, ConnectionError as ConnectionError

def _verify_response(issuer_cert, ocsp_response) -> None: ...
def _check_certificate(issuer_cert, ocsp_bytes, validate: bool = True):
    """A wrapper the return the validity of a known ocsp certificate"""
def _get_certificates(certs, issuer_cert, responder_name, responder_hash): ...
def _get_pubkey_hash(certificate): ...
def ocsp_staple_verifier(con, ocsp_bytes, expected: Incomplete | None = None):
    """An implementation of a function for set_ocsp_client_callback in PyOpenSSL.

    This function validates that the provide ocsp_bytes response is valid,
    and matches the expected, stapled responses.
    """

class OCSPVerifier:
    """A class to verify ssl sockets for RFC6960/RFC6961. This can be used
    when using direct validation of OCSP responses and certificate revocations.

    @see https://datatracker.ietf.org/doc/html/rfc6960
    @see https://datatracker.ietf.org/doc/html/rfc6961
    """
    SOCK: Incomplete
    HOST: Incomplete
    PORT: Incomplete
    CA_CERTS: Incomplete
    def __init__(self, sock, host, port, ca_certs: Incomplete | None = None) -> None: ...
    def _bin2ascii(self, der):
        """Convert SSL certificates in a binary (DER) format to ASCII PEM."""
    def components_from_socket(self):
        """This function returns the certificate, primary issuer, and primary ocsp
        server in the chain for a socket already wrapped with ssl.
        """
    def _certificate_components(self, cert):
        """Given an SSL certificate, retract the useful components for
        validating the certificate status with an OCSP server.

        Args:
            cert ([bytes]): A PEM encoded ssl certificate
        """
    def components_from_direct_connection(self):
        """Return the certificate, primary issuer, and primary ocsp server
        from the host defined by the socket. This is useful in cases where
        different certificates are occasionally presented.
        """
    def build_certificate_url(self, server, cert, issuer_cert):
        """Return the complete url to the ocsp"""
    def check_certificate(self, server, cert, issuer_url):
        """Checks the validity of an ocsp server for an issuer"""
    def is_valid(self):
        """Returns the validity of the certificate wrapping our socket.
        This first retrieves for validate the certificate, issuer_url,
        and ocsp_server for certificate validate. Then retrieves the
        issuer certificate from the issuer_url, and finally checks
        the validity of OCSP revocation status.
        """

import enum as _enum

__all__ = ['authenticate_service_account', 'authenticate_user']

class _CredentialType(_enum.Enum):
    NO_CHECK = 0
    USER = 1
    SERVICE_ACCOUNT = 2

def authenticate_user(clear_output: bool = True, project_id: str | None = None) -> None: ...
def authenticate_service_account(clear_output: bool = True) -> None: ...

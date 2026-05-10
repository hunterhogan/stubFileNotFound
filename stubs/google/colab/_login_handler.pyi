from jupyter_server.auth import login

class ColabLoginHandler(login.LoginHandler):
    @classmethod
    def validate_security(cls, *args, **kwargs) -> None: ...

import tornado
from google.colab import drive as drive
from jupyter_server.base import handlers

class ResourceUsageHandler(handlers.APIHandler):
    def initialize(self, kernel_manager) -> None: ...
    @tornado.web.authenticated
    def get(self, *unused_args, **unused_kwargs) -> None: ...

class DriveHandler(handlers.APIHandler):
    @tornado.web.authenticated
    def get(self, *unused_args, **unused_kwargs) -> None: ...

class BuildInfoHandler(handlers.APIHandler):
    @tornado.web.authenticated
    def get(self, *unused_args, **unused_kwargs) -> None: ...

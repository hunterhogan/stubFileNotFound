from typing import Final

from pika.adapters import (
    BaseConnection as BaseConnection,
    BlockingConnection as BlockingConnection,
    SelectConnection as SelectConnection,
)
from pika.connection import ConnectionParameters as ConnectionParameters, SSLOptions as SSLOptions, URLParameters as URLParameters

__version__: Final[str]

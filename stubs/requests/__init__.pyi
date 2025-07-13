import warnings
import urllib3
import ssl
import logging
from .exceptions import ConnectTimeout, ConnectionError, FileModeWarning, HTTPError, JSONDecodeError, ReadTimeout, RequestException, RequestsDependencyWarning, Timeout, TooManyRedirects, URLRequired
from charset_normalizer import __version__ as charset_normalizer_version
from chardet import __version__ as chardet_version
from urllib3.contrib import pyopenssl
from cryptography import __version__ as cryptography_version
from urllib3.exceptions import DependencyWarning
from logging import NullHandler
from . import packages as packages, utils as utils
from .__version__ import __author__ as __author__, __author_email__ as __author_email__, __build__ as __build__, __cake__ as __cake__, __copyright__ as __copyright__, __description__ as __description__, __license__ as __license__, __title__ as __title__, __url__ as __url__, __version__ as __version__
from .api import delete as delete, get as get, head as head, options as options, post as post, put as put, request as request
from .exceptions import ConnectTimeout as ConnectTimeout, ConnectionError as ConnectionError, HTTPError as HTTPError, JSONDecodeError as JSONDecodeError, ReadTimeout as ReadTimeout, RequestException as RequestException, Timeout as Timeout, TooManyRedirects as TooManyRedirects, URLRequired as URLRequired
from .models import PreparedRequest as PreparedRequest, Request as Request, Response as Response
from .sessions import Session as Session, session as session
from .status_codes import codes as codes

"""
Requests HTTP Library
~~~~~~~~~~~~~~~~~~~~~

Requests is an HTTP library, written in Python, for human beings.
Basic GET usage:

   >>> import requests
   >>> r = requests.get('https://www.python.org')
   >>> r.status_code
   200
   >>> b'Python is a programming language' in r.content
   True

... or POST:

   >>> payload = dict(key1='value1', key2='value2')
   >>> r = requests.post('https://httpbin.org/post', data=payload)
   >>> print(r.text)
   {
     ...
     "form": {
       "key1": "value1",
       "key2": "value2"
     },
     ...
   }

The other HTTP methods are supported - see `requests.api`. Full documentation
is at <https://requests.readthedocs.io>.

:copyright: (c) 2017 by Kenneth Reitz.
:license: Apache 2.0, see LICENSE for more details.
"""
def check_compatibility(urllib3_version, chardet_version, charset_normalizer_version) -> None: ...
def _check_cryptography(cryptography_version) -> None: ...

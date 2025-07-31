from .core import ENOVAL as ENOVAL, args_to_key as args_to_key, full_name as full_name
from .fanout import FanoutCache as FanoutCache
from _typeshed import Incomplete
from django.core.cache.backends.base import BaseCache

class DjangoCache(BaseCache):
    """Django-compatible disk and file backed cache."""
    _cache: Incomplete
    def __init__(self, directory, params) -> None:
        """Initialize DjangoCache instance.

        :param str directory: cache directory
        :param dict params: cache parameters

        """
    @property
    def directory(self):
        """Cache directory."""
    def cache(self, name):
        """Return Cache with given `name` in subdirectory.

        :param str name: subdirectory name for Cache
        :return: Cache with given name

        """
    def deque(self, name, maxlen=None):
        """Return Deque with given `name` in subdirectory.

        :param str name: subdirectory name for Deque
        :param maxlen: max length (default None, no max)
        :return: Deque with given name

        """
    def index(self, name):
        """Return Index with given `name` in subdirectory.

        :param str name: subdirectory name for Index
        :return: Index with given name

        """
    def add(self, key, value, timeout=..., version=None, read: bool = False, tag=None, retry: bool = True):
        """Set a value in the cache if the key does not already exist. If
        timeout is given, that timeout will be used for the key; otherwise the
        default cache timeout will be used.

        Return True if the value was stored, False otherwise.

        :param key: key for item
        :param value: value for item
        :param float timeout: seconds until the item expires
            (default 300 seconds)
        :param int version: key version number (default None, cache parameter)
        :param bool read: read value as bytes from file (default False)
        :param str tag: text to associate with key (default None)
        :param bool retry: retry if database timeout occurs (default True)
        :return: True if item was added

        """
    def get(self, key, default=None, version=None, read: bool = False, expire_time: bool = False, tag: bool = False, retry: bool = False):
        """Fetch a given key from the cache. If the key does not exist, return
        default, which itself defaults to None.

        :param key: key for item
        :param default: return value if key is missing (default None)
        :param int version: key version number (default None, cache parameter)
        :param bool read: if True, return file handle to value
            (default False)
        :param float expire_time: if True, return expire_time in tuple
            (default False)
        :param tag: if True, return tag in tuple (default False)
        :param bool retry: retry if database timeout occurs (default False)
        :return: value for item if key is found else default

        """
    def read(self, key, version=None):
        """Return file handle corresponding to `key` from Cache.

        :param key: Python key to retrieve
        :param int version: key version number (default None, cache parameter)
        :return: file open for reading in binary mode
        :raises KeyError: if key is not found

        """
    def set(self, key, value, timeout=..., version=None, read: bool = False, tag=None, retry: bool = True):
        """Set a value in the cache. If timeout is given, that timeout will be
        used for the key; otherwise the default cache timeout will be used.

        :param key: key for item
        :param value: value for item
        :param float timeout: seconds until the item expires
            (default 300 seconds)
        :param int version: key version number (default None, cache parameter)
        :param bool read: read value as bytes from file (default False)
        :param str tag: text to associate with key (default None)
        :param bool retry: retry if database timeout occurs (default True)
        :return: True if item was set

        """
    def touch(self, key, timeout=..., version=None, retry: bool = True):
        """Touch a key in the cache. If timeout is given, that timeout will be
        used for the key; otherwise the default cache timeout will be used.

        :param key: key for item
        :param float timeout: seconds until the item expires
            (default 300 seconds)
        :param int version: key version number (default None, cache parameter)
        :param bool retry: retry if database timeout occurs (default True)
        :return: True if key was touched

        """
    def pop(self, key, default=None, version=None, expire_time: bool = False, tag: bool = False, retry: bool = True):
        """Remove corresponding item for `key` from cache and return value.

        If `key` is missing, return `default`.

        Operation is atomic. Concurrent operations will be serialized.

        :param key: key for item
        :param default: return value if key is missing (default None)
        :param int version: key version number (default None, cache parameter)
        :param float expire_time: if True, return expire_time in tuple
            (default False)
        :param tag: if True, return tag in tuple (default False)
        :param bool retry: retry if database timeout occurs (default True)
        :return: value for item if key is found else default

        """
    def delete(self, key, version=None, retry: bool = True):
        """Delete a key from the cache, failing silently.

        :param key: key for item
        :param int version: key version number (default None, cache parameter)
        :param bool retry: retry if database timeout occurs (default True)
        :return: True if item was deleted

        """
    def incr(self, key, delta: int = 1, version=None, default=None, retry: bool = True):
        """Increment value by delta for item with key.

        If key is missing and default is None then raise KeyError. Else if key
        is missing and default is not None then use default for value.

        Operation is atomic. All concurrent increment operations will be
        counted individually.

        Assumes value may be stored in a SQLite column. Most builds that target
        machines with 64-bit pointer widths will support 64-bit signed
        integers.

        :param key: key for item
        :param int delta: amount to increment (default 1)
        :param int version: key version number (default None, cache parameter)
        :param int default: value if key is missing (default None)
        :param bool retry: retry if database timeout occurs (default True)
        :return: new value for item on success else None
        :raises ValueError: if key is not found and default is None

        """
    def decr(self, key, delta: int = 1, version=None, default=None, retry: bool = True):
        """Decrement value by delta for item with key.

        If key is missing and default is None then raise KeyError. Else if key
        is missing and default is not None then use default for value.

        Operation is atomic. All concurrent decrement operations will be
        counted individually.

        Unlike Memcached, negative values are supported. Value may be
        decremented below zero.

        Assumes value may be stored in a SQLite column. Most builds that target
        machines with 64-bit pointer widths will support 64-bit signed
        integers.

        :param key: key for item
        :param int delta: amount to decrement (default 1)
        :param int version: key version number (default None, cache parameter)
        :param int default: value if key is missing (default None)
        :param bool retry: retry if database timeout occurs (default True)
        :return: new value for item on success else None
        :raises ValueError: if key is not found and default is None

        """
    def has_key(self, key, version=None):
        """Returns True if the key is in the cache and has not expired.

        :param key: key for item
        :param int version: key version number (default None, cache parameter)
        :return: True if key is found

        """
    def expire(self):
        """Remove expired items from cache.

        :return: count of items removed

        """
    def stats(self, enable: bool = True, reset: bool = False):
        """Return cache statistics hits and misses.

        :param bool enable: enable collecting statistics (default True)
        :param bool reset: reset hits and misses to 0 (default False)
        :return: (hits, misses)

        """
    def create_tag_index(self) -> None:
        """Create tag index on cache database.

        Better to initialize cache with `tag_index=True` than use this.

        :raises Timeout: if database timeout occurs

        """
    def drop_tag_index(self) -> None:
        """Drop tag index on cache database.

        :raises Timeout: if database timeout occurs

        """
    def evict(self, tag):
        """Remove items with matching `tag` from cache.

        :param str tag: tag identifying items
        :return: count of items removed

        """
    def cull(self):
        """Cull items from cache until volume is less than size limit.

        :return: count of items removed

        """
    def clear(self):
        """Remove *all* values from the cache at once."""
    def close(self, **kwargs) -> None:
        """Close the cache connection."""
    def get_backend_timeout(self, timeout=...):
        """Return seconds to expiration.

        :param float timeout: seconds until the item expires
            (default 300 seconds)

        """
    def memoize(self, name=None, timeout=..., version=None, typed: bool = False, tag=None, ignore=()):
        """Memoizing cache decorator.

        Decorator to wrap callable with memoizing function using cache.
        Repeated calls with the same arguments will lookup result in cache and
        avoid function evaluation.

        If name is set to None (default), the callable name will be determined
        automatically.

        When timeout is set to zero, function results will not be set in the
        cache. Cache lookups still occur, however. Read
        :doc:`case-study-landing-page-caching` for example usage.

        If typed is set to True, function arguments of different types will be
        cached separately. For example, f(3) and f(3.0) will be treated as
        distinct calls with distinct results.

        The original underlying function is accessible through the __wrapped__
        attribute. This is useful for introspection, for bypassing the cache,
        or for rewrapping the function with a different cache.

        An additional `__cache_key__` attribute can be used to generate the
        cache key used for the given arguments.

        Remember to call memoize when decorating a callable. If you forget,
        then a TypeError will occur.

        :param str name: name given for callable (default None, automatic)
        :param float timeout: seconds until the item expires
            (default 300 seconds)
        :param int version: key version number (default None, cache parameter)
        :param bool typed: cache different types separately (default False)
        :param str tag: text to associate with arguments (default None)
        :param set ignore: positional or keyword args to ignore (default ())
        :return: callable decorator

        """

from .api import PlatformDirsABC
from collections.abc import Iterator
from pathlib import Path

__all__ = ['Unix']

class Unix(PlatformDirsABC):
    """
    On Unix/Linux, we follow the `XDG Basedir Spec <https://specifications.freedesktop.org/basedir-spec/basedir-spec-
    latest.html>`_.

    The spec allows overriding directories with environment variables. The examples shown are the default values,
    alongside the name of the environment variable that overrides them. Makes use of the `appname
    <platformdirs.api.PlatformDirsABC.appname>`, `version <platformdirs.api.PlatformDirsABC.version>`, `multipath
    <platformdirs.api.PlatformDirsABC.multipath>`, `opinion <platformdirs.api.PlatformDirsABC.opinion>`, `ensure_exists
    <platformdirs.api.PlatformDirsABC.ensure_exists>`.

    """
    @property
    def user_data_dir(self) -> str:
        """
        :return: data directory tied to the user, e.g. ``~/.local/share/$appname/$version`` or
         ``$XDG_DATA_HOME/$appname/$version``
        """
    @property
    def _site_data_dirs(self) -> list[str]: ...
    @property
    def site_data_dir(self) -> str:
        """
        :return: data directories shared by users (if `multipath <platformdirs.api.PlatformDirsABC.multipath>` is
         enabled and ``XDG_DATA_DIRS`` is set and a multi path the response is also a multi path separated by the
         OS path separator), e.g. ``/usr/local/share/$appname/$version`` or ``/usr/share/$appname/$version``
        """
    @property
    def user_config_dir(self) -> str:
        """
        :return: config directory tied to the user, e.g. ``~/.config/$appname/$version`` or
         ``$XDG_CONFIG_HOME/$appname/$version``
        """
    @property
    def _site_config_dirs(self) -> list[str]: ...
    @property
    def site_config_dir(self) -> str:
        """
        :return: config directories shared by users (if `multipath <platformdirs.api.PlatformDirsABC.multipath>`
         is enabled and ``XDG_CONFIG_DIRS`` is set and a multi path the response is also a multi path separated by
         the OS path separator), e.g. ``/etc/xdg/$appname/$version``
        """
    @property
    def user_cache_dir(self) -> str:
        """
        :return: cache directory tied to the user, e.g. ``~/.cache/$appname/$version`` or
         ``~/$XDG_CACHE_HOME/$appname/$version``
        """
    @property
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users, e.g. ``/var/cache/$appname/$version``"""
    @property
    def user_state_dir(self) -> str:
        """
        :return: state directory tied to the user, e.g. ``~/.local/state/$appname/$version`` or
         ``$XDG_STATE_HOME/$appname/$version``
        """
    @property
    def user_log_dir(self) -> str:
        """:return: log directory tied to the user, same as `user_state_dir` if not opinionated else ``log`` in it"""
    @property
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user, e.g. ``~/Documents``"""
    @property
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user, e.g. ``~/Downloads``"""
    @property
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user, e.g. ``~/Pictures``"""
    @property
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user, e.g. ``~/Videos``"""
    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user, e.g. ``~/Music``"""
    @property
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user, e.g. ``~/Desktop``"""
    @property
    def user_runtime_dir(self) -> str:
        """
        :return: runtime directory tied to the user, e.g. ``/run/user/$(id -u)/$appname/$version`` or
         ``$XDG_RUNTIME_DIR/$appname/$version``.

         For FreeBSD/OpenBSD/NetBSD, it would return ``/var/run/user/$(id -u)/$appname/$version`` if
         exists, otherwise ``/tmp/runtime-$(id -u)/$appname/$version``, if``$XDG_RUNTIME_DIR``
         is not set.
        """
    @property
    def site_runtime_dir(self) -> str:
        """
        :return: runtime directory shared by users, e.g. ``/run/$appname/$version`` or         ``$XDG_RUNTIME_DIR/$appname/$version``.

        Note that this behaves almost exactly like `user_runtime_dir` if ``$XDG_RUNTIME_DIR`` is set, but will
        fall back to paths associated to the root user instead of a regular logged-in user if it's not set.

        If you wish to ensure that a logged-in root user path is returned e.g. ``/run/user/0``, use `user_runtime_dir`
        instead.

        For FreeBSD/OpenBSD/NetBSD, it would return ``/var/run/$appname/$version`` if ``$XDG_RUNTIME_DIR`` is not set.
        """
    @property
    def site_data_path(self) -> Path:
        """:return: data path shared by users. Only return the first item, even if ``multipath`` is set to ``True``"""
    @property
    def site_config_path(self) -> Path:
        """:return: config path shared by the users, returns the first item, even if ``multipath`` is set to ``True``"""
    @property
    def site_cache_path(self) -> Path:
        """:return: cache path shared by users. Only return the first item, even if ``multipath`` is set to ``True``"""
    def iter_config_dirs(self) -> Iterator[str]:
        """:yield: all user and site configuration directories."""
    def iter_data_dirs(self) -> Iterator[str]:
        """:yield: all user and site data directories."""

from .api import PlatformDirsABC
from pathlib import Path

__all__ = ['MacOS']

class MacOS(PlatformDirsABC):
    """
    Platform directories for the macOS operating system.

    Follows the guidance from
    `Apple documentation <https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/MacOSXDirectories/MacOSXDirectories.html>`_.
    Makes use of the `appname <platformdirs.api.PlatformDirsABC.appname>`,
    `version <platformdirs.api.PlatformDirsABC.version>`,
    `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.

    """
    @property
    def user_data_dir(self) -> str:
        """:return: data directory tied to the user, e.g. ``~/Library/Application Support/$appname/$version``"""
    @property
    def site_data_dir(self) -> str:
        '''
        :return: data directory shared by users, e.g. ``/Library/Application Support/$appname/$version``.
          If we\'re using a Python binary managed by `Homebrew <https://brew.sh>`_, the directory
          will be under the Homebrew prefix, e.g. ``/opt/homebrew/share/$appname/$version``.
          If `multipath <platformdirs.api.PlatformDirsABC.multipath>` is enabled, and we\'re in Homebrew,
          the response is a multi-path string separated by ":", e.g.
          ``/opt/homebrew/share/$appname/$version:/Library/Application Support/$appname/$version``
        '''
    @property
    def site_data_path(self) -> Path:
        """:return: data path shared by users. Only return the first item, even if ``multipath`` is set to ``True``"""
    @property
    def user_config_dir(self) -> str:
        """:return: config directory tied to the user, same as `user_data_dir`"""
    @property
    def site_config_dir(self) -> str:
        """:return: config directory shared by the users, same as `site_data_dir`"""
    @property
    def user_cache_dir(self) -> str:
        """:return: cache directory tied to the user, e.g. ``~/Library/Caches/$appname/$version``"""
    @property
    def site_cache_dir(self) -> str:
        '''
        :return: cache directory shared by users, e.g. ``/Library/Caches/$appname/$version``.
          If we\'re using a Python binary managed by `Homebrew <https://brew.sh>`_, the directory
          will be under the Homebrew prefix, e.g. ``/opt/homebrew/var/cache/$appname/$version``.
          If `multipath <platformdirs.api.PlatformDirsABC.multipath>` is enabled, and we\'re in Homebrew,
          the response is a multi-path string separated by ":", e.g.
          ``/opt/homebrew/var/cache/$appname/$version:/Library/Caches/$appname/$version``
        '''
    @property
    def site_cache_path(self) -> Path:
        """:return: cache path shared by users. Only return the first item, even if ``multipath`` is set to ``True``"""
    @property
    def user_state_dir(self) -> str:
        """:return: state directory tied to the user, same as `user_data_dir`"""
    @property
    def user_log_dir(self) -> str:
        """:return: log directory tied to the user, e.g. ``~/Library/Logs/$appname/$version``"""
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
        """:return: videos directory tied to the user, e.g. ``~/Movies``"""
    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user, e.g. ``~/Music``"""
    @property
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user, e.g. ``~/Desktop``"""
    @property
    def user_runtime_dir(self) -> str:
        """:return: runtime directory tied to the user, e.g. ``~/Library/Caches/TemporaryItems/$appname/$version``"""
    @property
    def site_runtime_dir(self) -> str:
        """:return: runtime directory shared by users, same as `user_runtime_dir`"""

from .api import PlatformDirsABC

__all__ = ['Android']

class Android(PlatformDirsABC):
    """
    Follows the guidance `from here <https://android.stackexchange.com/a/216132>`_.

    Makes use of the `appname <platformdirs.api.PlatformDirsABC.appname>`, `version
    <platformdirs.api.PlatformDirsABC.version>`, `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.

    """
    @property
    def user_data_dir(self) -> str:
        """:return: data directory tied to the user, e.g. ``/data/user/<userid>/<packagename>/files/<AppName>``"""
    @property
    def site_data_dir(self) -> str:
        """:return: data directory shared by users, same as `user_data_dir`"""
    @property
    def user_config_dir(self) -> str:
        """
        :return: config directory tied to the user, e.g.         ``/data/user/<userid>/<packagename>/shared_prefs/<AppName>``
        """
    @property
    def site_config_dir(self) -> str:
        """:return: config directory shared by the users, same as `user_config_dir`"""
    @property
    def user_cache_dir(self) -> str:
        """:return: cache directory tied to the user, e.g.,``/data/user/<userid>/<packagename>/cache/<AppName>``"""
    @property
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users, same as `user_cache_dir`"""
    @property
    def user_state_dir(self) -> str:
        """:return: state directory tied to the user, same as `user_data_dir`"""
    @property
    def user_log_dir(self) -> str:
        """
        :return: log directory tied to the user, same as `user_cache_dir` if not opinionated else ``log`` in it,
          e.g. ``/data/user/<userid>/<packagename>/cache/<AppName>/log``
        """
    @property
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user e.g. ``/storage/emulated/0/Documents``"""
    @property
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user e.g. ``/storage/emulated/0/Downloads``"""
    @property
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user e.g. ``/storage/emulated/0/Pictures``"""
    @property
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user e.g. ``/storage/emulated/0/DCIM/Camera``"""
    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user e.g. ``/storage/emulated/0/Music``"""
    @property
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user e.g. ``/storage/emulated/0/Desktop``"""
    @property
    def user_runtime_dir(self) -> str:
        """
        :return: runtime directory tied to the user, same as `user_cache_dir` if not opinionated else ``tmp`` in it,
          e.g. ``/data/user/<userid>/<packagename>/cache/<AppName>/tmp``
        """
    @property
    def site_runtime_dir(self) -> str:
        """:return: runtime directory shared by users, same as `user_runtime_dir`"""

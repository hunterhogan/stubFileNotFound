import abc
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

class PlatformDirsABC(ABC, metaclass=abc.ABCMeta):
    """Abstract base class for platform directories."""

    appname: str | None
    """The name of application."""

    appauthor: str | Literal[False] | None
    """
    The name of the app author or distributing body for this application.

    Typically, it is the owning company name. Defaults to appname. You may pass False to disable it.
    """

    version: str | None
    """
    An optional version path element to append to the path.

    You might want to use this if you want multiple versions of your app to be able to run independently.
    If used, this would typically be <major>.<minor>.
    """

    roaming: bool
    """
    Whether to use the roaming appdata directory on Windows.

    That means that for users on a Windows network setup for roaming profiles,
    this user data will be synced on login.
    """

    multipath: bool
    """
    An optional parameter which indicates that the entire list of data dirs should be returned.

    By default, the first item would only be returned.
    """

    opinion: bool
    """A flag to indicating to use opinionated values."""

    ensure_exists: bool
    """
    Optionally create the directory (and any missing parents) upon access if it does not exist.

    By default, no directories are created.
    """

    def __init__(self, appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, multipath: bool = False, opinion: bool = True, ensure_exists: bool = False) -> None:
        """
        Create a new platform directory.

        :param appname: The name of application.
        :param appauthor: The name of the app author or distributing body for this application.
                          Typically it is the owning company name. Defaults to appname. You may pass False to disable it.
        :param version: An optional version path element to append to the path.
                        You might want to use this if you want multiple versions of your app to be able to run independently.
        :param roaming: Whether to use the roaming appdata directory on Windows.
        :param multipath: An optional parameter which indicates that the entire list of data dirs should be returned.
        :param opinion: A flag to indicating to use opinionated values.
        :param ensure_exists: Optionally create the directory (and any missing parents) upon access if it does not exist.
        """

    def _append_app_name_and_version(self, *base: str) -> str: ...
    def _optionally_create_directory(self, path: str) -> None: ...
    def _first_item_as_path_if_multipath(self, directory: str) -> Path: ...
    @property
    @abstractmethod
    def user_data_dir(self) -> str:
        """:return: data directory tied to the user"""
    @property
    @abstractmethod
    def site_data_dir(self) -> str:
        """:return: data directory shared by users"""
    @property
    @abstractmethod
    def user_config_dir(self) -> str:
        """:return: config directory tied to the user"""
    @property
    @abstractmethod
    def site_config_dir(self) -> str:
        """:return: config directory shared by the users"""
    @property
    @abstractmethod
    def user_cache_dir(self) -> str:
        """:return: cache directory tied to the user"""
    @property
    @abstractmethod
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users"""
    @property
    @abstractmethod
    def user_state_dir(self) -> str:
        """:return: state directory tied to the user"""
    @property
    @abstractmethod
    def user_log_dir(self) -> str:
        """:return: log directory tied to the user"""
    @property
    @abstractmethod
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user"""
    @property
    @abstractmethod
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user"""
    @property
    @abstractmethod
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user"""
    @property
    @abstractmethod
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user"""
    @property
    @abstractmethod
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user"""
    @property
    @abstractmethod
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user"""
    @property
    @abstractmethod
    def user_runtime_dir(self) -> str:
        """:return: runtime directory tied to the user"""
    @property
    @abstractmethod
    def site_runtime_dir(self) -> str:
        """:return: runtime directory shared by users"""
    @property
    def user_data_path(self) -> Path:
        """:return: data path tied to the user"""
    @property
    def site_data_path(self) -> Path:
        """:return: data path shared by users"""
    @property
    def user_config_path(self) -> Path:
        """:return: config path tied to the user"""
    @property
    def site_config_path(self) -> Path:
        """:return: config path shared by the users"""
    @property
    def user_cache_path(self) -> Path:
        """:return: cache path tied to the user"""
    @property
    def site_cache_path(self) -> Path:
        """:return: cache path shared by users"""
    @property
    def user_state_path(self) -> Path:
        """:return: state path tied to the user"""
    @property
    def user_log_path(self) -> Path:
        """:return: log path tied to the user"""
    @property
    def user_documents_path(self) -> Path:
        """:return: documents a path tied to the user"""
    @property
    def user_downloads_path(self) -> Path:
        """:return: downloads path tied to the user"""
    @property
    def user_pictures_path(self) -> Path:
        """:return: pictures path tied to the user"""
    @property
    def user_videos_path(self) -> Path:
        """:return: videos path tied to the user"""
    @property
    def user_music_path(self) -> Path:
        """:return: music path tied to the user"""
    @property
    def user_desktop_path(self) -> Path:
        """:return: desktop path tied to the user"""
    @property
    def user_runtime_path(self) -> Path:
        """:return: runtime path tied to the user"""
    @property
    def site_runtime_path(self) -> Path:
        """:return: runtime path shared by users"""
    def iter_config_dirs(self) -> Iterator[str]:
        """:yield: all user and site configuration directories."""
    def iter_data_dirs(self) -> Iterator[str]:
        """:yield: all user and site data directories."""
    def iter_cache_dirs(self) -> Iterator[str]:
        """:yield: all user and site cache directories."""
    def iter_runtime_dirs(self) -> Iterator[str]:
        """:yield: all user and site runtime directories."""
    def iter_config_paths(self) -> Iterator[Path]:
        """:yield: all user and site configuration paths."""
    def iter_data_paths(self) -> Iterator[Path]:
        """:yield: all user and site data paths."""
    def iter_cache_paths(self) -> Iterator[Path]:
        """:yield: all user and site cache paths."""
    def iter_runtime_paths(self) -> Iterator[Path]:
        """:yield: all user and site runtime paths."""

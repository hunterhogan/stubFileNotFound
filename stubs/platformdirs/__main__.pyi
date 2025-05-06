"""Main entry point stub file."""

from __future__ import annotations

from typing import Literal

from platformdirs import PlatformDirs
from platformdirs.version import __version__

PROPS: tuple[str, ...] = (
    "user_data_dir",
    "user_config_dir",
    "user_cache_dir",
    "user_state_dir",
    "user_log_dir",
    "user_documents_dir",
    "user_downloads_dir",
    "user_pictures_dir",
    "user_videos_dir",
    "user_music_dir",
    "user_runtime_dir",
    "site_data_dir",
    "site_config_dir",
    "site_cache_dir",
    "site_runtime_dir",
)

def main() -> None:
    """Run the main entry point, printing directory paths for various platformdirs locations."""

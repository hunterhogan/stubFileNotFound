from pandas._typing import JSONSerializable as JSONSerializable
from pandas.compat._optional import VERSIONS as VERSIONS, get_version as get_version, import_optional_dependency as import_optional_dependency

def _get_commit_hash() -> str | None:
    """
    Use vendored versioneer code to get git hash, which handles
    git worktree correctly.
    """
def _get_sys_info() -> dict[str, JSONSerializable]:
    """
    Returns system information as a JSON serializable dictionary.
    """
def _get_dependency_info() -> dict[str, JSONSerializable]:
    """
    Returns dependency information as a JSON serializable dictionary.
    """
def show_versions(as_json: str | bool = False) -> None:
    """
    Provide useful information, important for bug reports.

    It comprises info about hosting operation system, pandas version,
    and versions of other installed relative packages.

    Parameters
    ----------
    as_json : str or bool, default False
        * If False, outputs info in a human readable form to the console.
        * If str, it will be considered as a path to a file.
          Info will be written to that file in JSON format.
        * If True, outputs info in JSON format to the console.

    Examples
    --------
    >>> pd.show_versions()  # doctest: +SKIP
    Your output may look something like this:
    INSTALLED VERSIONS
    ------------------
    commit           : 37ea63d540fd27274cad6585082c91b1283f963d
    python           : 3.10.6.final.0
    python-bits      : 64
    OS               : Linux
    OS-release       : 5.10.102.1-microsoft-standard-WSL2
    Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022
    machine          : x86_64
    processor        : x86_64
    byteorder        : little
    LC_ALL           : None
    LANG             : en_GB.UTF-8
    LOCALE           : en_GB.UTF-8
    pandas           : 2.0.1
    numpy            : 1.24.3
    ...
    """

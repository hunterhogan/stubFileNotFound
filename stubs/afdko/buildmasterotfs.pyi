from _typeshed import Incomplete
from afdko.fdkutils import get_temp_file_path as get_temp_file_path, run_shell_command as run_shell_command, validate_path as validate_path

__version__: str
logger: Incomplete

class ShellCommandError(Exception): ...

def generalizeCFF(otfPath) -> None: ...

MKOT_OPT: str

def build_masters(opts) -> None:
    """
    Build master OTFs using supplied options.
    """
def get_options(args): ...
def validateDesignspaceDoc(dsDoc, **kwArgs) -> None:
    """
    Validate the dsDoc DesignSpaceDocument object. Raises Exceptions if
    certain criteria are not met. These are above and beyond the basic
    validations in fontTools.designspaceLib and are specific to
    buildmasterotfs.
    """
def main(args=None): ...

__version__: str
LEN_CID_TOK: int

def validate_path(path_str): ...
def get_temp_file_path(directory=None): ...
def get_temp_dir_path(path_comp=None): ...
def get_resources_dir(): ...
def get_font_format(font_file_path): ...
def run_shell_command(args, suppress_output: bool = False):
    """
    Runs a shell command.
    Returns True if the command was successful, and False otherwise.
    """
def run_shell_command_logging(args):
    """
    Runs a shell command while logging both standard output and standard error.

    An earlier version of this function that used Popen.stdout.readline()
    was failing intermittently in CI, so now we're trying this version that
    uses Popen.communicate().
    """
def get_shell_command_output(args, std_error: bool = False):
    """
    Runs a shell command and captures its output.
    To also capture standard error call with std_error=True.

    Returns a tuple; the first element will be True if the command was
    successful, and False otherwise. The second element will be a
    Unicode-encoded string, or None.
    """
def runShellCmd(cmd, shell: bool = True, timeout=None): ...
def runShellCmdLogging(cmd, shell: bool = True): ...

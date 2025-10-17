# stubFileNotFound: Stub Type Files that reduce complaints from Pylance and Ruff

- I usually run [stubdefaulter](https://github.com/JelleZijlstra/stubdefaulter).
- I sometimes add missing annotations.
- I sometimes use `Any` for missing annotations so type checkers don't complain about "UNKNOWN".
- Most changes are automated, but not sophisticated.

This is a useful tool for me, but I haven't really figured out how to make it useful for more people.

## How to Contribute

1. Create or improve a stub file.
2. Submit a Pull Request.

## Usage

The following might work for you, but I don't really know for sure.

To use the stub files from this repository:

1. Clone the repository.

   ```sh
   git clone https://github.com/hunterhogan/stubFileNotFound.git
   ```

2. Tell your type checker about the `stubs` directory.

### Visual Studio Code

Relevant settings may include

- `python.analysis.stubPath`
- `mypy-type-checker.args` "--custom-typeshed-dir=typings"

### Virtual directories ([symlinks](https://ss64.com/nt/mklink.html))

```cmd
(.venv) C:\apps\Z0Z_tools> MKLINK /D typings \apps\stubFileNotFound\stubs
```

## Tools I might use to create a stub

1. Copy and paste the signature into a stub file. Improve it.
2. [mypy](https://www.mypy-lang.org/) [`stubgen`](https://mypy.readthedocs.io/en/stable/stubgen.html)

   ```sh
   stubgen --verbose --include-docstrings ^
   --include-private --output typings ^
   --package nameOfAnInstalledPackage
   ```

   VS Code: `workbench.action.openGlobalKeybindingsFile`

   ```json
    {
        "args": {
            "text": "stubgen --include-private --include-docstrings --output typings --verbose -p ${selectedText}\n"
        },
        "command": "workbench.action.terminal.sendSequence",
        "key": "ctrl+shift+t"
    },
   ```

   2026-10-16: "Hey, thanks, old me! I wanted the keybinding information, and here it is!"

3. [pyright](https://github.com/microsoft/pyright)
   1. `pyright --createstub nameOfAnInstalledPackage`
   2. ["Quick Fix"](https://microsoft.github.io/pyright/#/type-stubs?id=generating-type-stubs-in-vs-code) in VS Code.
4. [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.python) if `reportMissingTypeStubs` is enabled and the stub is missing: ["Quick Fix"](https://microsoft.github.io/pyright/#/type-stubs?id=generating-type-stubs-in-vs-code)

## Tools that can allegedly create a stub

1. [MonkeyType](https://github.com/Instagram/MonkeyType)
2. [stubgen-pyx](https://github.com/jon-edward/stubgen-pyx): `stubgen-pyx /path/to/package`
3. [stub-generator](https://pypi.org/project/stub-generator/)

### Tools I probably won't use

1. [Nuitka-Stubgen](https://github.com/Nuitka/Nuitka-Stubgen): [`pip install AST-Stubgen`](https://pypi.org/project/AST-Stubgen/)

   ```python
   import pathlib
   import Ast_Stubgen
   pathlib.Path('typings/Ast_Stubgen').mkdir(parents=True, exist_ok=True)
   Ast_Stubgen.generate_stub('.venv/Lib/site-packages/Ast_Stubgen/stubgen.py', 'typings/Ast_Stubgen/stubgen.pyi')
   Ast_Stubgen.generate_stub('.venv/Lib/site-packages/Ast_Stubgen/__init__.py', 'typings/Ast_Stubgen/__init__.pyi')
   ```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/stubFileNotFound/refs/heads/main/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)

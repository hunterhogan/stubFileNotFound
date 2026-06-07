"""AI generated without edits, ugly, and IDC."""
from __future__ import annotations

from pathlib import Path
import re as regex

def remove_docstrings_from_file(pathFilename: Path) -> None:  # noqa: D103
    content = pathFilename.read_text(encoding='utf-8')

    # Remove triple-quoted docstrings (both """ and ''')
    # This regex handles multiline docstrings
    content = regex.sub(r'""".*?"""', '...', content, flags=regex.DOTALL)
    content = regex.sub(r"'''.*?'''", '...', content, flags=regex.DOTALL)

    # Clean up any resulting blank lines (optional)
    content = regex.sub(r'\n\s*\n\s*\n', '\n\n', content)

    pathFilename.write_text(content, encoding='utf-8')

def process_directory(pathRoot: Path) -> None:  # noqa: D103
    for root, _dirs, files in pathRoot.walk():
        for file in files:
            if file.endswith('.pyi'):
                pathFilename = root / file
                print(f"Processing {pathFilename}")  # noqa: T201
                remove_docstrings_from_file(pathFilename)

# Run it
# process_directory(Path("/apps/stubFileNotFound/stubs/soundfile"))
remove_docstrings_from_file(Path("/apps/stubFileNotFound/stubs/soundfile.pyi"))

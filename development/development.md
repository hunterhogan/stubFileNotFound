# stubFileNotFound Development Process

> **This document is the single source of truth for all development and automation in this repository. AI assistants and human contributors MUST update this file to reflect the current state of the project after any significant change.**

## Project Overview

`stubFileNotFound` is a Python package that distributes high-quality, community-maintained type stubs (`.pyi` files) for third-party packages. The goal is to improve static analysis and developer experience for Python users. This repository does not contain runtime modules—only stubs and essential tooling.

## Core Principles

- **This file (`development/development.md`) must always be kept up to date.**
- **No reinvention of the wheel:** Use existing, well-maintained tools (e.g., `pyupgrade` for modernizing stubs, `mypy`/`pyright`/`stubtest` for validation). Do not write custom scripts for problems already solved by the community, unless they significantly automate a tedious process (like `stubFileNotFound/generate_stubs.py`).
- **Minimum supported Python version is 3.10.**
- **All configuration must be in `pyproject.toml` whenever possible.**
- **All automation and documentation must be reproducible and understandable by both AI assistants and human contributors.**
- **Directory structure and workflow must be documented here.**

## Directory Structure

- `stubs/`: All third-party `.pyi` stub files, organized by package name.
- `stubFileNotFound/`: The main package directory. Contains importable modules like `generate_stubs.py`.
- `tests/`: All tests for stub validation and package logic. If missing, create it.
- `development/`: Contains supplementary documentation like `third_party_packages.md` and this file (`development.md`).
- `pyproject.toml`: The single source of truth for configuration and metadata.
- `README.md`: User and contributor documentation.
- Any other directory must be explicitly documented here or removed.

## Development Workflow

1. **Stub Generation**
   - Use the automation script `stubFileNotFound/generate_stubs.py` for generating initial stubs.
     - For standard Python packages: `python -m stubFileNotFound.generate_stubs <package_name>`
     - For Cython packages: `python -m stubFileNotFound.generate_stubs <package_name> --cython`
   - This script uses `mypy.stubgen` or `stubgen-pyx` and places generated stubs in `stubs/<package_name>/`.
   - Manual generation using `pyright --createstub` or VS Code Quick Fixes is also possible for specific cases, followed by manual placement in `stubs/`.

2. **Stub Modernization**
   - The `stubFileNotFound/generate_stubs.py` script automatically runs `pyupgrade --py310-plus` on the newly generated stubs.
   - To modernize all existing stubs, run:

     ```sh
     pyupgrade --py310-plus stubs/**/*.pyi
     ```

3. **Stub Validation**
   - Use `mypy`, `pyright`, and `stubtest` to validate stubs. Add tests in `tests/` to automate this process.
   - Example validation commands:

     ```sh
     mypy --python-version 3.10 stubs/package_name
     pyright --pythonversion 3.10 --typeshedpath stubs/ examples/usage_of_package.py # (If example usage exists)
     stubtest package_name --stub-dir stubs/package_name --ignore-missing-stub
     ```

   - Do not write custom validation logic unless it is not already provided by these tools.

4. **Testing**
   - All new or updated stubs must be validated with the above tools.
   - Add or update tests in `tests/` to ensure stubs remain valid.
   - If the package includes Python modules, write tests for them in `tests/` as well.

5. **Documentation and Process**
   - After any significant change, update this `development.md` to reflect the new state or process.
   - Update `development/third_party_packages.md` to track dependencies and stubbed packages.
   - Keep `README.md` up to date for users and contributors.

## Automation

- Use `stubFileNotFound/generate_stubs.py` for generating and initially modernizing stubs.
- Use `pyupgrade` to modernize all stubs:

  ```sh
  pyupgrade --py310-plus stubs/**/*.pyi
  ```

- Use `pytest`, `mypy`, `pyright`, and `stubtest` for validation and testing. Configure them in `pyproject.toml`.
- Do not duplicate functionality already provided by these tools.

## Third-Party Package Tracking

- Refer to `development/third_party_packages.md` for a list of dependencies, packages with generated stubs, considered packages, and rejected packages. Keep this file updated.

## Open Issues / TODOs

- Reduce code duplication and remove any custom logic that is already solved by community tools (review existing stubs/code).
- Add or improve tests in `tests/` for stub validation.
- Ensure `development/third_party_packages.md` is comprehensive.
- Always update this file (`development/development.md`) after any process or structure change.

---

**If you are an AI assistant, you must update this file to match the current state of the project after any significant change.**

# stubFileNotFound Development Process

> **This document is the single source of truth for all development and automation in this repository. AI assistants and human contributors MUST update this file to reflect the current state of the project after any significant change.**

## Project Overview

`stubFileNotFound` is a Python package that distributes high-quality, community-maintained type stubs (`.pyi` files) for third-party packages. The goal is to improve static analysis and developer experience for Python users. This repository does not contain runtime modules—only stubs and essential tooling.

## Core Principles

- **This file (`development.md`) must always be kept up to date.**
- **No reinvention of the wheel:** Use existing, well-maintained tools (e.g., `pyupgrade` for modernizing stubs, `mypy`/`pyright`/`stubtest` for validation). Do not write custom scripts for problems already solved by the community.
- **Minimum supported Python version is 3.10.**
- **All configuration must be in `pyproject.toml` whenever possible.**
- **All automation and documentation must be reproducible and understandable by both AI assistants and human contributors.**
- **Directory structure and workflow must be documented here.**

## Directory Structure

- `stubs/`: All third-party `.pyi` stub files, organized by package name.
- `stubFileNotFound/`: The main package directory. Contains only importable, documented, and essential code. No random scripts.
- `tests/`: All tests for stub validation and package logic. If missing, create it.
- `pyproject.toml`: The single source of truth for configuration and metadata.
- `README.md`: User and contributor documentation.
- `development.md`: This file. Must always reflect the current state and process.
- Any other directory must be explicitly documented here or removed.

## Development Workflow

1. **Stub Generation**
   - Use `stubgen`, `pyright`, or similar tools to generate initial stubs for third-party packages.
   - Place generated stubs in `stubs/`, organized by package name.
   - Do not write custom stub generation scripts unless absolutely necessary and not already solved by existing tools.

2. **Stub Modernization**
   - Use [`pyupgrade`](https://github.com/asottile/pyupgrade) to automatically modernize stubs to Python 3.10+ syntax (e.g., remove deprecated `typing.List`, etc.).
   - Do not write custom code to check or fix deprecated types—run `pyupgrade` on all stubs instead.

3. **Stub Validation**
   - Use `mypy`, `pyright`, and `stubtest` to validate stubs. Add tests in `tests/` to automate this process.   - Example validation commands:

     ```sh
     mypy --python-version 3.10 stubs/package_name
     pyright --pythonversion 3.10 --typeshedpath stubs/ examples/usage_of_package.py
     stubtest package_name --stub-dir stubs/package_name --ignore-missing-stub
     ```

   - Do not write custom validation logic unless it is not already provided by these tools.

4. **Testing**
   - All new or updated stubs must be validated with the above tools.
   - Add or update tests in `tests/` to ensure stubs remain valid.
   - If the package includes Python modules, write tests for them in `tests/` as well.

5. **Documentation and Process**
   - After any significant change, update this `development.md` to reflect the new state or process.
   - Remove or document any directory (like `scripts/`) that is not part of the standard Python package structure.
   - Keep `README.md` up to date for users and contributors.

## Automation

- Use `pyupgrade` to modernize all stubs:

  ```sh
  pyupgrade --py310-plus stubs/**/*.pyi
  ```

- Use `pytest`, `mypy`, `pyright`, and `stubtest` for validation and testing. Configure them in `pyproject.toml`.
- Do not duplicate functionality already provided by these tools.

## Open Issues / TODOs

- Reduce code duplication and remove any custom logic that is already solved by community tools.
- Add or improve tests in `tests/`.
- Always update this file after any process or structure change.

---

**If you are an AI assistant, you must update this file to match the current state of the project after any significant change.**

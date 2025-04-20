# Third-Party Package Directory

This document tracks the third-party Python packages relevant to the `stubFileNotFound` project.

## Currently Used Packages (Dependencies)

These packages are listed in `pyproject.toml` and are essential for the project's functionality or development workflow.

* **mypy**: Used for static type checking and the `stubgen` tool.
* **pyright**: Used for static type checking and its stub generation capabilities.
* **stubgen-pyx**: Used specifically for generating stubs for Cython-based packages.
* **pyupgrade**: Used to automatically modernize Python code and stub files to newer syntax.
* **pytest**: Used for running tests.
* **pytest-cov**: Used for measuring test coverage.
* **pytest-xdist**: Used for parallel test execution.
* **setuptools**: Used as the build backend.
* **MonkeyType**: Listed as a potential tool in README, dependency kept for now.
* **AST-Stubgen**: Listed as a tool not likely to be used in README, dependency kept for now.

## Considered Packages (Potential Future Use)

These packages might be useful in the future, either as dependencies or for stub generation.

* *[Package Name]*: [Reason for consideration]

## Rejected Packages

These packages were considered but ultimately decided against.

* **Nuitka-Stubgen (via AST-Stubgen)**: Mentioned in `README.md` as unlikely to be used. Seems less maintained or standard compared to `mypy`/`pyright`.
* *[Package Name]*: [Reason for rejection]

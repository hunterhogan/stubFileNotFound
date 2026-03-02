IF NOT DEFINED VIRTUAL_ENV (
	ECHO "VIRTUAL_ENV not set. Please activate your virtual environment."
	EXIT /B 1
)
SET pathRoot=%VIRTUAL_ENV%\..

SET pathFix=%pathRoot%\stubs\

ruff check --fix --config ruff.toml %pathFix%
isort %pathFix%

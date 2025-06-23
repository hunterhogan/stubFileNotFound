robocopy typeshed\stdlib stubs\stdlib /s
robocopy pandas-stubs\pandas-stubs stubs\pandas /s

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

cd stubs
stubdefaulter --stdlib-path stdlib --packages .

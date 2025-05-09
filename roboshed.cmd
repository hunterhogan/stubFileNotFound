robocopy typeshed\stdlib stubs\stdlib /s

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

stubdefaulter --stdlib-path stubs\stdlib

robocopy typeshed\stubs stubs /s
robocopy typeshed\stdlib stubs\stdlib /s

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

stubdefaulter --stdlib-path stubs\stdlib

pushd stubs
py -m u
popd

call deactivate

call .312\Scripts\activate.bat

pushd stubs
py -m u312
popd

call deactivate

CALL .venv\Scripts\activate.bat

@REM forfiles /p stubs /m *.pyi /s /c "CMD /c pyupgrade --py310-plus @path"

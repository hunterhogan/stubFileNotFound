@ECHO OFF
robocopy typeshed\stdlib stubs\stdlib /S /MT
robocopy pandas-stubs\pandas-stubs stubs\pandas /S /MT
robocopy typeshed\stubs\tqdm stubs /S /MT /XD "@tests" /XF "METADATA.toml"

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

pushd stubs
stubdefaulter --stdlib-path stdlib --packages .
popd
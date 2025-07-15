@ECHO OFF
robocopy C:\clones\typeshed\stdlib stubs\stdlib /S /MT
robocopy C:\clones\pandas-stubs\pandas-stubs stubs\pandas /S /MT
robocopy C:\clones\typeshed\stubs\tqdm stubs /S /MT /XD "@tests" /XF "METADATA.toml"

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

pushd stubs
stubdefaulter --stdlib-path stdlib --packages .
popd

stubFileNotFound\missing2Any.py
@ECHO OFF

git -C C:\clones\pandas-stubs pull
git -C C:\clones\typeshed pull

PUSHd C:\apps\stubFileNotFound

robocopy C:\clones\pandas-stubs\pandas-stubs stubs\pandas /S /MT
robocopy C:\clones\typeshed\stdlib stubs\stdlib /S /MT
robocopy C:\clones\typeshed\stubs\networkx stubs /S /MT /XD "@tests" /XF "METADATA.toml"
robocopy C:\clones\typeshed\stubs\requests stubs /S /MT /XD "@tests" /XF "METADATA.toml"
robocopy C:\clones\typeshed\stubs\requests-oauthlib stubs /S /MT /XD "@tests" /XF "METADATA.toml"
robocopy C:\clones\typeshed\stubs\tqdm stubs /S /MT /XD "@tests" /XF "METADATA.toml"
robocopy C:\clones\typeshed\stubs\cffi stubs /S /MT /XD "@tests" /XF "METADATA.toml"

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

POPd

PUSHd C:\apps\stubFileNotFound\stubs
start /MIN "stubdefaulter-stdlib" stubdefaulter --stdlib-path stdlib --fix --add-complex-defaults

FOR /F %%G IN ('dir /B /AD') DO (
	IF NOT "%%G"=="stdlib" (
		start /MIN "stubdefaulter-%%G" stubdefaulter --packages %%G --fix --add-complex-defaults
		PING -n 3 127.0.0.1>nul
		@REM Wait 2 seconds
	)
)

FOR /F %%G IN ('dir /B *.pyi') DO (
	IF NOT "%%G"=="stdlib" (
		start /MIN "stubdefaulter-%%G" stubdefaulter --packages %%G --fix --add-complex-defaults
		PING -n 3 127.0.0.1>nul
		@REM Wait 2 seconds
	)
)

POPd

PUSHd C:\apps\stubFileNotFound

stubFileNotFound\missing2Any.py

POPd

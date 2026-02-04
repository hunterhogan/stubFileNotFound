@ECHO OFF

git -C C:\clones\pandas-stubs pull
git -C C:\clones\typeshed pull

PUSHd C:\apps\stubFileNotFound

ROBOCOPY C:\clones\pandas-stubs\pandas-stubs stubs\pandas /S /MT
ROBOCOPY C:\clones\typeshed\stdlib stubs\stdlib /S /MT

IF NOT DEFINED VIRTUAL_ENV CALL .venv\Scripts\activate.bat

POPd

PUSHd C:\apps\stubFileNotFound\stubs
START /MIN "stubdefaulter-stdlib" stubdefaulter --stdlib-path stdlib --fix --add-complex-defaults

FOR /F %%G IN ('dir /B /AD') DO (
	IF NOT "%%G"=="stdlib" (
		START /MIN "stubdefaulter-%%G" stubdefaulter --packages %%G --fix --add-complex-defaults
		PING -n 3 127.0.0.1>nul
		@REM Wait 2 seconds
	)
)

FOR /F %%G IN ('dir /B *.pyi') DO (
	IF NOT "%%G"=="stdlib" (
		START /MIN "stubdefaulter-%%G" stubdefaulter --packages %%G --fix --add-complex-defaults
		PING -n 3 127.0.0.1>nul
		@REM Wait 2 seconds
	)
)

POPd

PUSHd C:\apps\stubFileNotFound

py stubFileNotFound\missing2Any.py

POPd

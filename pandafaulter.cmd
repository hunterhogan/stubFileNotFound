@ECHO OFF

git -C C:\clones\pandas-stubs pull

PUSHd C:\apps\stubFileNotFound

ROBOCOPY C:\clones\pandas-stubs\pandas-stubs stubs\pandas /S /MT /NJH /NDL /NFL

IF NOT DEFINED VIRTUAL_ENV CALL .venv\Scripts\activate.bat

POPd

PUSHd C:\apps\stubFileNotFound\stubs

FOR /F %%G IN ('dir /B /AD') DO (
	IF NOT "%%G"=="stdlib" (
		START /MIN "stubdefaulter-%%G" stubdefaulter --packages %%G --fix --add-complex-defaults
		PING -n 3 127.0.0.1>nul
		@REM Wait 2 seconds
	)
)

POPd

PUSHd C:\apps\stubFileNotFound

py -m stubFileNotFound.missing2AnyTransformers

POPd

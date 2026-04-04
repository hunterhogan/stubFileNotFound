PUSHd C:\apps\stubFileNotFound\stubs
FOR /F %%G IN ('dir /B /AD') DO (
	IF NOT "%%G"=="stdlib" (
		PUSHd %%G
		stubdefaulter --packages %%G --fix --add-complex-defaults
		POPd
	)
)
POPd

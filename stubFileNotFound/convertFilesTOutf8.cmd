@ECHO OFF

FOR /F %%G IN ('DIR /B /S C:\apps\stubFileNotFound\stubs\matplotlib\*.pyi') DO (
    normalizer -n -m -r -f "%%~G"
)

@ECHO OFF

FOR /F %%G IN ('DIR /B /S C:\apps\stubFileNotFound\stubs\*.pyi') DO (
    normalizer -n -m -r -f "%%~G"
)
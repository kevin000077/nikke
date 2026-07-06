@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python virtual environment was not found:
    echo %PYTHON%
    pause
    exit /b 1
)

pushd "%ROOT%"
"%PYTHON%" -m marble_aim
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" (
    echo.
    echo Marble Aim failed to start. Error code: %RESULT%
    pause
)
exit /b %RESULT%

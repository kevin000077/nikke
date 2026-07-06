@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Virtual environment not found: %PYTHON%
    pause
    exit /b 1
)

pushd "%ROOT%"
"%PYTHON%" -m pip install -e ".[package]"
if errorlevel 1 goto :failed

"%PYTHON%" -m PyInstaller --noconfirm --clean MarbleAim.spec
if errorlevel 1 goto :failed

echo.
echo Build completed:
echo %ROOT%dist\MarbleAim.exe
popd
pause
exit /b 0

:failed
echo.
echo Build failed.
popd
pause
exit /b 1

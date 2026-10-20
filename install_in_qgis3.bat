@echo off
rem install_in_qgis3.bat
rem Installs the legacy pythermonet into the Python environment of QGIS 3.
rem Usage: right-click -> "Run as administrator"   (normal install)
rem        install_in_qgis3.bat -e                (editable install, for developers)
rem Packages already shipped with QGIS are frozen as constraints, so pip can add
rem missing packages but never upgrade or downgrade the ones QGIS depends on.

setlocal
cd /d "%~dp0"

echo === pythermonet (legacy) installer for QGIS 3 ===
echo.

net session >nul 2>&1
if errorlevel 1 goto :no_admin

set "QGIS_ROOT="
for /d %%Q in ("%ProgramFiles%\QGIS 3*") do if exist "%%Q\bin\o4w_env.bat" set "QGIS_ROOT=%%Q"
if not defined QGIS_ROOT if exist "C:\OSGeo4W\bin\o4w_env.bat" set "QGIS_ROOT=C:\OSGeo4W"
if not defined QGIS_ROOT goto :no_qgis

rem Load the QGIS environment (as the OSGeo4W Shell does): without its bin\ on
rem PATH, Python's ssl module cannot load its DLLs and pip cannot reach PyPI.
call "%QGIS_ROOT%\bin\o4w_env.bat"

set "PY="
for /d %%P in ("%QGIS_ROOT%\apps\Python3*") do if exist "%%P\python.exe" set "PY=%%P\python.exe"
if not defined PY goto :no_qgis
echo QGIS Python found: %PY%
echo.

set "MODE="
if /i "%~1"=="-e" set "MODE=-e"

set "CONSTRAINTS=%TEMP%\pythermonet_qgis_constraints.txt"
"%PY%" -m pip list --format=freeze --exclude pythermonet > "%CONSTRAINTS%"
if errorlevel 1 goto :failed

rem setuptools never empties build\, so files left by an earlier build would end up in this install.
if exist build rmdir /s /q build

echo Installing pythermonet %MODE% ...
"%PY%" -m pip install %MODE% . -c "%CONSTRAINTS%"
if errorlevel 1 goto :failed

echo.
"%PY%" -c "import importlib.metadata as m, pythermonet; print('OK: pythermonet', m.version('pythermonet'), 'installed at', pythermonet.__path__[0])"
if errorlevel 1 goto :failed

echo.
echo === Installation completed. You can now start QGIS. ===
goto :end

:no_admin
echo ERROR: Administrator rights are required.
echo Close this window, right-click install_in_qgis3.bat and choose "Run as administrator".
goto :end

:no_qgis
echo ERROR: QGIS 3 was not found in "%ProgramFiles%" or C:\OSGeo4W.
echo Install QGIS 3 first, then run this script again.
goto :end

:failed
echo.
echo ERROR: Installation failed. No packages shipped with QGIS were changed.
echo Please send a screenshot of this window to the QThermonet developers.

:end
echo.
pause
endlocal

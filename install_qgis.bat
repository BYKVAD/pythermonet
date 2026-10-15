@echo off
rem install_qgis.bat
rem Installs pythermonet into the Python environment of QGIS 4.2.
rem Usage: right-click -> "Run as administrator"   (normal install)
rem        install_qgis.bat -e                    (editable install, for developers)
rem Packages already shipped with QGIS are frozen as constraints, so pip can add
rem missing packages but never upgrade or downgrade the ones QGIS depends on.

setlocal
cd /d "%~dp0"

echo === pythermonet installer for QGIS 4.2 ===
echo.

net session >nul 2>&1
if errorlevel 1 goto :no_admin

set "PY="
for /d %%Q in ("%ProgramFiles%\QGIS 4.2*") do for /d %%P in ("%%Q\apps\Python3*") do if exist "%%P\python.exe" set "PY=%%P\python.exe"
if not defined PY for /d %%P in ("C:\OSGeo4W\apps\Python3*") do if exist "%%P\python.exe" set "PY=%%P\python.exe"
if not defined PY goto :no_qgis
echo QGIS Python found: %PY%
echo.

set "MODE="
if /i "%~1"=="-e" set "MODE=-e"

set "CONSTRAINTS=%TEMP%\pythermonet_qgis_constraints.txt"
"%PY%" -m pip list --format=freeze --exclude pythermonet > "%CONSTRAINTS%"
if errorlevel 1 goto :failed

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
echo Close this window, right-click install_qgis.bat and choose "Run as administrator".
goto :end

:no_qgis
echo ERROR: QGIS 4.2 was not found in "%ProgramFiles%" or C:\OSGeo4W.
echo Install QGIS 4.2 first, then run this script again.
goto :end

:failed
echo.
echo ERROR: Installation failed. No packages shipped with QGIS were changed.
echo Please send a screenshot of this window to the QThermonet developers.

:end
echo.
pause
endlocal

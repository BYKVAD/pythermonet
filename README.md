# pythermonet
New stable release of pythermonet - Legacy version can either be found at https://github.com/soeb1978/pythermonet or branch "legacy"


## Installation

### Standalone
By running the command .\setup.ps1 on a windows machine will trigger the powershell script to create a virtual environment with the required dependencies to run the examples in the example folder. This environment is separate from QGIS.

### In QGIS 4.2 (e.g. for the QThermonet plugin)
1. Close QGIS.
2. Download and unzip this repository.
3. Right-click `install_qgis.bat` and choose **Run as administrator**.
4. Wait for "Installation completed", then start QGIS.

The script finds QGIS 4.2 itself and installs pythermonet into its Python environment. Packages shipped with QGIS are never upgraded or downgraded. To update pythermonet later, download the new version and run the script again. Developers can use `install_qgis.bat -e` for an editable install.

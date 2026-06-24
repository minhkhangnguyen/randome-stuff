@echo off
echo ================================================
echo   Installing Chinese + Japanese to Vietnamese
echo ================================================
echo.

"D:\AI generated\local_runtime\Python\Python312\python.exe" -m pip install argostranslate

echo.
echo Downloading translation models...
"D:\AI generated\local_runtime\Python\Python312\python.exe" -c "import argostranslate.package; argostranslate.package.update_package_index(); pkgs = [p for p in argostranslate.package.get_available_packages() if (p.from_code=='zh' and p.to_code=='vi') or (p.from_code=='ja' and p.to_code=='vi')]; [argostranslate.package.install_from_path(p.download()) for p in pkgs]"

echo.
echo ✅ Translation models installed successfully!
pause

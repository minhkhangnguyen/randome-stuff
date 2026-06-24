@echo off
cd /d %~dp0

echo Checking for updates...
git pull

echo Installing/Updating requirements...
"D:\AI generated\local_runtime\Python\Python312\python.exe" -m pip install -r requirements.txt

echo Starting screen area translator...
echo Drag to select an area. Right-click subtitle box to close.
"D:\AI generated\local_runtime\Python\Python312\python.exe" screen_translator.py
pause

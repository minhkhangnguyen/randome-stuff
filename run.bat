@echo off
cd /d D:\randome-stuff

echo Checking for updates...
git pull

echo Installing/Updating requirements...
"D:\AI generated\local_runtime\Python\Python312\python.exe" -m pip install -r requirements.txt >nul 2>&1

echo Starting translator...
"D:\AI generated\local_runtime\Python\Python312\python.exe" video_translator.py
pause

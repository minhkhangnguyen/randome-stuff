@echo off
cd /d D:\randome-stuff

echo Checking for updates...
git pull

echo Starting translator...
"D:\AI generated\local_runtime\Python\Python312\python.exe" video_translator.py
pause

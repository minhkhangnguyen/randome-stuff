@echo off
cd /d %~dp0

echo Checking for updates...
git pull

echo Installing/Updating requirements...
"D:\AI generated\local_runtime\Python\Python312\python.exe" -m pip install -r requirements.txt

echo.
echo Starting FULL GOOGLE mode...
echo Speech: Google Cloud Speech-to-Text
echo Translation: Google Cloud Translation
echo.
echo IMPORTANT: GOOGLE_APPLICATION_CREDENTIALS must point to your Google Cloud JSON key.
echo Example:
echo set GOOGLE_APPLICATION_CREDENTIALS=D:\google-key.json
echo.

set SPEECH_ENGINE=google
set TRANSLATOR=google_cloud
"D:\AI generated\local_runtime\Python\Python312\python.exe" video_translator.py
pause

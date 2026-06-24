@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Installing Chinese + Japanese to Vietnamese
echo ================================================
echo.

set "PYTHON_EXE=D:\AI generated\local_runtime\Python\Python312\python.exe"

"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo Downloading translation models and Whisper model...
"%PYTHON_EXE%" -c "from video_translator import ensure_translation_package, load_whisper_model; ensure_translation_package('zh','vi'); ensure_translation_package('ja','vi'); load_whisper_model(); print('All models installed successfully.')"
if errorlevel 1 goto fail

echo.
echo ✅ Translation and Whisper models installed successfully!
pause
exit /b 0

:fail
echo.
echo ❌ Installation failed. Please copy the error above and send it for fixing.
pause
exit /b 1

@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Installing Tesseract OCR for screen translate
echo ================================================
echo.

echo Step 1/2: Installing Tesseract OCR app...
where tesseract >nul 2>&1
if errorlevel 1 (
    where winget >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ❌ winget is not available on this PC.
        echo Please install Tesseract manually from:
        echo https://github.com/UB-Mannheim/tesseract/wiki
        echo.
        pause
        exit /b 1
    )
    winget install --id UB-Mannheim.TesseractOCR -e --source winget
) else (
    echo Tesseract already found in PATH.
)

echo.
echo Step 2/2: Downloading OCR language files locally...
if not exist tessdata mkdir tessdata

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$langs=@('eng','chi_sim','chi_tra','jpn');" ^
  "foreach($l in $langs){" ^
  "  $out=Join-Path 'tessdata' ($l+'.traineddata');" ^
  "  if(!(Test-Path $out)){" ^
  "    Write-Host ('Downloading '+$l+'...');" ^
  "    Invoke-WebRequest -Uri ('https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/'+$l+'.traineddata') -OutFile $out;" ^
  "  } else { Write-Host ($l+' already exists.'); }" ^
  "}"
if errorlevel 1 goto fail

echo.
echo ✅ Tesseract setup done.
echo Now run run_screen.bat again.
pause
exit /b 0

:fail
echo.
echo ❌ Failed to download language files.
echo Check your internet connection, then run this file again.
pause
exit /b 1

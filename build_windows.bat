@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run: python -m venv .venv
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
  echo PyInstaller is not installed. Run:
  echo .venv\Scripts\python.exe -m pip install -r requirements-build.txt
  pause
  exit /b 1
)

echo Building TrailTimePredictor portable edition...
".venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm trail_predictor.spec
if errorlevel 1 (
  echo Build failed. Check the output above.
  pause
  exit /b 1
)

copy /Y "PORTABLE_README.txt" "dist\TrailTimePredictor\使用说明.txt" >nul
echo.
echo Build completed: dist\TrailTimePredictor\TrailTimePredictor.exe
pause

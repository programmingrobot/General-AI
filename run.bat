@echo off
setlocal
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 exit /b %errorlevel%
python download_model.py
if errorlevel 1 exit /b %errorlevel%
python app.py

@echo off
REM Double-clickable shortcut to launch the MIST Tauri frontend dev server.
REM Calls scripts\start_frontend.py which handles npm install / run dev.

cd /d "%~dp0.."
python scripts\start_frontend.py %*
pause

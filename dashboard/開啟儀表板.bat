@echo off
rem Driver Station at the bottom, dashboard above it. See start_dashboard.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_dashboard.ps1"
if errorlevel 1 pause

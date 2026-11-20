@echo off
rem One-click start: FRC Driver Station + robot dashboard.
rem Same as dashboard\start_dashboard.ps1 (DS at the bottom, dashboard above it); kept here for the old shortcut.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dashboard\start_dashboard.ps1"
if errorlevel 1 pause

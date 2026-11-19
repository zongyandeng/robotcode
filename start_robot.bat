@echo off
rem One-click start: FRC Driver Station + robot dashboard
rem Edit the paths below if your folders are different.

set "DS=C:\Program Files (x86)\FRC Driver Station\DriverStation.exe"
set "DASH=C:\Users\Edison\Desktop\claude\TEL_YOLO\dashboard\dashboard.html"
set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"
set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

rem 1) Driver Station first
if exist "%DS%" (
    start "" "%DS%"
) else (
    echo Driver Station not found: %DS%
)

rem 2) Wait a few seconds so the robot link can come up
timeout /t 3 /nobreak >nul

rem 3) Dashboard (Chrome, or Edge if Chrome is missing)
if exist "%CHROME%" (
    start "" "%CHROME%" "file:///%DASH:\=/%"
) else if exist "%EDGE%" (
    start "" "%EDGE%" "file:///%DASH:\=/%"
) else (
    start "" "%DASH%"
)

exit /b 0

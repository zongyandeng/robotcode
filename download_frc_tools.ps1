$downloadsFolder = "C:\Users\Edison\Desktop\antigravity\robot\downloads"
if (-not (Test-Path $downloadsFolder)) {
    New-Item -ItemType Directory -Path $downloadsFolder | Out-Null
    Write-Host "Created downloads folder: $downloadsFolder" -ForegroundColor Green
}

function Download-File-Curl {
    param (
        [string]$url,
        [string]$outputPath,
        [string]$name
    )
    Write-Host "Downloading $name using curl..." -ForegroundColor Cyan
    Write-Host "Source: $url" -ForegroundColor DarkGray
    
    # 呼叫 curl.exe 下載，加上 -C - 支援斷點續傳
    curl.exe -C - -L --progress-bar -o $outputPath $url
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Download complete: $name -> $outputPath" -ForegroundColor Green
    } else {
        Write-Host "Failed to download $name. Curl exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Please download manually: $url" -ForegroundColor Yellow
    }
}

# 1. WPILib 2026.2.1 Windows ISO (完整大小約為 2514407424 bytes)
$wpilibUrl = "https://packages.wpilib.workers.dev/installer/v2026.2.1/Win64/WPILib_Windows-2026.2.1.iso"
$wpilibPath = Join-Path $downloadsFolder "WPILib_Windows-2026.2.1.iso"

$needDownload = $true
if (Test-Path $wpilibPath) {
    $size = (Get-Item $wpilibPath).Length
    if ($size -ge 2514000000) {
        $needDownload = $false
        Write-Host "WPILib 2026 Installer already exists and seems complete ($([Math]::Round($size / 1MB)) MB). Skipping." -ForegroundColor Yellow
    } else {
        Write-Host "Found incomplete WPILib Installer ($([Math]::Round($size / 1MB)) MB). Resuming download..." -ForegroundColor Yellow
    }
}

if ($needDownload) {
    Download-File-Curl -url $wpilibUrl -outputPath $wpilibPath -name "WPILib 2026 Installer (2.4 GB)"
}

# 2. AdvantageScope v26.0.2 (完整大小約為 163196543 bytes)
$advScopeUrl = "https://github.com/Mechanical-Advantage/AdvantageScope/releases/download/v26.0.2/advantagescope-win-x64-v26.0.2.exe"
$advScopePath = Join-Path $downloadsFolder "advantagescope-win-x64-v26.0.2.exe"
$needAdv = $true
if (Test-Path $advScopePath) {
    $size = (Get-Item $advScopePath).Length
    if ($size -ge 163190000) {
        $needAdv = $false
        Write-Host "AdvantageScope already exists. Skipping." -ForegroundColor Yellow
    }
}
if ($needAdv) {
    Download-File-Curl -url $advScopeUrl -outputPath $advScopePath -name "AdvantageScope v26.0.2"
}

# 3. PathPlanner v2026.1.2 (完整大小約為 12432756 bytes)
$pathPlannerUrl = "https://github.com/mjansen4857/pathplanner/releases/download/v2026.1.2/PathPlanner-Windows-v2026.1.2-setup.exe"
$pathPlannerPath = Join-Path $downloadsFolder "PathPlanner-Windows-v2026.1.2-setup.exe"
$needPath = $true
if (Test-Path $pathPlannerPath) {
    $size = (Get-Item $pathPlannerPath).Length
    if ($size -ge 12430000) {
        $needPath = $false
        Write-Host "PathPlanner already exists. Skipping." -ForegroundColor Yellow
    }
}
if ($needPath) {
    Download-File-Curl -url $pathPlannerUrl -outputPath $pathPlannerPath -name "PathPlanner v2026.1.2"
}

Write-Host ""
Write-Host "--- Manual Download Guide ---" -ForegroundColor Yellow
Write-Host "Please download or install the following tools manually:" -ForegroundColor White
Write-Host "- CTRE Phoenix Tuner X (Open Microsoft Store...)" -ForegroundColor Cyan
Start-Process "https://apps.microsoft.com/store/detail/phoenix-tuner-x/9P6PRG013531"

Write-Host "- REV Hardware Client 2 (Open download page...)" -ForegroundColor Cyan
Start-Process "https://rhc2.revrobotics.com/download/download.html"

Write-Host "- NI FRC Game Tools 2026 (Open download page...)" -ForegroundColor Cyan
Start-Process "https://www.ni.com/en/support/downloads/drivers/download.frc-game-tools.html"

Write-Host ""
Write-Host "Script execution completed. Check files in: $downloadsFolder" -ForegroundColor Green

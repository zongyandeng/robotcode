[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$headers = @{"User-Agent" = "Mozilla/5.0"}

Write-Host "Fetching WPILib..."
$wpilib = Invoke-RestMethod -Uri "https://api.github.com/repos/wpilibsuite/allwpilib/releases/tags/v2026.2.1" -Headers $headers
$wpilib.assets | Select-Object name, browser_download_url | Format-Table | Out-String | Write-Host

Write-Host "Fetching AdvantageScope..."
$adv = Invoke-RestMethod -Uri "https://api.github.com/repos/Mechanical-Advantage/AdvantageScope/releases/latest" -Headers $headers
$adv.assets | Select-Object name, browser_download_url | Format-Table | Out-String | Write-Host

Write-Host "Fetching PathPlanner..."
$path = Invoke-RestMethod -Uri "https://api.github.com/repos/mjansen4857/pathplanner/releases/latest" -Headers $headers
$path.assets | Select-Object name, browser_download_url | Format-Table | Out-String | Write-Host

Write-Host "Fetching CTRE Phoenix..."
$ctre = Invoke-RestMethod -Uri "https://api.github.com/repos/CrossTheRoadElec/Phoenix-Releases/releases/latest" -Headers $headers
$ctre.assets | Select-Object name, browser_download_url | Format-Table | Out-String | Write-Host

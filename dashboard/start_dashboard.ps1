# 開比賽畫面：FRC Driver Station 貼在螢幕最下方，儀表板（Edge App 模式，沒有網址列）填滿 DS 上方的空間。
# 用法：雙擊同資料夾的「開啟儀表板.bat」（或在 PowerShell 執行 .\start_dashboard.ps1）。
#   - DS 沒開會自動開；DS 最小化會先還原
#   - 儀表板用獨立的 Edge 設定檔（記得 roboRIO 位址、相機網址），不影響平常用的 Edge
#   - Driver Station 以系統管理員權限執行，一般程式移不動它的視窗：移不動時會跳一次 UAC，
#     只有「移動 DS」這一步用系統管理員權限（-MoveDs），儀表板瀏覽器仍是一般權限
param([string]$MoveDs)   # "x,y,寬,高"
$ErrorActionPreference = "Stop"

$DsExe   = "C:\Program Files (x86)\FRC Driver Station\DriverStation.exe"
$Html    = Join-Path $PSScriptRoot "dashboard.html"
$ProfileDir = Join-Path $env:LOCALAPPDATA "FRC9427Dashboard"
$Browser = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Google\Chrome\Application\chrome.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Browser) { throw "找不到 Edge 或 Chrome" }

Add-Type @'
using System; using System.Text; using System.Runtime.InteropServices;
public class DashWin {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr p);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, int msg, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool repaint);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  /** 第一個「可見、標題以 prefix 開頭」的頂層視窗 */
  public static IntPtr Find(string prefix) {
    IntPtr found = IntPtr.Zero;
    EnumWindows((h, p) => {
      if (!IsWindowVisible(h)) return true;
      var sb = new StringBuilder(256); GetWindowText(h, sb, 256);
      if (sb.ToString().StartsWith(prefix)) { found = h; return false; }
      return true; }, IntPtr.Zero);
    return found; }
  /** 第一個「可見、標題完全等於 title」的頂層視窗 */
  public static IntPtr FindExact(string title) {
    IntPtr found = IntPtr.Zero;
    EnumWindows((h, p) => {
      if (!IsWindowVisible(h)) return true;
      var sb = new StringBuilder(256); GetWindowText(h, sb, 256);
      if (sb.ToString() == title) { found = h; return false; }
      return true; }, IntPtr.Zero);
    return found; }
}
'@

function Wait-Window([string]$prefix, [int]$seconds) {
    $t = [DateTime]::Now.AddSeconds($seconds)
    while ([DateTime]::Now -lt $t) {
        $h = [DashWin]::Find($prefix)
        if ($h -ne [IntPtr]::Zero) { return $h }
        Start-Sleep -Milliseconds 300
    }
    return [IntPtr]::Zero
}

# 系統管理員模式：只移動 DS 視窗就結束（由下面的一般模式在移不動時呼叫）
if ($MoveDs) {
    $h = Wait-Window "FRC Driver Station" 5
    $v = $MoveDs.Split(",") | ForEach-Object { [int]$_ }
    if ($h -ne [IntPtr]::Zero) { [void][DashWin]::MoveWindow($h, $v[0], $v[1], $v[2], $v[3], $true) }
    exit
}

# 1. Driver Station：沒開就開，最小化就還原，移到螢幕最下方（左下角，保持 DS 原本大小）
if (-not (Get-Process DriverStation -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path $DsExe)) { throw "找不到 Driver Station：$DsExe" }
    Start-Process $DsExe
}
$ds = Wait-Window "FRC Driver Station" 30
Add-Type -AssemblyName System.Windows.Forms
$work = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$dsTop = $work.Bottom - 230   # 量不到 DS 時的預設高度
if ($ds -ne [IntPtr]::Zero) {
    if ([DashWin]::IsIconic($ds)) {
        # DS 是 LabVIEW 程式，ShowWindow 還原無效；送「工作列還原」(WM_SYSCOMMAND / SC_RESTORE) 才有用
        [void][DashWin]::SendMessage($ds, 0x0112, [IntPtr]0xF120, [IntPtr]::Zero)
        Start-Sleep -Milliseconds 800
    }
    $r = New-Object DashWin+RECT
    [void][DashWin]::GetWindowRect($ds, [ref]$r)
    $dsW = $r.R - $r.L; $dsH = $r.B - $r.T
    if ($dsH -gt 50 -and $dsH -lt $work.Height / 2) {
        $dsTop = $work.Bottom - $dsH
        if ($r.T -ne $dsTop -or $r.L -ne $work.Left) {
            if (-not [DashWin]::MoveWindow($ds, $work.Left, $dsTop, $dsW, $dsH, $true)) {
                # 存取被拒（DS 是系統管理員權限）：用系統管理員權限只做「移動 DS」，會跳一次 UAC
                $moveArg = "{0},{1},{2},{3}" -f $work.Left, $dsTop, $dsW, $dsH
                try {
                    Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList @(
                        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-MoveDs", $moveArg)
                } catch {
                    Write-Warning "沒有允許系統管理員權限，Driver Station 留在原處（可以自己拖到螢幕最下方）"
                }
            }
        }
    } else {
        Write-Warning "量不到 Driver Station 的大小，儀表板高度用預設值"
    }
} else {
    Write-Warning "30 秒內沒看到 Driver Station 視窗，儀表板高度用預設值"
}

# 2. 儀表板：Edge App 模式，放在 DS 上方
$w = $work.Width
$h = $dsTop - $work.Top
$url = "file:///" + ($Html -replace '\\', '/')
$DashTitle = "9427 機器人儀表板"   # dashboard.html 的 <title>；App 模式視窗標題就是它（一般瀏覽器分頁會多「 - Google Chrome」等字尾，不會被抓到）
$dash = [DashWin]::FindExact($DashTitle)
if ($dash -eq [IntPtr]::Zero) {
    $browserArgs = @("--app=$url", "--user-data-dir=$ProfileDir", "--no-first-run", "--no-default-browser-check",
              "--window-position=$($work.Left),$($work.Top)", "--window-size=$w,$h")
    Start-Process $Browser -ArgumentList $browserArgs
    # 瀏覽器的視窗大小參數有時不準（例如設定檔記住舊大小）：找到視窗後再精確移一次
    $t = [DateTime]::Now.AddSeconds(20)
    while ($dash -eq [IntPtr]::Zero -and [DateTime]::Now -lt $t) {
        Start-Sleep -Milliseconds 300
        $dash = [DashWin]::FindExact($DashTitle)
    }
}   # 已經開著：不再開第二個，只移位置
if ($dash -ne [IntPtr]::Zero) {
    if ([DashWin]::IsIconic($dash)) { [void][DashWin]::SendMessage($dash, 0x0112, [IntPtr]0xF120, [IntPtr]::Zero) }   # 最小化就先還原
    Start-Sleep -Milliseconds 500
    [void][DashWin]::MoveWindow($dash, $work.Left, $work.Top, $w, $h, $true)
}
"儀表板 {0}x{1}，Driver Station 從 y={2} 開始" -f $w, $h, $dsTop

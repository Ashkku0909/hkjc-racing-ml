# =====================================================================
# HKJC Quant Terminal - online start (phone trackable)
# Starts Streamlit on 0.0.0.0 + a Cloudflare quick tunnel (public HTTPS
# URL, no router port-forwarding needed). Run via start_online.bat or:
#   powershell -ExecutionPolicy Bypass -File start_online.ps1
# =====================================================================
$ErrorActionPreference = 'Stop'
$port = 8501
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# --- 1. Streamlit (local/LAN) ----------------------------------------
$listen = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $listen) {
    Start-Process -FilePath "python" -ArgumentList @(
        "-m", "streamlit", "run", "app.py",
        "--server.headless", "true",
        "--server.port", "$port",
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root\streamlit.log" -RedirectStandardError "$root\streamlit.err.log"
    Write-Host "[1/2] Streamlit starting -> http://localhost:$port  | LAN: http://<PC-IP>:$port"
} else {
    Write-Host "[1/2] Streamlit already listening on :$port"
}

# --- 2. Cloudflare quick tunnel (public, HTTPS) ----------------------
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) {
    $cf = "$env:LOCALAPPDATA\cloudflared\cloudflared.exe"
}
if (Test-Path $cf) {
    $tlog = "$root\tunnel.log"
    Remove-Item $tlog, "$root\tunnel.err.log" -ErrorAction SilentlyContinue
    Start-Process -FilePath $cf -ArgumentList @("tunnel", "--url", "http://localhost:$port", "--no-autoupdate") `
        -WindowStyle Hidden -RedirectStandardOutput $tlog -RedirectStandardError "$root\tunnel.err.log"
    Write-Host "[2/2] Cloudflare tunnel starting..."
    $found = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 1
        foreach ($f in @($tlog, "$root\tunnel.err.log")) {
            if (Test-Path $f) {
                $m = Select-String -Path $f -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
                if ($m) {
                    Write-Host ""
                    Write-Host "PUBLIC URL (open on your phone): $($m.Matches[0].Value)"
                    Write-Host "LAN URL (same WiFi):               http://192.168.0.214:$port"
                    $found = $true
                    break
                }
            }
        }
        if ($found) { break }
    }
    if (-not $found) { Write-Host "No tunnel URL yet - check tunnel.log / tunnel.err.log" }
} else {
    Write-Host "[2/2] cloudflared not found - LAN only."
    Write-Host "      Install: https://github.com/cloudflare/cloudflared/releases (cloudflared-windows-amd64.exe)"
}

# daily_pipeline.ps1
# 매일 운영 DB sync → 신호 산출 → forward-test 트래킹
#
# 권장 실행 시각: KR 마감 + 운영 DB 업데이트 직후 (현재 17:00)
# 실행:
#   .\daily_pipeline.ps1
#   .\daily_pipeline.ps1 -Full       # asset_quality, products도 sync
#   .\daily_pipeline.ps1 -AsOf 2026-04-30

param(
    [switch]$Full = $false,
    [string]$AsOf = $null
)

$ErrorActionPreference = "Stop"

# 환경 변수
$env:DB_NAME = "equity"
$env:PGPASSWORD = "postgres"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
$LogFile = Join-Path $LogDir ("daily_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Run-Stage([string]$name, [scriptblock]$action) {
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "  [$name]" -ForegroundColor Cyan
    Write-Host ("=" * 72)
    $t0 = Get-Date
    & $action
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    Write-Host ("  → {0:F1}s" -f $elapsed) -ForegroundColor DarkGreen
}

# 1) Sync
Run-Stage "STAGE 1 — Sync (운영 DB → 로컬)" {
    $args = @("sync_equity_db.py")
    if ($Full) { $args += "--full" }
    python @args 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) { throw "sync_equity_db.py 실패 (exit $LASTEXITCODE)" }
}

# 2) Daily signals + forward-test 트래킹
Run-Stage "STAGE 2 — Daily Signals + Forward-Test 트래킹" {
    $args = @("daily_signals.py")
    if ($AsOf) { $args += "--as-of"; $args += $AsOf }
    python @args 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) { throw "daily_signals.py 실패 (exit $LASTEXITCODE)" }
}

Write-Host ""
Write-Host ("=" * 72)
Write-Host "  완료. 로그: $LogFile" -ForegroundColor Green
Write-Host ("=" * 72)

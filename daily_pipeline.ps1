# daily_pipeline.ps1
# 매일 운영: portfolio_db_dev → 로컬 equity 증분 sync → daily_signals(catch-up) → forward 추적
#
# 권장 실행: KR 마감 + dev 일일 가격 작업 완료 후 (스케줄러 기본 18:00)
# 수동 실행:
#   .\daily_pipeline.ps1
#   .\daily_pipeline.ps1 -AsOf 2026-04-30     # 특정일만
#   .\daily_pipeline.ps1 -Full                # products/asset_quality도 sync

param(
    [switch]$Full = $false,
    [string]$AsOf = $null,
    [switch]$Quiet = $false     # 완료 알림 팝업 끄기
)

$ErrorActionPreference = "Stop"   # PowerShell cmdlet 오류는 즉시 중단

# conda dashboard 환경 (psycopg2/pandas 보유)
$Py = "C:\Users\aleph\.conda\envs\dashboard\python.exe"
if (-not (Test-Path $Py)) { throw "Python 인터프리터 없음: $Py" }

$env:DB_NAME        = "equity"
$env:PGPASSWORD     = "postgres"
$env:PYTHONUTF8     = "1"
$env:PYTHONWARNINGS = "ignore"   # pandas SQLAlchemy UserWarning 소음 억제(로그 정결)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir ("daily_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

# 네이티브(python) 실행 헬퍼 — 5.1에서 2>&1 이 Stop 모드와 충돌하지 않게
# ErrorActionPreference 를 잠시 Continue 로 낮추고 $LASTEXITCODE 로 성공 판정.
function Invoke-Py {
    param([string[]]$PyArgs, [string]$Stage)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Py @PyArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "$Stage 실패 (exit $code) — 로그: $LogFile" }
}

function Run-Stage([string]$name, [scriptblock]$action) {
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "  [$name]" -ForegroundColor Cyan
    Write-Host ("=" * 72)
    $t0 = Get-Date
    & $action
    Write-Host ("  -> {0:F1}s" -f ((Get-Date) - $t0).TotalSeconds) -ForegroundColor DarkGreen
}

# 완료/실패 알림 팝업 (WScript.Shell — 어셈블리 로드 불필요, 자동 타임아웃 지원).
#   Type: 64=정보(i) 16=오류(x).  성공 60초, 실패 무기한(확인 필요) 후 자동닫힘 방지 위해 큰 값.
function Show-Popup([string]$title, [string]$text, [int]$type, [int]$seconds) {
    if ($Quiet) { return }
    try {
        (New-Object -ComObject WScript.Shell).Popup($text, $seconds, $title, $type) | Out-Null
    } catch {
        Write-Host "팝업 표시 실패(무시): $($_.Exception.Message)"
    }
}

Write-Host ("### Daily Pipeline 시작 " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

try {
    # 1) Sync — 라이브 필수 테이블만 증분 (소스 portfolio_db_dev = sync_equity_db.py 기본값)
    Run-Stage "STAGE 1 - Sync (portfolio_db_dev -> local equity)" {
        $a = @("sync_equity_db.py", "--only", "market_data",
               "rt_expected_returns", "rt_asset_metrics")
        if ($Full) { $a += "--full" }
        Invoke-Py -PyArgs $a -Stage "sync_equity_db.py"
    }

    # 2) Daily signals + forward 추적 (catch-up: forward_capital 이후 누락 거래일 자동 백필)
    Run-Stage "STAGE 2 - Daily Signals + Forward (catch-up)" {
        $a = @("daily_signals.py")
        if ($AsOf) { $a += @("--as-of", $AsOf) }
        Invoke-Py -PyArgs $a -Stage "daily_signals.py"
    }

    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host ("  완료. 로그: $LogFile") -ForegroundColor Green
    Write-Host ("=" * 72)

    # 완료 알림 — forward 현황 요약
    $summary = ""
    try {
        $ErrorActionPreference = "Continue"
        $summary = (& $Py "pipeline_summary.py" 2>$null | Out-String).Trim()
        $ErrorActionPreference = "Stop"
    } catch { $summary = "" }
    if (-not $summary) { $summary = "파이프라인이 정상 완료되었습니다." }
    Show-Popup "ATH 일일 파이프라인 ✅ 완료" $summary 64 60
}
catch {
    $err = $_.Exception.Message
    Write-Host "실패: $err" -ForegroundColor Red
    Show-Popup "ATH 일일 파이프라인 ⚠ 실패" "$err`n`n로그: $LogFile" 16 0
    exit 1
}

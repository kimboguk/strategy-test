# register_daily_task.ps1
# ATH 자동 운영 작업을 Windows 작업 스케줄러에 등록 (idempotent — 재실행 시 갱신).
#   ① ATH-Daily-Pipeline (기본 18:00) — sync + daily_signals(catch-up) forward 추적
#   ② ATH-Open-Entry     (기본 09:01) — auto_trade ON 시 개장 자동 진입 (키움 시장가)
#   ③ ATH-Exit-Monitor   (09:00~15:30 5분 주기) — 장중 TP/SL 도달 시 시장가 청산
#
#   .\register_daily_task.ps1                       # 셋 다 등록
#   .\register_daily_task.ps1 -PipelineAt "18:30" -OpenEntryAt "09:05"
#   .\register_daily_task.ps1 -Unregister           # 셋 다 해제

param(
    [string]$PipelineAt  = "18:00",
    [string]$OpenEntryAt = "09:01",
    [switch]$Unregister  = $false
)

$ErrorActionPreference = "Stop"

$PipelineName = "ATH-Daily-Pipeline"
$PipelineScript = "D:\study\finance\trading\strategy-test\daily_pipeline.ps1"
$PipelineDir = "D:\study\finance\trading\strategy-test"

$OpenName = "ATH-Open-Entry"
$OpenScript = "D:\study\finance\trading\dashboard\backend\run_open_entry.ps1"
$OpenDir = "D:\study\finance\trading\dashboard\backend"

$ExitName = "ATH-Exit-Monitor"
$ExitScript = "D:\study\finance\trading\dashboard\backend\run_exit_monitor.ps1"
$ExitDir = "D:\study\finance\trading\dashboard\backend"

function Unregister-One([string]$name) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "등록 해제됨: $name" -ForegroundColor Yellow
    } else {
        Write-Host "작업 없음: $name"
    }
}

function Register-One([string]$name, [string]$script, [string]$dir, [string]$at, [string]$desc) {
    if (-not (Test-Path $script)) { throw "스크립트 없음: $script" }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
        -WorkingDirectory $dir
    $trigger = New-ScheduledTaskTrigger -Daily -At $at
    # StartWhenAvailable: 예약 시각에 PC가 꺼져 있었으면 다음 가용 시 실행.
    #  - 파이프라인: catch-up 과 결합되어 빠진 날 복구
    #  - 개장 진입: 장중 가드가 있어 장 마감 후 실행되면 자동으로 skip(헛주문 없음)
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force -Description $desc | Out-Null
    Write-Host "등록 완료: $name (매일 $at, StartWhenAvailable)" -ForegroundColor Green
}

if ($Unregister) {
    Unregister-One $PipelineName
    Unregister-One $OpenName
    Unregister-One $ExitName
    return
}

Register-One $PipelineName $PipelineScript $PipelineDir $PipelineAt `
    "ATH 일일 파이프라인: portfolio_db_dev 증분 sync + daily_signals(catch-up) forward 추적"
Register-One $OpenName $OpenScript $OpenDir $OpenEntryAt `
    "ATH 개장 자동 진입: auto_trade ON 시 arm된 pending 진입을 키움 시장가로 제출(장중 가드)"

# ③ 장중 청산 감시 — 09:00 시작, 5분 주기로 15:30까지 반복
if (-not (Test-Path $ExitScript)) { throw "스크립트 없음: $ExitScript" }
$exitAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ExitScript`"" `
    -WorkingDirectory $ExitDir
# 반복 트리거: 09:00 시작 → 5분마다 → 6.5시간(15:30) 동안
$exitTrigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$rep = (New-ScheduledTaskTrigger -Once -At "09:00" `
        -RepetitionInterval (New-TimeSpan -Minutes 5) `
        -RepetitionDuration (New-TimeSpan -Hours 6 -Minutes 30)).Repetition
$exitTrigger.Repetition = $rep
$exitSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$exitPrincipal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $ExitName -Action $exitAction -Trigger $exitTrigger `
    -Settings $exitSettings -Principal $exitPrincipal -Force `
    -Description "ATH 장중 청산 감시: 보유종목 TP/SL 도달 시 시장가 매도(5분 주기, 장중 가드)" | Out-Null
Write-Host "등록 완료: $ExitName (매일 09:00~15:30, 5분 주기)" -ForegroundColor Green

Write-Host ""
Get-ScheduledTask -TaskName $PipelineName, $OpenName, $ExitName |
    Select-Object TaskName, State | Format-Table -AutoSize

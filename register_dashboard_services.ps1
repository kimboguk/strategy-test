# register_dashboard_services.ps1
# 대시보드 백엔드/프런트를 "로그온 시 자동 시작 + 상시 실행" 서비스로 등록 (VSCode 무관).
#   ATH-Backend  : uvicorn :8000  (dashboard\backend\run_backend.ps1)
#   ATH-Frontend : Next.js :3000  (dashboard\frontend\run_frontend.ps1)
# hidden VBS 런처로 콘솔 창 없이 백그라운드 실행. 크래시 시 자동 재시작.
#
#   .\register_dashboard_services.ps1              # 등록
#   .\register_dashboard_services.ps1 -Unregister  # 해제

param([switch]$Unregister = $false)

$ErrorActionPreference = "Stop"

$Vbs = "D:\study\finance\trading\strategy-test\run_hidden.vbs"
$svcs = @(
    @{ Name = "ATH-Backend";
       Script = "D:\study\finance\trading\dashboard\backend\run_backend.ps1";
       Dir = "D:\study\finance\trading\dashboard\backend";
       Desc = "대시보드 백엔드(uvicorn :8000) 상시 실행 — 로그온 자동" },
    @{ Name = "ATH-Frontend";
       Script = "D:\study\finance\trading\dashboard\frontend\run_frontend.ps1";
       Dir = "D:\study\finance\trading\dashboard\frontend";
       Desc = "대시보드 프런트(Next.js dev :3000) 상시 실행 — 로그온 자동" }
)

if ($Unregister) {
    foreach ($s in $svcs) {
        if (Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $s.Name -Confirm:$false
            Write-Host "해제됨: $($s.Name)" -ForegroundColor Yellow
        } else { Write-Host "작업 없음: $($s.Name)" }
    }
    return
}

foreach ($s in $svcs) {
    if (-not (Test-Path $s.Script)) { throw "스크립트 없음: $($s.Script)" }
    $action = New-ScheduledTaskAction -Execute "wscript.exe" `
        -Argument "`"$Vbs`" `"$($s.Script)`"" -WorkingDirectory $s.Dir
    # 로그온 시 시작 + PC 켜질 때도(StartWhenAvailable). 상시 실행이라 시간제한 없음(PT0S).
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $s.Name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force -Description $s.Desc | Out-Null
    Write-Host "등록 완료: $($s.Name) (로그온 자동, 상시 실행)" -ForegroundColor Green
}

Write-Host ""
Get-ScheduledTask -TaskName "ATH-Backend", "ATH-Frontend" |
    Select-Object TaskName, State | Format-Table -AutoSize

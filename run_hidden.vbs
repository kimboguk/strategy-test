' run_hidden.vbs
' 콘솔 창을 전혀 띄우지 않고(hidden) PowerShell 스크립트를 실행.
' 스케줄 작업이 다음 형태로 호출:
'   wscript.exe "run_hidden.vbs" "<대상.ps1>"
' → 5분 주기 등 잦은 실행에도 화면 깜빡임 없음. 로그/알림 팝업은 스크립트 내부에서 그대로 동작.

Set sh = CreateObject("WScript.Shell")
If WScript.Arguments.Count = 0 Then WScript.Quit 1
scriptPath = WScript.Arguments(0)
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & scriptPath & """"
' 0 = 창 숨김(깜빡임 없음), True = 종료까지 대기(작업 완료/타임아웃 정상 추적)
WScript.Quit sh.Run(cmd, 0, True)

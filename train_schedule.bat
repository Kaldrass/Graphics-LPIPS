@echo off
setlocal enabledelayedexpansion

REM Force explicit values
set "WORKDIR=D:\These\Graphics-LPIPS"
set "CMD=C:\Users\gauti\AppData\Local\Programs\Python\Python312\python.exe D:\These\Graphics-LPIPS\train.py"
set "TASK_NAME=LPIPS_Train_Once"
set "START_TIME=18:00"

echo START_TIME raw: "%START_TIME%"
echo.

echo Testing schtasks command that will be executed:
echo schtasks /Create /SC ONCE /TN "%TASK_NAME%" /TR "cmd.exe /c \"cd /d %WORKDIR% && %CMD%\"" /ST "%START_TIME%" /F
echo.

pause

schtasks /Create /SC ONCE /TN "%TASK_NAME%" /TR "cmd.exe /c \"cd /d %WORKDIR% && %CMD%\"" /ST "%START_TIME%" /F
pause

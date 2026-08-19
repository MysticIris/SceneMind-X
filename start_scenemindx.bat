@echo off
chcp 65001 >nul
setlocal
title SceneMind-X Launcher

set "SCENEMINDX_ROOT=%~dp0"
set "SCENEMINDX_LAUNCHER=%SCENEMINDX_ROOT%scripts\start_scenemindx.ps1"

if not exist "%SCENEMINDX_LAUNCHER%" (
  echo.
  echo [SceneMind-X] 启动脚本缺失 / Startup helper is missing:
  echo   %SCENEMINDX_LAUNCHER%
  goto :failed
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCENEMINDX_LAUNCHER%" %*
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo [SceneMind-X] 启动未完成，未停止任何已有进程。
echo [SceneMind-X] Startup did not complete. No existing process was stopped.
echo [SceneMind-X] 请查看上方错误，并参阅 README 的“运行环境”与“常见问题”。
echo [SceneMind-X] Read the error above and see Environment and Troubleshooting in README.
if /I not "%SCENEMINDX_START_NO_PAUSE%"=="1" pause
exit /b 1

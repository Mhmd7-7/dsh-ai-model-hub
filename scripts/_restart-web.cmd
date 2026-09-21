@echo off
REM Detached restart watcher for the DSH web GUI.
REM
REM Why this exists: the hub fix lives in code the running server already loaded,
REM so it needs a fresh process. The agent that triggered this restart is a child
REM of that server, so the relaunch must not depend on it — it is spawned
REM detached, waits for the old process to disappear, then starts a new one.

setlocal
set "NODE=C:\Program Files\nodejs\node.exe"
set "BIN=C:\Users\fdgrr\AppData\Roaming\npm\node_modules\@deepseek-ai\dsh\lib\bin.js"
set "LOG=%TEMP%\dsh-web-restart.log"

echo [%DATE% %TIME%] restart watcher started, pid %~1 >> "%LOG%"

:waitloop
tasklist /FI "PID eq %~1" 2>nul | find "%~1" >nul
if not errorlevel 1 (
  timeout /t 1 /nobreak >nul
  goto waitloop
)

echo [%DATE% %TIME%] old server gone; starting a new one >> "%LOG%"
start "" /B "%NODE%" "%BIN%" web --no-open >> "%LOG%" 2>&1
echo [%DATE% %TIME%] relaunch issued >> "%LOG%"

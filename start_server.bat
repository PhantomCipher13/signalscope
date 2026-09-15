@echo off
REM ============================================================
REM  SignalScope — One-Command Demo Launcher
REM  Starts FastAPI + Cloudflare tunnel for live demo
REM  Local:  http://localhost:8000
REM  Public: trycloudflare.com URL (shown below after ~15 sec)
REM ============================================================

echo.
echo ============================================================
echo  SignalScope — Starting Demo
echo ============================================================

REM ── Kill any leftover processes ───────────────────────────
taskkill /F /IM cloudflared.exe /T 2>nul
timeout /t 1 /nobreak >nul

REM ── Load .env if it exists ────────────────────────────────
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" set %%A=%%B
    )
)

REM ── Set device to CPU for stable inference ────────────────
set SIGNALSCOPE_DEVICE=cpu

REM ── Detect python executable ──────────────────────────────
set PYTHON_CMD=python
if exist "C:\Python311\python.exe" set PYTHON_CMD=C:\Python311\python.exe

REM ── Start FastAPI server ──────────────────────────────────
echo [1/2] Starting SignalScope API on port 8000...
start /B "SignalScope API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo       Waiting 15 seconds for model to load...
timeout /t 15 /nobreak >nul

REM ── Health check ─────────────────────────────────────────
echo       Verifying server...
%PYTHON_CMD% -c "import requests; r=requests.get('http://localhost:8000/health',timeout=5); d=r.json(); print('  Model:', 'LOADED' if d.get('model_loaded') else 'FAILED'); print('  Calibration:', 'LOADED' if d.get('calibration_loaded') else 'NOT LOADED'); print('  Architecture:', d.get('architecture','?'))" 2>nul || echo   WARNING: health check failed — server may still be loading

REM ── Start Cloudflare tunnel ───────────────────────────────
echo.
echo [2/2] Starting Cloudflare tunnel for public access...
del /F /Q "%TEMP%\cf_tunnel.log" 2>nul

REM Check if cloudflared.exe is available
if exist "C:\Users\Admin\Downloads\cloudflared.exe" (
    start /B "Cloudflare Tunnel" "C:\Users\Admin\Downloads\cloudflared.exe" tunnel --url http://localhost:8000 --logfile "%TEMP%\cf_tunnel.log"
    echo       Waiting 12 seconds for tunnel...
    timeout /t 12 /nobreak >nul
    echo.
    echo ============================================================
    echo  PUBLIC URL (share this with testers):
    echo ============================================================
    type "%TEMP%\cf_tunnel.log" 2>nul | findstr /I "trycloudflare"
    if errorlevel 1 (
        echo   NOTE: Tunnel URL not yet visible — check %TEMP%\cf_tunnel.log
    )
) else (
    echo   NOTE: cloudflared.exe not found at C:\Users\Admin\Downloads\
    echo   For local demo only, open: http://localhost:8000
)

echo.
echo ============================================================
echo  LOCAL URL:   http://localhost:8000
echo  API DOCS:    http://localhost:8000/docs
echo  ADMIN:       http://localhost:8000/admin/analyses (needs ADMIN_TOKEN)
echo.
echo  To stop: taskkill /F /IM python.exe /T
echo            taskkill /F /IM cloudflared.exe /T
echo ============================================================
pause

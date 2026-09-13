@echo off
REM SignalScope — Start Server + Public Tunnel
REM Run this to bring up the full stack including public URL for testers

echo ============================================================
echo  SignalScope — Starting Local Server + Cloudflare Tunnel
echo ============================================================

REM Kill any leftover processes
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM cloudflared.exe /T 2>nul
timeout /t 2 /nobreak >nul

REM Set environment (edit .env or set here)
REM set SUPABASE_URL=your_url_here
REM set SUPABASE_KEY=your_key_here
set SIGNALSCOPE_DEVICE=auto

echo [1/2] Starting SignalScope API on port 8000...
start /B "SignalScope API" C:\Python311\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo Waiting for model to load (15 seconds)...
timeout /t 15 /nobreak >nul

echo [2/2] Starting Cloudflare tunnel...
del /F /Q "%TEMP%\cf_tunnel.log" 2>nul
start /B "Cloudflare Tunnel" "C:\Users\Admin\Downloads\cloudflared.exe" tunnel --url http://localhost:8000 --logfile "%TEMP%\cf_tunnel.log"

echo Waiting for tunnel URL (12 seconds)...
timeout /t 12 /nobreak >nul

echo.
echo ============================================================
echo  Reading tunnel URL...
echo ============================================================
type "%TEMP%\cf_tunnel.log" | findstr /I "trycloudflare"

echo.
echo ============================================================
echo  Server is running. Share the trycloudflare.com URL above.
echo  Admin feedback: https://[your-url]/admin/feedback
echo  API docs:       http://localhost:8000/docs
echo.
echo  To stop: taskkill /F /IM python.exe /T
echo            taskkill /F /IM cloudflared.exe /T
echo ============================================================
pause

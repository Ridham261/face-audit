@echo off
echo ============================================
echo   Starting Goggles Wrapper API
echo   Environment: face-validation
echo ============================================
echo.
echo POST http://localhost:8000/validate  <-- use this endpoint
echo.
call conda activate face-validation
cd /d D:\audit
python goggles_wrapper_api.py

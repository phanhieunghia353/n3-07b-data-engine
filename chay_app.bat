@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 (
    python -m streamlit run app/app.py --server.headless false
) else (
    py -3.12 -m streamlit run app/app.py --server.headless false
)
set "N3_APP_EXIT=%errorlevel%"
if not "%N3_APP_EXIT%"=="0" (
    echo.
    echo Khong mo duoc ung dung. Xem thong bao loi o tren.
    pause
)
exit /b %N3_APP_EXIT%

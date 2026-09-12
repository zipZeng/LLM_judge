@echo off
chcp 65001 >nul
cd /d D:\Project\shixun3\LLM_judge

where streamlit >nul 2>nul
if errorlevel 1 (
    echo [ERROR] streamlit not found. Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting corpus quality evaluation platform...
echo Browser will open http://localhost:8501
echo Close this window to stop the server.
echo.
streamlit run web/app.py

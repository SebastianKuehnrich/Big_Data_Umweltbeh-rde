@echo off
REM Startet das EPA Air Quality Dashboard
echo Starte EPA Air Quality Dashboard...
echo.

if not exist .venv (
    echo [ERROR] Virtuelle Umgebung nicht gefunden!
    echo Bitte führen Sie zuerst setup.bat aus.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
streamlit run epa_dashboard2.py


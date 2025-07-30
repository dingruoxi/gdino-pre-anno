@echo off
echo Starting Annotation Tool...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check if requirements are installed
echo Checking requirements...
pip install -r requirements.txt

:: Run the application
echo Starting Streamlit application...
streamlit run app.py

pause
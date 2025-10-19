@echo off
title Blood Cell Classification App
echo.
echo 🩸 Blood Cell Classification System
echo ===================================
echo.
echo Starting the application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "models\mobilenetv2_cbam_best.h5" (
    echo ❌ Model file not found: models\mobilenetv2_cbam_best.h5
    echo Please ensure the trained model is in the models directory
    pause
    exit /b 1
)

REM Install requirements (optional)
set /p install_deps=Install/update requirements? (y/n): 
if /i "%install_deps%"=="y" (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
    echo ✅ Requirements installed successfully!
    echo.
)

REM Run the Streamlit app
echo 🚀 Starting Streamlit app...
echo The app will open in your default browser
echo Press Ctrl+C to stop the app
echo.
streamlit run app.py

pause
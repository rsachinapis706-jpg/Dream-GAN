@echo off
echo ===================================================
echo   Dream Analysis AC-TimeGAN - Auto-Runner
echo ===================================================

echo [1/3] Checking Dependencies...
pip install -r requirements.txt

echo [2/3] Verifying Data...
if not exist "data_extracted" (
    echo Extracting Data...
    powershell -command "Expand-Archive -Path '22133105.zip' -DestinationPath 'data_extracted' -Force"
)

echo [3/3] Starting Training (500 Epochs - Optimized Q1 Run)...
python -m src.train

echo.
echo ===================================================
echo Training Complete! Results saved to logs.
echo ===================================================
pause

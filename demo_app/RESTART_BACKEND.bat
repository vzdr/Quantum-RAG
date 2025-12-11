@echo off
echo ========================================
echo  Clearing Python Cache and Restarting
echo ========================================
echo.

REM Kill all Python processes
taskkill /F /IM python.exe 2>nul

REM Go to backend directory
cd /d %~dp0backend

REM Clear ALL cache
echo Clearing cache...
del /s /q __pycache__ 2>nul
rmdir /s /q __pycache__ 2>nul
del /s /q api\__pycache__ 2>nul
rmdir /s /q api\__pycache__ 2>nul
del /s /q models\__pycache__ 2>nul
rmdir /s /q models\__pycache__ 2>nul
del /s /q services\__pycache__ 2>nul
rmdir /s /q services\__pycache__ 2>nul

REM Go to core and clear cache
cd /d %~dp0..\..
del /s /q core\__pycache__ 2>nul
rmdir /s /q core\__pycache__ 2>nul

REM Back to backend
cd /d %~dp0backend

echo.
echo Cache cleared! Starting backend...
echo.

REM Start backend
python main.py

pause

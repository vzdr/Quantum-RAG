@echo off
echo ========================================
echo  Quantum-RAG Investor Demo
echo ========================================
echo.
echo Starting backend and frontend servers...
echo.

REM Start backend in new window
start "Quantum-RAG Backend" cmd /k "cd /d %~dp0backend && python main.py"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak > nul

REM Start frontend in new window
start "Quantum-RAG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Servers starting...
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Opening demo in browser in 5 seconds...
timeout /t 5 /nobreak > nul

start http://localhost:3000

echo.
echo Demo is running! Close the terminal windows to stop.

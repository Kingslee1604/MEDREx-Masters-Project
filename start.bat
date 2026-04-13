@echo off
title MEDREx Launcher
color 0A
cd /d "%~dp0"

echo.
echo  ==========================================
echo    MEDREx - Medical Evidence and Decision
echo    Reasoning eXchange
echo    CSC 590 Masters Project - CSUDH 2026
echo  ==========================================
echo.
echo  Starting Backend API on port 8000...
start "MEDREx Backend" cmd /k "cd /d "%~dp0backend" && uvicorn main:app --port 8000 --reload"

echo  Waiting 8 seconds for backend to start...
timeout /t 8 /nobreak > nul

echo  Starting Frontend UI on port 8501...
start "MEDREx Frontend" cmd /k "cd /d "%~dp0frontend" && streamlit run app.py --server.port 8501"

echo.
echo  ==========================================
echo    Both servers are starting!
echo.
echo    Frontend UI:  http://localhost:8501
echo    Backend API:  http://localhost:8000
echo    API Docs:     http://localhost:8000/docs
echo.
echo    First run trains model (~60 sec)
echo    ChromaDB builds in background after that
echo  ==========================================
echo.
pause

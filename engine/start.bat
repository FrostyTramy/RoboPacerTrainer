@echo off
echo Installing Flask if needed...
pip install flask --quiet
echo.
echo Starting RoboPacer Studio...
echo Open your browser at http://localhost:5000
echo.
python "%~dp0server.py"
pause

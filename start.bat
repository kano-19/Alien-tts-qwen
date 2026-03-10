@echo off
title Alien TTS Qwen
color 09

echo.
echo  ====================================================
echo       Alien TTS Qwen - Iniciando Servidor
echo  ====================================================
echo.

REM Check venv exists
if not exist "venv\Scripts\activate.bat" (
    echo  [ERROR] Entorno virtual no encontrado!
    echo  Por favor ejecuta install.bat primero.
    echo.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Create outputs dir
if not exist "outputs" mkdir outputs

echo  Iniciando servidor Gradio...
echo  Abre tu navegador en: http://localhost:7860
echo.
python app.py

pause

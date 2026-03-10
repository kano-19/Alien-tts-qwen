@echo off
setlocal enabledelayedexpansion
title Alien TTS Qwen - Installer
color 09
echo.
echo  ====================================================
echo       Alien TTS Qwen - Instalador Automatico
echo  ====================================================
echo.

REM ---- Step 1: Find Python 3.12 ----
echo  [1/5] Buscando Python 3.12...
echo.

set PYTHON_CMD=

REM Try py launcher first (best for multiple Python versions)
py -3.12 --version >nul 2>&1
if !errorlevel! equ 0 (
    set PYTHON_CMD=py -3.12
    goto :python_found
)

REM Try common install locations directly
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :python_found
)
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :python_found
)
if exist "C:\Program Files\Python312\python.exe" (
    set "PYTHON_CMD=C:\Program Files\Python312\python.exe"
    goto :python_found
)

REM ---- Python 3.12 not found, install via winget ----
echo  [!] Python 3.12 no encontrado en tu sistema.
echo  [!] Intentando instalar Python 3.12 via winget...
echo.
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
echo.
echo  [OK] Instalacion de Python completada.
echo  [!] Buscando la nueva instalacion...
echo.

REM Update PATH for this session
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
set "PATH=C:\Python312;C:\Python312\Scripts;%PATH%"

REM Try py launcher again
py -3.12 --version >nul 2>&1
if !errorlevel! equ 0 (
    set PYTHON_CMD=py -3.12
    goto :python_found
)

REM Try direct paths again
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :python_found
)
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :python_found
)

REM Try just "python" now
python --version 2>nul | findstr "3.12" >nul 2>&1
if !errorlevel! equ 0 (
    set PYTHON_CMD=python
    goto :python_found
)

REM Still not found
echo.
echo  ====================================================
echo   Python 3.12 se instalo pero no se detecto en PATH.
echo.
echo   SOLUCION: Cierra esta ventana y vuelve a ejecutar
echo   install.bat (a veces el PATH necesita reiniciar).
echo.
echo   Si sigue sin funcionar, descarga manualmente:
echo   https://www.python.org/downloads/release/python-31210/
echo   Marca "Add Python to PATH" al instalar.
echo  ====================================================
echo.
pause
exit /b 1

:python_found
echo  [OK] Python 3.12 encontrado!
echo       Comando: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM ---- Step 2: Create virtual environment ----
echo  [2/5] Creando entorno virtual...
if not exist "venv" (
    %PYTHON_CMD% -m venv venv
    if !errorlevel! neq 0 (
        echo  [ERROR] No se pudo crear el entorno virtual.
        echo.
        pause
        exit /b 1
    )
    echo  [OK] Entorno virtual creado.
) else (
    echo  [OK] Entorno virtual ya existe.
)
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM ---- Step 3: Install PyTorch with CUDA ----
echo  [3/5] Instalando PyTorch con soporte CUDA...
echo         (esto puede tomar unos minutos, espera...)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet 2>nul
if !errorlevel! neq 0 (
    echo  [WARN] CUDA no disponible, instalando version CPU...
    pip install torch torchaudio --quiet
)
echo  [OK] PyTorch instalado.
echo.

REM ---- Step 4: Install dependencies ----
echo  [4/5] Instalando dependencias del proyecto...
pip install -r requirements.txt --quiet
echo  [OK] Dependencias instaladas.
echo.

REM ---- Step 5: Done ----
echo  [5/5] Configuracion final...
echo  [INFO] FlashAttention 2 no es compatible con Windows.
echo         Se usara SDPA (igual de rapido, sin problemas).
echo.

REM Create directories
if not exist "outputs" mkdir outputs

REM ---- System Check ----
echo  ====================================================
echo       Informacion del Sistema
echo  ====================================================
echo.
python system_check.py
echo.

echo  ====================================================
echo.
echo       INSTALACION COMPLETADA EXITOSAMENTE!
echo.
echo   Ejecuta 'start.bat' para iniciar el servidor.
echo.
echo  ====================================================
echo.
echo  Presiona una tecla para cerrar...
pause >nul
exit /b 0

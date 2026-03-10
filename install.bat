@echo off
title Alien TTS Qwen - Installer
color 09
chcp 65001 >nul 2>&1

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║       🛸 Alien TTS Qwen - Instalador 🛸          ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM ─── Step 1: Find Python 3.12 ─────────────────────────────────────────
echo  [1/5] Buscando Python 3.12...
echo.

set PYTHON_CMD=

REM Try "python" first
python --version 2>nul | findstr "3.12" >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :python_found
)

REM Try "python3" 
python3 --version 2>nul | findstr "3.12" >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    goto :python_found
)

REM Try "py -3.12"
py -3.12 --version 2>nul | findstr "3.12" >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.12
    goto :python_found
)

REM Try common install paths
if exist "C:\Python312\python.exe" (
    set PYTHON_CMD=C:\Python312\python.exe
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :python_found
)

REM Python 3.12 not found - try to install via winget
echo  [!] Python 3.12 no encontrado. Intentando instalar...
echo.
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements 2>nul
if %errorlevel% equ 0 (
    echo  [OK] Python 3.12 instalado. Reiniciando deteccion...
    REM Refresh PATH
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    if exist "%PYTHON_CMD%" goto :python_found
)

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║  ERROR: No se pudo encontrar/instalar Python 3.12║
echo  ║                                                  ║
echo  ║  Descargalo manualmente:                          ║
echo  ║  https://www.python.org/downloads/release/        ║
echo  ║  python-3120/                                     ║
echo  ║                                                  ║
echo  ║  Asegurate de marcar "Add to PATH" al instalar.  ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause
exit /b 1

:python_found
echo  [OK] Python encontrado: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM ─── Step 2: Create virtual environment ────────────────────────────────
echo  [2/5] Creando entorno virtual...
if not exist "venv" (
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo  [ERROR] No se pudo crear el entorno virtual.
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

REM ─── Step 3: Install PyTorch with CUDA ─────────────────────────────────
echo  [3/5] Instalando PyTorch con soporte CUDA...
echo         (esto puede tomar unos minutos)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet 2>nul
if %errorlevel% neq 0 (
    echo  [WARN] CUDA version no disponible, instalando CPU version...
    pip install torch torchaudio --quiet
)
echo  [OK] PyTorch instalado.
echo.

REM ─── Step 4: Install dependencies ─────────────────────────────────────
echo  [4/5] Instalando dependencias del proyecto...
pip install -r requirements.txt --quiet
echo  [OK] Dependencias instaladas.
echo.

REM ─── Step 5: Optional FlashAttention ───────────────────────────────────
echo  [5/5] Intentando instalar FlashAttention 2 (opcional)...
pip install flash-attn --no-build-isolation --quiet 2>nul
if %errorlevel% neq 0 (
    echo  [INFO] FlashAttention 2 no disponible en Windows.
    echo         Se usara atencion estandar (funciona igual).
) else (
    echo  [OK] FlashAttention 2 instalado.
)
echo.

REM Create directories
if not exist "outputs" mkdir outputs

REM ─── System Check ─────────────────────────────────────────────────────
echo  ╔══════════════════════════════════════════════════╗
echo  ║           Informacion del Sistema                ║
echo  ╚══════════════════════════════════════════════════╝
echo.
python system_check.py
echo.

echo  ╔══════════════════════════════════════════════════╗
echo  ║        ✅ Instalacion Completada! ✅              ║
echo  ║                                                  ║
echo  ║  Ejecuta 'start.bat' para iniciar el servidor.   ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause

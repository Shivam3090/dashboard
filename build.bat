@echo off

REM Install Rust if not already installed
where rustc >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    call "%USERPROFILE%\.cargo\env.bat"
)

REM Ensure Cargo environment is sourced
call "%USERPROFILE%\.cargo\env.bat"

REM Install Python dependencies
pip install -r requirements.txt
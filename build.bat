@echo off

REM Activate your Anaconda environment
call "C:\Users\31220\anaconda3\Scripts\activate.bat" new_env

REM Set PATH to Rust and Cargo
set PATH=%PATH%;C:\Users\31220\.cargo\bin

REM Install Python dependencies
pip install -r requirements.txt

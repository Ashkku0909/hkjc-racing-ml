@echo off
chcp 65001 >nul
title HKJC Racing ML Setup
color 0B

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║          HKJC Racing ML Pipeline Setup       ║
echo  ║        One-Click Installer for Windows       ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: Check Python
echo  [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ❌ Python not found! Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  ✅ Python %%v found

:: Install pip packages
echo.
echo  [2/5] Installing Python packages...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo  ⚠️  Some packages failed. Try: pip install -r requirements.txt
) else (
    echo  ✅ Dependencies installed
)

:: Install Playwright
echo.
echo  [3/5] Installing Playwright browser...
playwright install chromium
if %errorlevel% neq 0 (
    echo  ⚠️  Playwright install failed. Run: playwright install chromium
) else (
    echo  ✅ Playwright ready
)

:: Setup .env
echo.
echo  [4/5] Setting up configuration...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo  ✅ Created .env from .env.example
        echo  ⚠️  IMPORTANT: Edit .env and add your API keys!
        start notepad .env
    ) else (
        echo  ⚠️  .env.example not found, creating blank .env
        echo GEMINI_API_KEY=your_key_here> .env
        echo DATABASE_URL=postgresql+asyncpg://postgres:pass@localhost:5432/hkjc_db>> .env
        echo DISCORD_TOKEN=your_token_here>> .env
    )
) else (
    echo  ✅ .env already exists
)

:: Done
echo.
echo  [5/5] Setup complete!
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║           All set! Launch with:              ║
echo  ║                                              ║
echo  ║       streamlit run app.py                   ║
echo  ║                                              ║
echo  ║  Then open http://localhost:8501             ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  ⚠️  Don't forget to edit .env with your API keys!
echo.
pause

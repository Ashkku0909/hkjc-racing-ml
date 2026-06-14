#!/bin/bash
set -e

echo ""
echo " ╔══════════════════════════════════════════════╗"
echo " ║          HKJC Racing ML Pipeline Setup       ║"
echo " ║        One-Click Installer for Linux/Mac     ║"
echo " ╚══════════════════════════════════════════════╝"
echo ""

# Check Python
echo " [1/5] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo " ❌ Python not found! Install Python 3.10+"
    exit 1
fi
echo " ✅ $($PYTHON --version)"

# Install pip packages
echo ""
echo " [2/5] Installing Python packages..."
$PYTHON -m pip install -r requirements.txt -q || {
    echo " ⚠️  Some packages failed: pip install -r requirements.txt"
}

# Install Playwright
echo ""
echo " [3/5] Installing Playwright browser..."
$PYTHON -m playwright install chromium || {
    echo " ⚠️  Playwright install failed: playwright install chromium"
}

# Setup .env
echo ""
echo " [4/5] Setting up configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo " ✅ Created .env from .env.example"
        echo " ⚠️  IMPORTANT: Edit .env and add your API keys!"
    else
        echo "GEMINI_API_KEY=your_key_here" > .env
        echo "DATABASE_URL=postgresql+asyncpg://postgres:pass@localhost:5432/hkjc_db" >> .env
        echo "DISCORD_TOKEN=your_token_here" >> .env
    fi
else
    echo " ✅ .env already exists"
fi

# Done
echo ""
echo " [5/5] Setup complete!"
echo ""
echo " ╔══════════════════════════════════════════════╗"
echo " ║           All set! Launch with:              ║"
echo " ║                                              ║"
echo " ║       streamlit run app.py                   ║"
echo " ║                                              ║"
echo " ║  Then open http://localhost:8501             ║"
echo " ╚══════════════════════════════════════════════╝"
echo ""
echo " ⚠️  Don't forget to edit .env with your API keys!"
echo ""

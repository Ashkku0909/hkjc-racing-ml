# HKJC Horse Racing ML Pipeline 🏇

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-3C8D5F.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Bot-Discord-5865F2.svg?logo=discord&logoColor=white)](https://discord.com)
[![Gemini](https://img.shields.io/badge/AI-Gemini-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev)

**AI-powered horse racing data analysis — scraping, ML modeling & an interactive web dashboard**

</div>

<br>

> ⚠️ **EDUCATIONAL USE ONLY** — For study & research. NOT for gambling. [Full Disclaimer →](DISCLAIMER.md)

---

## ✨ Why This Project?

<table>
<tr>
<td width="50%">

### 🎯 Features
- 🤖 **AI Race Analysis** via Google Gemini
- 📊 **Interactive Web Dashboard** (Streamlit)
- 🧠 **30+ Engineered Features** — speed figures, win rates, pace, ROI
- 🎯 **LightGBM Win Probability Model** — walk-forward backtested
- 🔌 **Discord Bot** — real-time race discussion with VLM support

</td>
<td width="50%">

### ⚡ Developer Experience
- **One-click setup** — `setup.bat` / `setup.sh`
- **Beautiful Web UI** — no coding needed
- **Production-grade ML** — calibration, power-law, simultaneous Kelly
- **Real scraping** — Playwright + stealth, rate-limit safe
- **Clean code** — documented, modular, testable

</td>
</tr>
</table>

---

## ⚡ Quick Start

| Platform | Command |
|----------|---------|
| **Windows** | Double-click `setup.bat` or `.\setup.bat` |
| **Mac / Linux** | `chmod +x setup.sh && ./setup.sh` |

Then launch the web app:

```bash
streamlit run app.py
```

Open **http://localhost:8501** 🎉

<details>
<summary>📋 Or do it manually...</summary>

```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Edit .env with your API keys
streamlit run app.py
```
</details>

### 📱 Track from your phone (PC stays home)

Double-click **`start_online.bat`** (or run `powershell -ExecutionPolicy Bypass -File start_online.ps1`).
It starts Streamlit on `0.0.0.0` and opens a **Cloudflare quick tunnel** — it prints a
public `https://…trycloudflare.com` URL (works anywhere, no router port-forwarding,
no account needed). Also prints the LAN URL for same-WiFi phones:

- **Same WiFi:** `http://192.168.0.214:8501` (if unreachable, allow inbound port 8501 once, as admin:
  `New-NetFirewallRule -DisplayName "HKJC Quant Terminal 8501" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow`)
- **Anywhere (4G/5G):** the printed `https://…trycloudflare.com` URL

> The tunnel URL is public — anyone with the link can view the board.

<details>
<summary>📋 Manual online launch</summary>

```bash
streamlit run app.py --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
# second window:
%LOCALAPPDATA%\cloudflared\cloudflared.exe tunnel --url http://localhost:8501
```
</details>

---

## 🖥️ Web App Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | Stats overview, top horses chart, race timeline, quick search |
| 🔍 **Race Lookup** | Find any race → full field with model probabilities + visual chart |
| 🐴 **Horse Analysis** | Career stats, performance chart, win/place metrics |
| 🤖 **AI Analysis** | Gemini-powered natural-language breakdown of any race or horse |
| 📁 **Data Overview** | Raw data browser, column reference, CSV export |

---

## 🏗️ Architecture

```
 User → [Streamlit UI / Discord Bot] → [Gemini AI] → [ML Model (LightGBM)]
                                                    ↓
                                              [Predictions + EV]
                                                    ↓
                                          [30+ Features Engineered]
                                                    ↓
                                         [HKJC Scraper (Playwright)]
                                                    ↓
                                          [PostgreSQL / CSV Storage]
```

---

## 🤖 Discord Bot (Optional)

```bash
python bot/discord_bot.py
```

| Command | Example |
|---------|---------|
| `!analyze GOLDEN SIXTY` | Career deep-dive |
| `!analyze 2024-12-08_Race7` | Race analysis |
| `!live 2026-06-14 ST 5` | Live odds + model overlay |
| `!scan_overlays 2026-06-14 ST` | Full-day value scan |

Also: DM the bot or @mention with a race card image → VLM analysis.

---

## 🔧 Configuration

`.env.example` → copy to `.env`:

```env
GEMINI_API_KEY=your_key_here        # From aistudio.google.com/apikey
DATABASE_URL=postgresql+asyncpg://postgres:pass@localhost:5432/hkjc_db
DISCORD_TOKEN=your_bot_token_here   # From discord.com/developers
```

| Key | Required? | Get it from |
|-----|-----------|-------------|
| `GEMINI_API_KEY` | For AI | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `DATABASE_URL` | For DB | Local Postgres |
| `DISCORD_TOKEN` | For Bot | [discord.com/developers](https://discord.com/developers/applications) |

---

## 🧪 Pipeline (for developers)

```bash
python main.py                          # Full pipeline
python scraping/scraper.py              # Scrape only
python data_pipeline/feature_engineering.py  # Engineer features
python modeling/model_training.py       # Train + backtest
python modeling/backtesting.py          # Backtest only
```

---

## 📁 Project Structure

```
├── app.py                  # 🌐 Web UI — launch this!
├── setup.bat / setup.sh    # ⚡ One-click install
├── main.py                 # Pipeline runner
├── .env.example            # API key template
├── DISCLAIMER.md           # ⚠️ Read first
├── scraping/               # 🕷️ HKJC data collection
├── data_pipeline/          # 🔧 Feature engineering
├── modeling/               # 🧠 LightGBM + backtesting
├── bot/                    # 🤖 Discord bot + AI prompts
└── testing/                # 🧪 Tests
```

---

## 📄 Legal

- **Code**: [MIT License](LICENSE)
- **Usage**: [DISCLAIMER.md](DISCLAIMER.md) — **Educational study only.** NOT for gambling. Zero liability.

---

<div align="center">
  <sub>Built for the ML & horse racing data community ⚡</sub>
</div>

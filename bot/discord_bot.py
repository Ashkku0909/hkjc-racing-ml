import sys
import os
import time
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discord
from discord.ext import commands, tasks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scraping.live_scraper import (
    scrape_live_odds, get_odds_baseline, get_labelled_snapshot,
    fetch_meeting_schedule, browser_manager, scrape_stalled,
    BASELINE_LEAD_SECONDS,
)
from bot.analyzer_service import (
    load_data, get_data, get_intent_features,
    merge_live_odds_with_predictions, estimate_probabilities_from_history,
    calculate_smart_money_metrics, apply_smart_money_bayesian_update,
    SMART_MONEY_CENTER, SMART_STEAMER_THRESHOLD, SMART_DRIFTER_THRESHOLD,
)
from modeling.exotics_pricing import build_core_satellite_bets, market_win_probs, quinella_place_prob
from datetime import datetime

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Initialize Discord Bot
intents = discord.Intents.default()
intents.message_content = True

# Use proxy if defined in environment
proxy_url = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
bot = commands.Bot(command_prefix="!", intents=intents, proxy=proxy_url)

bot.remove_command("help")

# Overlay band must stay in sync with the Kelly optimizer in modeling/model_training.py
# EV here is the decay-adjusted expected_value (true_prob * odds * decay - 1)
# Tuned via the 2021-2026 walk-forward sensitivity sweep (Pareto-optimal combo)
KELLY_MIN_EV = 0.22        # EV threshold 0.22 <-> odds-ratio EV > 1.22
KELLY_MIN_ODDS = 4.5
KELLY_MAX_ODDS = 8.0


def fmt(value, spec='{:.2f}', fallback='—'):
    """NaN-safe number formatter for embed fields."""
    if value is None:
        return fallback
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return fallback
    except (TypeError, ValueError):
        return str(value)
    return spec.format(v)


# =====================================================================
# Automated Pre-Race Alert Daemon (smart money triggers, no LLM)
# =====================================================================
ALERT_POLL_SECONDS = 45        # daemon tick rate
ALERT_FORMATION_SECONDS = 10 * 60  # T-10m market-formation push
ALERT_BASELINE_SECONDS = 15 * 60   # establish baseline at T-15m
ALERT_EVAL_SECONDS = 3 * 60        # T-3m actionable decision window
ALERT_SLIP_SECONDS = 45            # T-30s slippage/collapse check window
ALERT_T0_SECONDS = 12              # best-effort T-0 close capture window
SCRAPE_TIMEOUT_SECONDS = 15        # hard timeout on every live poll (anti-freeze)
WATCHDOG_STALL_SECONDS = 180       # respawn browser if a scrape stalls longer

_auto_alerts_enabled = False   # auto-enabled at startup when channel is set
_alerted_races = {}    # (date, venue) -> set(race_no) prime alert already sent
_baselined_races = {}  # (date, venue) -> set(race_no) baseline recorded
_formation_sent = {}   # (date, venue) -> set(race_no) T-10m formation sent
_slip_sent = {}        # (date, venue) -> set(race_no) T-30s check done
_t0_captured = {}      # (date, venue) -> set(race_no) T-0 close snapshot taken
_flagged_horses = {}   # (date, venue) -> {race_no: [horse_name, ...]}


def expected_venues(today=None) -> tuple:
    """Auto venue detection by weekday: Wednesday -> Happy Valley, otherwise
    Sha Tin first. The daemon still probes both, this only sets the order."""
    now = today or datetime.now().astimezone()
    if now.weekday() == 2:      # Wednesday night meeting = HV
        return ("HV", "ST")
    return ("ST", "HV")         # Sat/Sun -> Sha Tin first


def get_alert_channel():
    """Resolves DISCORD_ALERT_CHANNEL_ID to a channel, or None if unset/invalid."""
    channel_id = os.getenv("DISCORD_ALERT_CHANNEL_ID", "").strip()
    if not channel_id:
        return None
    try:
        return bot.get_channel(int(channel_id))
    except ValueError:
        return None


def get_alert_ping():
    """Optional ping prefix from env: 'here' -> @here, digits -> <@&role_id>."""
    ping = os.getenv("DISCORD_ALERT_PING", "").strip()
    if ping.lower() == "here":
        return "@here"
    if ping.isdigit():
        return f"<@&{ping}>"
    return ""


def select_prime_value_bets(merged, ev_col='live_expected_value', smart_col='smart_money_score'):
    """Rows triggering [PRIME VALUE BET]: odds in [4.5, 8.0], EV_adj >= 1.22
    (decay-adjusted EV >= 0.22), smart money score >= 50. NaN-safe."""
    if 'win_odds' not in merged.columns or ev_col not in merged.columns:
        return merged.iloc[0:0]
    odds = pd.to_numeric(merged['win_odds'], errors='coerce')
    ev = pd.to_numeric(merged[ev_col], errors='coerce')
    smart = pd.to_numeric(merged.get(smart_col), errors='coerce').fillna(50.0)
    mask = (odds.between(KELLY_MIN_ODDS, KELLY_MAX_ODDS)
            & (ev >= KELLY_MIN_EV)
            & (smart >= 50.0))
    return merged[mask.fillna(False)]


async def _scrape_safe(date_str: str, venue: str, race_no: int,
                       time_to_post: float = None):
    """Live poll with a hard SCRAPE_TIMEOUT_SECONDS deadline (anti-freeze)."""
    return await asyncio.wait_for(
        scrape_live_odds(date_str, venue, race_no, time_to_post=time_to_post),
        timeout=SCRAPE_TIMEOUT_SECONDS)


def _merge_and_score(live_df, date_str: str, venue: str, race_no: int):
    """Shared live pipeline: model merge -> (T-15m baseline) smart money score
    -> Bayesian live-prob update. Never touches final dividends/results."""
    baseline, baseline_age, baseline_mature = get_odds_baseline(
        date_str, venue, race_no, min_age_seconds=ALERT_BASELINE_SECONDS)

    df = get_data()
    race_id = f"{date_str}_Race{race_no}"
    if df is not None and len(df) > 0:
        pred_df = df[df['race_id'] == race_id]
        if len(pred_df) > 0:
            merged = merge_live_odds_with_predictions(live_df, pred_df)
        else:
            merged = estimate_probabilities_from_history(live_df, df)
    else:
        merged = live_df.copy()

    if baseline is not None and len(baseline):
        merged = calculate_smart_money_metrics(merged, baseline)
        merged = apply_smart_money_bayesian_update(merged)
    else:
        merged['smart_money_score'] = SMART_MONEY_CENTER
        merged['flow_signal'] = '➖ STABLE'
    return merged, baseline, baseline_age, baseline_mature


async def _send(channel, embed: discord.Embed) -> bool:
    ping = get_alert_ping()
    try:
        await channel.send(content=ping if ping else None, embed=embed)
        return True
    except Exception as e:
        print(f"[ALERT] Send failed: {e}")
        return False


async def send_formation_embed(channel, date_str: str, venue: str, race_no: int,
                               minutes_to: float) -> bool:
    """Stage 1 (T-10m): market formation - Top 3 by model prob, barrier,
    current odds and the T-15m opening baseline odds."""
    live_df = await _scrape_safe(date_str, venue, race_no,
                                 time_to_post=minutes_to * 60.0)
    if live_df is None or len(live_df) == 0:
        return False
    merged, baseline, _, _ = _merge_and_score(live_df, date_str, venue, race_no)

    prob_col = 'true_prob' if 'true_prob' in merged.columns else None
    if prob_col is not None and merged[prob_col].notna().any():
        ranked = merged[merged[prob_col].notna()].sort_values(prob_col, ascending=False)
    elif merged['win_odds'].notna().any():
        ranked = merged[merged['win_odds'].notna()].sort_values('win_odds')
    else:
        return False
    top = ranked.head(3)

    open_map = {}
    if baseline is not None and len(baseline):
        for _, r in baseline.iterrows():
            try:
                open_map[int(r['horse_number'])] = r['win_odds']
            except (TypeError, ValueError):
                pass

    embed = discord.Embed(
        title=f"🌐 {venue} R{race_no} — Market Formation (T-{max(minutes_to, 0):.0f}m)",
        description=f"{date_str} · Top 3 by model probability · smart-money baseline armed",
        color=0x1d3557)
    for _, r in top.iterrows():
        name = r.get('horse_name', '?')
        prob = r.get('true_prob') if prob_col else None
        odds = r.get('win_odds')
        opening = open_map.get(r.get('horse_number'))
        value = (f"🛡️ Barrier {fmt(r.get('barrier_draw'), '{:.0f}')} · "
                 f"🎯 P {fmt(prob * 100 if prob is not None else None, '{:.1f}%')} · "
                 f"📊 Current {fmt(odds)} · "
                 f"🔓 Opening {fmt(opening)}")
        embed.add_field(name=f"{int(r.get('horse_number')) if pd.notna(r.get('horse_number')) else '?'} · {name}",
                        value=value, inline=False)
    return await _send(channel, embed)


async def send_slip_check_embed(channel, date_str: str, venue: str, race_no: int,
                                flagged_names) -> bool:
    """Stage 3 (T-30s): odds shift T-3m -> T-30s for previously flagged horses.
    🟢 Steaming (compressed) vs ⚠️ Price Collapsed (drifted out) vs stable."""
    if not flagged_names:
        return False
    t3 = get_labelled_snapshot(date_str, venue, race_no, target_seconds=180.0,
                               window=(90.0, 330.0))
    if t3 is None or len(t3) == 0:
        return False
    dec_map = {}
    for _, r in t3.iterrows():
        try:
            dec_map[int(r['horse_number'])] = float(r['win_odds'])
        except (TypeError, ValueError):
            pass

    live_df = await _scrape_safe(date_str, venue, race_no, time_to_post=30.0)
    if live_df is None or len(live_df) == 0:
        return False

    embed = discord.Embed(
        title=f"⏱️ {venue} R{race_no} — T-30s Slippage Check",
        description="Odds move T-3m → T-30s on flagged runners",
        color=0x457b9d)
    for name in flagged_names:
        row = live_df[live_df['horse_name'].astype(str).str.upper() == str(name).upper()]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        try:
            hn = int(r['horse_number'])
            o30 = float(r['win_odds'])
            o3 = dec_map.get(hn)
        except (TypeError, ValueError):
            continue
        if o3 is None or o30 is None or o3 <= 0:
            continue
        move = (o30 / o3 - 1.0) * 100.0
        if move <= -3.0:
            signal = '🟢 STEAMING'
        elif move >= 3.0:
            signal = '⚠️ PRICE COLLAPSED'
        else:
            signal = '➖ STABLE'
        embed.add_field(name=str(name),
                        value=f"T-3m {fmt(o3)} → T-30s {fmt(o30)} ({move:+.1f}%) · {signal}",
                        inline=False)
    if not embed.fields:
        return False
    return await _send(channel, embed)


async def evaluate_race_for_alert(channel, date_str, venue, race_no, minutes_to):
    """Stage 2 (T-3m): smart money evaluation + PRIME VALUE alert (once/race).
    Returns True if an alert was dispatched; records flagged names for the
    T-30s slippage check."""
    live_df = await _scrape_safe(date_str, venue, race_no,
                                 time_to_post=minutes_to * 60.0)
    if live_df is None or len(live_df) == 0:
        return False

    merged, baseline, baseline_age, baseline_mature = _merge_and_score(
        live_df, date_str, venue, race_no)
    ev_col = ('live_expected_value'
              if baseline is not None and len(baseline) and 'live_expected_value' in merged.columns
              else ('expected_value' if 'expected_value' in merged.columns else None))
    if ev_col is None:
        return False
    hits = select_prime_value_bets(merged, ev_col=ev_col)
    if len(hits) == 0:
        return False

    # Remember flagged runners for the T-30s slippage/collapse check
    flagged = _flagged_horses.setdefault((date_str, venue), {})
    flagged.setdefault(race_no, []).extend(
        [n for n in hits['horse_name'].astype(str).tolist()
         if n not in flagged.get(race_no, [])])

    embed = discord.Embed(
        title=f"🚨 PRIME VALUE BET ALERT — {venue} R{race_no}",
        description=(f"{date_str} · T-{max(minutes_to, 0):.0f}m to post · "
                     f"{len(hits)} qualifying horse(s)"),
        color=0xe63946)
    for _, r in hits.head(6).iterrows():
        prob = r.get('live_prob', r.get('true_prob'))
        ev = r.get('live_expected_value', r.get('expected_value'))
        odds = r.get('win_odds')
        score = float(r.get('smart_money_score', 50.0) or 50.0)
        decay = float(r.get('decay_factor', 1.0) or 1.0)
        try:
            ev_ratio = float(prob) * float(odds) * decay
            kelly_full = max((ev_ratio - 1.0) / (float(odds) - 1.0), 0.0) if float(odds) > 1.0 else 0.0
            kelly_stake = min(kelly_full * 0.15, 0.02)
            kelly_txt = f" · Kelly {kelly_stake * 100:.1f}%"
        except (TypeError, ValueError):
            kelly_txt = ""
        value = (f"🛡️ Barrier {fmt(r.get('barrier_draw'), '{:.0f}')} · "
                 f"🎯 P {fmt(prob * 100 if prob is not None else None, '{:.1f}%')} · "
                 f"📊 Odds {fmt(odds)} · "
                 f"💡 Smart {fmt(score, '{:.0f}')} {r.get('flow_signal', '➖ STABLE')} · "
                 f"💰 EV {fmt(ev, '{:+.2f}')}{kelly_txt}")
        embed.add_field(name=f"{r.get('horse_name', '?')}", value=value, inline=False)
    if not baseline_mature:
        embed.set_footer(text=f"Baseline age {baseline_age / 60.0:.0f}m (<15m, scores approximate)")

    sent = await _send(channel, embed)
    if sent:
        print(f"[ALERT] Sent prime value alert for {date_str} {venue} R{race_no} "
              f"({len(hits)} hits) to #{getattr(channel, 'name', '?')}")
    return sent


@tasks.loop(seconds=ALERT_POLL_SECONDS)
async def pre_race_alert_daemon():
    """Zero-touch three-stage pre-race surveillance worker.

    Stage 1 T-10m : market formation embed (Top 3 by model prob + opening odds)
    Stage 2 T-3m  : PRIME VALUE alert (EV>=1.22, odds 4.5-8.0, smart>=50)
    Stage 3 T-30s : slippage/collapse check on flagged runners (T-3m -> T-30s)
    Baselines are armed at T-15m; T-0 close snapshots are captured for audit.
    """
    if not _auto_alerts_enabled:
        return
    channel = get_alert_channel()
    if channel is None:
        print("[ALERT] DISCORD_ALERT_CHANNEL_ID not set/invalid - daemon idle.")
        return

    # --- Heartbeat watchdog: respawn a stalled browser ---
    try:
        if scrape_stalled(WATCHDOG_STALL_SECONDS):
            print("[ALERT] Watchdog: scrape stalled >3min - respawning browser.")
            await browser_manager.recycle()
    except Exception as e:
        print(f"[ALERT] Watchdog error: {e}")

    now = datetime.now().astimezone()
    date_str = now.strftime("%Y-%m-%d")

    # Prune stale state from previous days
    for _dict in (_alerted_races, _baselined_races, _formation_sent,
                  _slip_sent, _t0_captured, _flagged_horses):
        for key in [k for k in _dict if k[0] != date_str]:
            _dict.pop(key, None)

    for venue in expected_venues(now):
        schedule = await asyncio.to_thread(fetch_meeting_schedule, date_str, venue)
        if not schedule:
            continue
        key = (date_str, venue)
        alerted = _alerted_races.setdefault(key, set())
        baselined = _baselined_races.setdefault(key, set())
        formed = _formation_sent.setdefault(key, set())
        slipped = _slip_sent.setdefault(key, set())
        t0_taken = _t0_captured.setdefault(key, set())
        flags = _flagged_horses.setdefault(key, {})

        for race in schedule:
            race_no = race['race_no']
            post_time = race['post_time']
            if post_time.tzinfo is None:
                post_time = post_time.replace(tzinfo=now.tzinfo)
            minutes_to = (post_time.astimezone() - now).total_seconds() / 60.0

            # --- T-15m: arm the odds baseline (persisted, labelled) ---
            if 0 < minutes_to <= ALERT_BASELINE_SECONDS / 60.0 and race_no not in baselined:
                try:
                    live_df = await _scrape_safe(date_str, venue, race_no,
                                                 time_to_post=minutes_to * 60.0)
                    if live_df is not None and len(live_df):
                        baselined.add(race_no)
                        print(f"[ALERT] Baseline armed for {date_str} {venue} R{race_no} "
                              f"(T-{minutes_to:.0f}m)")
                except asyncio.TimeoutError:
                    print(f"[ALERT] Baseline poll timed out for {venue} R{race_no}")
                except Exception as e:
                    print(f"[ALERT] Baseline poll failed for {venue} R{race_no}: {e}")

            # --- Stage 1 / T-10m: market formation push (once) ---
            if 0 < minutes_to <= ALERT_FORMATION_SECONDS / 60.0 and race_no not in formed:
                try:
                    if await send_formation_embed(channel, date_str, venue, race_no, minutes_to):
                        formed.add(race_no)
                except asyncio.TimeoutError:
                    print(f"[ALERT] Formation poll timed out for {venue} R{race_no}")
                except Exception as e:
                    print(f"[ALERT] Formation failed for {venue} R{race_no}: {e}")

            # --- Stage 2 / T-3m: actionable PRIME VALUE alert (once) ---
            if 0 < minutes_to <= ALERT_EVAL_SECONDS / 60.0 and race_no not in alerted:
                try:
                    if await evaluate_race_for_alert(channel, date_str, venue, race_no, minutes_to):
                        alerted.add(race_no)
                except asyncio.TimeoutError:
                    print(f"[ALERT] Eval poll timed out for {venue} R{race_no}")
                except Exception as e:
                    print(f"[ALERT] Eval failed for {venue} R{race_no}: {e}")

            # --- Stage 3 / T-30s: slippage & collapse check (once) ---
            if 0 < minutes_to <= ALERT_SLIP_SECONDS / 60.0 and race_no not in slipped:
                slipped.add(race_no)  # attempt once per race
                try:
                    await send_slip_check_embed(channel, date_str, venue, race_no,
                                                flags.get(race_no, []))
                except asyncio.TimeoutError:
                    print(f"[ALERT] Slip poll timed out for {venue} R{race_no}")
                except Exception as e:
                    print(f"[ALERT] Slip check failed for {venue} R{race_no}: {e}")

            # --- T-0: best-effort pre-close capture (execution audit) ---
            if 0 < minutes_to <= ALERT_T0_SECONDS / 60.0 and race_no not in t0_taken:
                try:
                    await _scrape_safe(date_str, venue, race_no, time_to_post=0.0)
                    t0_taken.add(race_no)
                    print(f"[ALERT] T-0 close snapshot taken for {venue} R{race_no}")
                except Exception as e:
                    print(f"[ALERT] T-0 capture failed for {venue} R{race_no}: {e}")


@pre_race_alert_daemon.before_loop
async def _before_alert_daemon():
    await bot.wait_until_ready()


@bot.command(name="auto_alerts")
@commands.has_permissions(administrator=True)
async def auto_alerts_cmd(ctx, state: str = None):
    """Toggle the automated pre-race alert daemon (admin only).
    Usage: !auto_alerts on | off | status"""
    global _auto_alerts_enabled
    channel_ok = get_alert_channel() is not None
    ping = get_alert_ping()
    if state is None:
        ping_txt = 'none'
        if ping == '@here':
            ping_txt = '@here'
        elif ping.startswith('<@&'):
            ping_txt = f"role {ping.strip('<@&>')}"
        await ctx.send(
            f"🔔 **Auto-alerts:** {'✅ ON' if _auto_alerts_enabled else '⛔ OFF'} · "
            f"channel {'✅ configured' if channel_ok else '❌ DISCORD_ALERT_CHANNEL_ID not set'} · "
            f"ping: {ping_txt}")
        return
    s = state.lower().strip()
    if s in ("on", "enable", "start"):
        _auto_alerts_enabled = True
        if not pre_race_alert_daemon.is_running():
            pre_race_alert_daemon.start()
        await ctx.send("🔔 Auto-alerts **enabled** (poll every 45s, baseline T-15m, eval T-3m).")
    elif s in ("off", "disable", "stop"):
        _auto_alerts_enabled = False
        await ctx.send("⛔ Auto-alerts **disabled**.")
    else:
        await ctx.send("Usage: `!auto_alerts on` / `!auto_alerts off` / `!auto_alerts` (status)")

@bot.event
async def on_ready():
    global _auto_alerts_enabled
    load_data()
    print(f'Logged in as {bot.user.name}')
    print('Pure quant mode active (no LLM/VLM). Use !live, !scan_overlays, !analyze.')
    print('⚠️ EDUCATIONAL USE ONLY — Not for actual gambling. See DISCLAIMER.md')
    # Zero-touch startup: alerts auto-enable whenever an alert channel is set.
    # Use `!auto_alerts off` to disable for the session.
    if get_alert_channel() is not None:
        _auto_alerts_enabled = True
    if not pre_race_alert_daemon.is_running():
        pre_race_alert_daemon.start()
    print(f"Pre-race alert daemon started (45s tick, zero-touch enabled={_auto_alerts_enabled}, "
          f"channel={'set' if get_alert_channel() else 'NOT SET'}).")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands first (like !live)
    await bot.process_commands(message)

    # Check if the bot is mentioned or if it's a DM
    is_mentioned = bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)

    # If it's not a command, and the bot is mentioned or it's a DM,
    # answer deterministically (no LLM round-trip).
    if (is_mentioned or is_dm) and not message.content.startswith('!'):
        await handle_quant_mention(message)

async def handle_quant_mention(message):
    """Deterministic reply for DMs / mentions - zero LLM latency."""
    df = get_data()
    if df is None or len(df) == 0:
        await message.channel.send(
            "⚙️ **Pure quant mode is active (no LLM).**\n"
            "No data loaded yet — run `modeling/generate_predictions.py` and then `!reload`.")
        return
    n_races = df['race_id'].nunique()
    await message.channel.send(
        "⚙️ **Pure quant mode — no LLM latency.**\n"
        f"Loaded {len(df):,} rows · {n_races:,} races through {df['race_date'].max().date()}.\n"
        "Commands: `!live <date> <venue> <race>` · `!scan_overlays <date> <venue>` · `!analyze <race_id|horse>`")


@bot.command(name="live")
async def analyze_live_race(ctx, date_str: str = None, venue: str = "S1", race_num: int = 1, *, mode: str = "all"):
    """Pure-quant live race analysis with the smart money flow engine.
    Usage: !live 2026-03-01 ST 6
    The legacy `mode` argument is accepted but ignored (all analysis is now deterministic).
    """
    t0 = time.perf_counter()
    if date_str is None:
        # Default to today
        date_str = datetime.now().strftime("%Y-%m-%d")

    await ctx.send(f"📡 Polling live odds for {date_str} Venue {venue} Race {race_num}...")

    try:
        live_df = await scrape_live_odds(date_str, venue, race_num)

        if live_df is None or len(live_df) == 0:
            await ctx.send(f"Could not find any live race data for {date_str} {venue} Race {race_num}. There might not be a race, or odds are not posted yet.")
            return

        # scrape_live_odds already persisted this poll; fetch the mature baseline
        baseline, baseline_age, baseline_mature = get_odds_baseline(
            date_str, venue, race_num, min_age_seconds=BASELINE_LEAD_SECONDS)

        df = get_data()
        race_id = f"{date_str}_Race{race_num}"
        meta = {}

        if df is not None and len(df) > 0:
            pred_df = df[df['race_id'] == race_id]
            if len(pred_df) > 0:
                merged = merge_live_odds_with_predictions(live_df, pred_df)
                m0 = pred_df.iloc[0]
                meta = {
                    'track': m0.get('track', venue),
                    'race_class': m0.get('race_class', '?'),
                    'distance': m0.get('distance', '?'),
                    'pace_scenario': m0.get('pace_scenario', '?'),
                }
            else:
                merged = estimate_probabilities_from_history(live_df, df)
        else:
            merged = live_df.copy()

        # --- Smart money flow scoring + Bayesian prior-posterior update ---
        if baseline is not None and len(baseline):
            merged = calculate_smart_money_metrics(merged, baseline)
            merged = apply_smart_money_bayesian_update(merged)
        else:
            merged['smart_money_score'] = SMART_MONEY_CENTER
            merged['flow_signal'] = '➖ STABLE'
            if 'true_prob' in merged.columns:
                merged['live_prob'] = merged['true_prob']
            if 'expected_value' in merged.columns:
                merged['live_expected_value'] = merged['expected_value']

        embed = discord.Embed(
            title=f"🏁 {date_str} · {venue} R{race_num} — Live Quant Card",
            description=(f"Class {meta.get('race_class', '?')} · {meta.get('distance', '?')}m · "
                         f"Pace {meta.get('pace_scenario', '?')} · {meta.get('track', venue)}"),
            color=0x2a9d8f)

        merged = merged.sort_values('live_prob', ascending=False) if 'live_prob' in merged.columns else merged

        for _, r in merged.head(12).iterrows():
            score = float(r.get('smart_money_score', SMART_MONEY_CENTER) or SMART_MONEY_CENTER)
            signal = r.get('flow_signal', '➖ STABLE')
            prob = r.get('live_prob', r.get('true_prob'))
            ev = r.get('live_expected_value', r.get('expected_value'))
            odds = r.get('win_odds')

            advice = '[PASS]'
            try:
                oddsf = float(odds) if odds is not None else None
                evf = float(ev) if ev is not None else None
                probs = float(prob) if prob is not None else None
                if (oddsf is not None and evf is not None and probs is not None
                        and KELLY_MIN_ODDS <= oddsf <= KELLY_MAX_ODDS
                        and evf >= KELLY_MIN_EV and score >= 50.0):
                    decay = float(r.get('decay_factor', 1.0) or 1.0)
                    ev_ratio = probs * oddsf * decay
                    kelly_full = max((ev_ratio - 1.0) / (oddsf - 1.0), 0.0) if oddsf > 1.0 else 0.0
                    kelly_stake = min(kelly_full * 0.15, 0.02)
                    advice = f'[🎯 PRIME VALUE BET (Kelly {kelly_stake * 100:.1f}%)]'
                elif evf is not None and evf > 0 and score < SMART_DRIFTER_THRESHOLD:
                    advice = '[⚠️ VALUE WITH DRIFT RISK]'
            except (TypeError, ValueError):
                advice = '[PASS]'

            value = (f"🛡️ Barrier {fmt(r.get('barrier_draw'), '{:.0f}')} · "
                     f"🎯 P {fmt(prob * 100 if prob is not None else None, '{:.1f}%')} · "
                     f"📊 Odds {fmt(odds)} · "
                     f"💡 Smart {fmt(score, '{:.0f}')} {signal} · "
                     f"💰 EV {fmt(ev, '{:+.2f}')}\n{advice}")
            embed.add_field(name=f"{r.get('horse_name', '?')}", value=value, inline=False)

        compute_ms = (time.perf_counter() - t0) * 1000.0
        footer = f"Pure quant inference · compute {compute_ms:.0f}ms · zero LLM latency"
        if baseline is not None:
            footer += f" · baseline age {baseline_age / 60.0:.0f}m"
        else:
            footer += " · baseline warming up (needs ≥15m of polling)"
        embed.set_footer(text=footer)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"⚠️ Live quant error: {e}")

@bot.command(name="scan_overlays")
async def scan_overlays(ctx, date_str: str, venue: str = "ST", max_races: int = 11):
    """
    Scans all races on a given day to find and rank the best value bets (overlays).
    Usage: !scan_overlays 2026-03-01 ST
    """
    df = get_data()
    if df is None:
        await ctx.send("Data not loaded. Please run the model training script first.")
        return

    def safe_fmt(value, fmt='{:.2f}', fallback='—'):
        if value is None:
            return fallback
        try:
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return fallback
        except TypeError:
            pass
        return fmt.format(value)

    status_msg = await ctx.send(f"🔍 Scanning up to {max_races} races at {venue} on {date_str} for overlays. This might take a minute...\nProgress: 0/{max_races}")

    all_overlays = []
    all_qpl_pairs = []

    for race_num in range(1, max_races + 1):
        try:
            if race_num % 2 == 0 or race_num == max_races:
                await status_msg.edit(content=f"🔍 Scanning up to {max_races} races at {venue} on {date_str} for overlays. This might take a minute...\nProgress: {race_num}/{max_races}")

            live_df = await scrape_live_odds(date_str, venue, race_num)
            if live_df is None or len(live_df) == 0:
                continue

            race_id = f"{date_str}_Race{race_num}"
            pred_df = df[df['race_id'] == race_id]
            if len(pred_df) == 0:
                continue

            merged_df = merge_live_odds_with_predictions(live_df, pred_df)
            if 'expected_value' not in merged_df.columns:
                continue

            # --- Smart money flow scoring vs snapshot baseline ---
            # scrape_live_odds already recorded this poll in the snapshot cache.
            baseline, _, _ = get_odds_baseline(date_str, venue, race_num,
                                                min_age_seconds=BASELINE_LEAD_SECONDS)
            if baseline is not None and len(baseline):
                merged_df = calculate_smart_money_metrics(merged_df, baseline)
                merged_df = apply_smart_money_bayesian_update(merged_df)
                ev_col, prob_col = 'live_expected_value', 'live_prob'
            else:
                merged_df['smart_money_score'] = SMART_MONEY_CENTER
                merged_df['flow_signal'] = '➖ STABLE'
                ev_col, prob_col = 'expected_value', 'true_prob'

            # Attach trainer-intent columns for satellite scoring
            intent = get_intent_features()
            if intent is not None:
                race_intent = intent[intent['race_id'] == race_id]
                if len(race_intent):
                    merged_df = merged_df.merge(
                        race_intent[['horse_name', 'trainer_urgency_index',
                                     'is_forgive_run', 'trial_won_before_race']],
                        on='horse_name', how='left')

            has_odds = merged_df['win_odds'].notna() & merged_df[prob_col].notna()
            priced = merged_df[has_odds].copy()

            # --- Core win picks (value band: KELLY_MIN_ODDS..KELLY_MAX_ODDS, EV > KELLY_MIN_EV) ---
            core = priced[(priced[ev_col] > KELLY_MIN_EV)
                          & priced['win_odds'].between(KELLY_MIN_ODDS, KELLY_MAX_ODDS)]
            for _, row in core.iterrows():
                odds = float(row['win_odds'])
                decay = float(row.get('decay_factor', 1.0) or 1.0)
                ev_ratio = float(row[prob_col]) * odds * decay
                kelly_full = max((ev_ratio - 1.0) / (odds - 1.0), 0.0) if odds > 1 else 0.0
                kelly_stake = min(kelly_full * 0.15, 0.02)
                smart_score = float(row.get('smart_money_score', 50.0) or 50.0)
                all_overlays.append({
                    'race': race_num,
                    'horse': row['horse_name'],
                    'barrier': row.get('barrier_draw'),
                    'prob': float(row[prob_col]),
                    'odds': odds,
                    'ev': float(row[ev_col]),
                    'edge': float(row.get('prob_edge', 0.0) or 0.0),
                    'kelly': kelly_stake,
                    'smart': smart_score,
                    'signal': row.get('flow_signal', '➖ STABLE'),
                })

            # --- QPL exotic pairs (tiered banker x satellite) ---
            picks = build_core_satellite_bets(priced)
            if picks['qpl_pairs']:
                p_view = priced.reset_index(drop=True)
                p_model = p_view['true_prob'].values
                p_model = p_model / p_model.sum()
                p_mkt = market_win_probs(p_view['win_odds'].values)
                idx_of = {name: i for i, name in enumerate(p_view['horse_name'])}
                for b_name, s_name, tier in picks['qpl_pairs']:
                    i, j = idx_of[b_name], idx_of[s_name]
                    qp_model = quinella_place_prob(p_model, i, j)
                    qp_mkt = quinella_place_prob(p_mkt, i, j)
                    edge = (qp_model / qp_mkt - 1.0) if qp_mkt and qp_mkt > 0 else None
                    all_qpl_pairs.append({
                        'race': race_num, 'banker': b_name, 'satellite': s_name,
                        'tier': int(tier), 'qp_model': float(qp_model), 'edge': edge,
                    })
        except Exception as e:
            print(f"Error scanning race {race_num}: {e}")

    if not all_overlays and not all_qpl_pairs:
        await ctx.send("No overlays found or odds are not available yet.")
        return

    await status_msg.edit(content=f"✅ Scan complete! {len(all_overlays)} win picks, {len(all_qpl_pairs)} QPL pairs.")

    embeds = []

    # 🎯 Core Win Picks embed
    top_picks = sorted(all_overlays, key=lambda x: x['ev'], reverse=True)[:10]
    if top_picks:
        e = discord.Embed(
            title=f"🎯 Core Win Picks — {date_str} {venue}",
            description=(f"Odds {KELLY_MIN_ODDS:.1f}–{KELLY_MAX_ODDS:.1f} · EV > {KELLY_MIN_EV:.2f} "
                         f"(decay-adjusted) · smart-money scored"),
            color=0x4361ee)
        for i, o in enumerate(top_picks):
            if o['smart'] >= 50.0:
                marker = ' · 🎯 PRIME VALUE'
            else:
                marker = ' · ⚠️ DRIFT RISK'
            val = (f"🏇 Barrier {safe_fmt(o['barrier'], '{:.0f}', '?')} · "
                   f"🎯 P {safe_fmt(o['prob'] * 100, '{:.1f}%')} · "
                   f"📊 Odds {safe_fmt(o['odds'])} · "
                   f"💡 Smart {safe_fmt(o['smart'], '{:.0f}')} {o['signal']} · "
                   f"💰 EV {safe_fmt(o['ev'], '{:+.2f}')} · "
                   f"📈 Kelly {safe_fmt(o['kelly'] * 100, '{:.1f}%')} bankroll{marker}")
            e.add_field(name=f"{i + 1}. R{o['race']} — {o['horse']}", value=val, inline=False)
        embeds.append(e)

    # 💎 QPL Value Pairs embed
    if all_qpl_pairs:
        e = discord.Embed(
            title="💎 QPL Value Pairs (Banker × Satellite)",
            description="Tier 1/2 Banker paired with high-intent longshot satellites",
            color=0x9b59b6)
        for p in sorted(all_qpl_pairs, key=lambda x: -(x['edge'] if x['edge'] is not None else -9))[:10]:
            val = (f"🔗 {p['banker']} × {p['satellite']} · "
                   f"P(QP) {safe_fmt(p['qp_model'] * 100, '{:.2f}%')} · "
                   f"Edge {safe_fmt(p['edge'] * 100 if p['edge'] is not None else None, '{:+.1f}%')} · "
                   f"Tier {p['tier']}")
            e.add_field(name=f"R{p['race']}", value=val, inline=False)
        embeds.append(e)

    for embed in embeds:
        await ctx.send(embed=embed)

@bot.command(name="analyze")
async def analyze_race(ctx, *, query: str = None):
    """
    Analyzes a race or horse from the trained model data (deterministic).
    Usage: !analyze race 2026-01-25_Race1
    Usage: !analyze horse BEAUTY GEMINI
    """
    if query is None:
        await ctx.send("Please provide a race ID or horse name to analyze. Example: `!analyze BEAUTY GEMINI` or `!analyze 2026-01-25_Race1`")
        return

    df = get_data()
    if df is None:
        await ctx.send("Data not loaded. Please run the model training script first.")
        return

    await ctx.send(f"🏇 Analyzing your request: '{query}'... Please wait.")

    query_upper = query.upper()
    filtered_data = None
    context_type = ""
    
    # 1. Try to find a specific race_id (e.g., 2026-01-25_Race1)
    if "RACE" in query_upper and "_" in query:
        words = query.split()
        for word in words:
            if "_" in word:
                race_id = word
                filtered_data = df[df['race_id'] == race_id]
                context_type = f"Race {race_id}"
                break
    
    # 2. Try to find a specific horse name
    if filtered_data is None or len(filtered_data) == 0:
        # Remove common words to isolate the horse name
        horse_name = query_upper.replace("HORSE", "").replace("ANALYZE", "").strip()
        
        # Exact match first
        filtered_data = df[df['horse_name'].str.upper() == horse_name]
        
        # Partial match if exact fails
        if len(filtered_data) == 0:
            filtered_data = df[df['horse_name'].str.upper().str.contains(horse_name, na=False)]
            
        if len(filtered_data) > 0:
            # Get the actual matched horse name
            matched_horse = filtered_data.iloc[0]['horse_name']
            context_type = f"Horse {matched_horse}"
            # Sort by date descending to get the most recent races
            filtered_data = filtered_data.sort_values(by='race_date', ascending=False).head(10)

    if filtered_data is None or len(filtered_data) == 0:
        await ctx.send(f"Could not find any data matching '{query}'. Try specifying a race_id (e.g., `2026-01-25_Race1`) or a horse name (e.g., `BEAUTY GEMINI`).")
        return

    # Deterministic quantitative summary (no LLM round-trip)
    if 'expected_value' not in filtered_data.columns and {'true_prob', 'win_odds'}.issubset(filtered_data.columns):
        filtered_data = filtered_data.copy()
        filtered_data['expected_value'] = (filtered_data['true_prob'] * filtered_data['win_odds']) - 1

    lines = [f"📐 **Quant Analysis — {context_type}** (deterministic, no LLM)"]

    if context_type.startswith("Race"):
        r = filtered_data.copy()
        if 'true_prob' in r.columns:
            r = r.sort_values('true_prob', ascending=False)
            lines.append("🏆 **Model top 3:**")
            for _, h in r.head(3).iterrows():
                lines.append(
                    f"   • {h['horse_name']} — P {fmt(h['true_prob'] * 100, '{:.1f}%')}"
                    f" · Odds {fmt(h.get('win_odds'))} · EV {fmt(h.get('expected_value'), '{:+.2f}')}")
        if {'win_odds', 'expected_value'}.issubset(r.columns):
            value = r[(r['win_odds'].between(KELLY_MIN_ODDS, KELLY_MAX_ODDS))
                      & (r['expected_value'] > KELLY_MIN_EV)]
            if len(value):
                lines.append(f"🎯 **Value bets ({KELLY_MIN_ODDS:.1f}–{KELLY_MAX_ODDS:.1f}, EV > {KELLY_MIN_EV:.2f}):**")
                for _, h in value.head(5).iterrows():
                    lines.append(f"   • {h['horse_name']} — EV {fmt(h['expected_value'], '{:+.2f}')}"
                                 f" · Odds {fmt(h['win_odds'])}")
            else:
                lines.append("🎯 **Value bets:** none in the overlay band")
    else:
        r = filtered_data.copy()
        starts = len(r)
        if starts:
            pos = pd.to_numeric(r['finish_position'].astype(str).str.extract(r'^(\d+)')[0], errors='coerce')
            wins = int((pos == 1).sum())
            lines.append(f"🏇 Starts: {starts} · Wins: {wins} · Win%: {wins / starts * 100:.1f}%")
            if 'speed_figure' in r.columns:
                lines.append(f"⚡ Best speed figure: {fmt(r['speed_figure'].max())}")
            if 'true_prob' in r.columns:
                lines.append("📈 Recent model probs (last 5): "
                             + ", ".join(fmt(p * 100, '{:.0f}%') for p in r['true_prob'].tail(5)))
        else:
            lines.append("No history available.")

    await ctx.send('\n'.join(lines)[:1990])

@bot.command(name="bot_help", aliases=["help", "commands"])
async def show_help(ctx):
    """Shows all available commands and how to use them."""
    help_text = """
🏇 **HKJC Quant Analyst Bot - Available Commands** 🏇

**📡 LIVE QUANT & SMART MONEY**
`!live <YYYY-MM-DD> <Venue> <Race_Num>`
*Polls live tote odds, scores smart-money flow (Steamer/Stable/Drifter), applies the Bayesian live-probability update and prints a full quant card with execution advice.*
*Example:* `!live 2026-03-01 ST 6`

**🔍 OVERLAY SCANNER**
`!scan_overlays <YYYY-MM-DD> <Venue> [Max_Races]`
*Scans the day's card, scores every race with the smart-money engine and ranks Prime Value bets.*
*Example:* `!scan_overlays 2026-03-01 ST 11`

**📊 HISTORICAL ANALYSIS (deterministic)**
`!analyze <Race_ID>` — model top-3 + value bets for a past race
`!analyze <Horse_Name>` — career record + recent model probs
*Example:* `!analyze 2026-01-25_Race1`

**⚙️ SYSTEM**
`!reload` — reloads the predictions data file
`!auto_alerts on|off` — toggle the pre-race alert daemon (admin)

⚙️ This bot is 100% deterministic — no LLM/API latency, sub-second responses.
"""
    await ctx.send(help_text)

@bot.command(name="reload")
async def reload_data_command(ctx):
    """Reloads the predictions CSV file."""
    load_data()
    df = get_data()
    if df is not None:
        await ctx.send(f"Successfully reloaded data. {len(df)} rows available.")
    else:
        await ctx.send("Failed to load data. Check if data/all_predictions.csv exists.")

if __name__ == "__main__":
    print("Starting Discord Bot...")
    bot.run(DISCORD_TOKEN)

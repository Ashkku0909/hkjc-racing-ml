"""
HKJC Quant Terminal (v4) — Bloomberg-style low-latency surveillance UI
========================================================================
Master-spec implementation:
  Law 1  Strict single-race execution: ONLY the active race tab is scraped /
         scored per tick. The "Overview" tab is a cold 60 s summary and never
         competes with the active loop (no card scan, no scoring of other
         races while a race tab is selected).
  Law 2  Real data only, zero synthetic odds. A card whose ENTIRE field shows
         <= 1.01 is unformed -> every row is tagged ⏳ pre-open and EV is
         zeroed. A 1.0 favourite inside an OPEN race is a real heavy favourite
         (verified 2026-09-06 ST R3: ~97% of a live $2.5M win pool, red
         "Favourite" tag on bet.hkjc.com) and keeps its raw decay-free EV - the
         open/closed decision is RACE-LEVEL, never per-row.
  Cache  Model / calibrator / features are loaded ONCE per process
         (web_live.ensure_data_loaded -> module singletons). No reload per tick.

    streamlit run app.py
"""
import sys
import os
import time
import re
import html as htmlmod
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="HKJC Quant Terminal", page_icon="🏇", layout="wide",
                   initial_sidebar_state="collapsed")

from modeling.model_training import EDGE_DECAY_C, EDGE_DECAY_GAMMA  # noqa: E402

import web_live
from web_live import (
    ensure_data_loaded, poll_race, card_size, score_race, flag_row,
    de_vig_market_probs, race_post_time, snap_store, SLOW_TTL,
    effective_poll_ttl, race_meta, MAX_RACES, DISCOVER_TTL,
)
from scraping.live_scraper import _extras_cache

HKT = timezone(timedelta(hours=8))
PRIME_COLOR = "#FFD700"
STEAM_COLOR = "#00E676"
DRIFT_COLOR = "#FF5252"
STABLE_COLOR = "#9E9E9E"
MUTED_COLOR = "#6b7a99"

st.markdown("""
<style>
.stApp { background:#0b0f17; }
[data-testid="stSidebar"] { background:#0e1420; }
.qt-term { font-family:'Cascadia Mono','Consolas',monospace; color:#e6edf3; }
.qt-head {
  display:flex; align-items:center; gap:22px; flex-wrap:wrap;
  background:linear-gradient(90deg,#101a2e,#0d1420); border:1px solid #1e2635;
  border-radius:10px; padding:10px 18px; margin:6px 0 10px 0;
}
.qt-head .qt-title { font-size:17px; font-weight:700; color:#fff; letter-spacing:.5px; }
.qt-head .qt-cell { font-size:12px; color:#aab4c8; }
.qt-head .qt-cell b { color:#fff; font-size:14px; }
.qt-kpis { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:8px 0 12px 0; }
.qt-kpi {
  background:#121826; border:1px solid #1e2635; border-radius:8px; padding:8px 14px;
}
.qt-kpi .k { font-size:10px; color:#6b7a99; text-transform:uppercase; letter-spacing:1px; }
.qt-kpi .v { font-size:20px; font-weight:700; color:#fff; font-variant-numeric:tabular-nums; }
.qt-row {
  display:grid; grid-template-columns: 210px 196px 116px 116px 122px 142px 84px 150px;
  gap:6px; align-items:center; padding:6px 12px; margin:3px 0;
  background:#101724; border:1px solid #1e2635; border-left:3px solid transparent;
  border-radius:6px; font-size:12px;
}
.qt-row.prime { border:1px solid #FFD700; box-shadow:0 0 0 1px rgba(255,215,0,.25); }
.qt-row.preopen { opacity:.55; }
.qt-hdr {
  display:grid; grid-template-columns: 210px 196px 116px 116px 122px 142px 84px 150px;
  gap:6px; padding:2px 12px 4px 12px; color:#6b7a99; font-size:10px;
  text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid #1e2635;
}
.qt-top4 {
  background:linear-gradient(90deg,#2a2205,#141a2c); border:1px solid rgba(255,215,0,.6);
  border-radius:8px; padding:9px 16px; margin:6px 0 12px 0; color:#FFD700;
  font-size:13.5px; font-weight:700; letter-spacing:.3px;
}
.qt-top4 .r { color:#8ea2c0; font-weight:400; }
.qt-horse { color:#e6edf3; font-weight:600; }
.qt-horse .no { color:#8ea2c0; font-weight:400; font-size:11px; margin-left:2px; }
.qt-bars { font-size:10px; color:#8ea2c0; }
.qt-bars .lbl { display:inline-block; width:38px; color:#6b7a99; }
.qt-bar { display:inline-block; width:74px; height:6px; background:#1b2436; border-radius:3px; vertical-align:middle; margin:0 4px; }
.qt-bar i { display:block; height:6px; border-radius:3px; }
.qt-bar.model i { background:#4dabf7; }
.qt-bar.mkt i { background:#9775fa; }
.qt-bar.place i { background:#ffd166; }
.qt-pool { display:flex; flex-direction:column; gap:3px; }
.qt-chip { display:inline-flex; align-items:center; gap:4px; padding:2px 9px;
  border-radius:4px; font-weight:700; font-size:11.5px; font-variant-numeric:tabular-nums; }
.qt-chip.w { background:#2a2106; color:#FFB300; border:1px solid rgba(255,179,0,.45); }
.qt-chip.p { background:#062028; color:#00E5FF; border:1px solid rgba(0,229,255,.35); }
.qt-chip .dn { color:#00E676; }
.qt-chip .up { color:#FF5252; }
.qt-ev-chip { font-size:11px; font-weight:700; color:#e6edf3; padding-left:2px;
  font-variant-numeric:tabular-nums; }
.qt-ev-chip.prime { color:#FFD700; }
.qt-meta { font-size:11px; color:#8ea2c0; }
.qt-meta .jt { color:#aab4c8; }
.qt-meta .f3 { color:#4dabf7; font-weight:700; }
.qt-lq { color:#FFD700; font-weight:700; font-size:10px; }
.qt-smart { font-variant-numeric:tabular-nums; }
.qt-smart .flow { margin-left:4px; }
.qt-verdict { font-weight:700; font-size:11px; white-space:nowrap; }
.qt-verdict.prime { color:#FFD700; }
.qt-verdict.pass { color:#9E9E9E; }
.qt-verdict.preopen { color:#6b7a99; }
.qt-verdict.closed { color:#FFB300; }
.qt-note { color:#6b7a99; font-size:11px; margin-top:10px; }
.qt-up { color:#00E676; }
.qt-down { color:#FF5252; }
.qt-r1 { color:#FFD700; } .qt-r2 { color:#C0C0C0; } .qt-r3 { color:#CD7F32; } .qt-r4 { color:#7C9CD6; }
.qt-ov-row {
  display:grid; grid-template-columns: 62px 62px 78px 92px 68px 1fr 168px;
  gap:8px; align-items:center; padding:6px 12px; margin:3px 0;
  background:#101724; border:1px solid #1e2635; border-radius:6px; font-size:11.5px;
}
.qt-ov-hdr { display:grid; grid-template-columns:62px 62px 78px 92px 68px 1fr 168px;
  gap:8px; padding:2px 12px 4px 12px; color:#6b7a99; font-size:10px;
  text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid #1e2635; }
.qt-ov-top4 { font-size:10.5px; }
.qt-ov-row > span, .qt-ov-hdr > span { white-space:nowrap; }
.qt-pace { display:inline-block; padding:3px 10px; border-radius:4px; font-size:11px;
  font-weight:700; margin:2px 6px 10px 0; }
.qt-pace.meltdown { background:#3a1010; color:#FF5252; border:1px solid rgba(255,82,82,.5); }
.qt-pace.lone { background:#0f2a1a; color:#00E676; border:1px solid rgba(0,230,118,.4); }
.qt-pace.normal { background:#141c2e; color:#9E9E9E; }
.qt-pace.slow { background:#2a1f0a; color:#FFB300; border:1px solid rgba(255,179,0,.5); }
.qt-gate {
  display:inline-block; padding:3px 12px; border-radius:4px; font-size:11px;
  font-weight:700; margin:2px 6px 10px 0; background:#33260a; color:#FFB300;
  border:1px solid rgba(255,179,0,.65); animation:qtgate 1s ease-in-out infinite;
}
.qt-turbo {
  display:inline-block; padding:3px 12px; border-radius:4px; font-size:11px;
  font-weight:700; margin:2px 6px 10px 0; background:#1a2433; color:#ffd166;
  border:1px solid rgba(255,209,102,.5);
}
@keyframes qtgate { 0%,100% { opacity:1; } 50% { opacity:.4; } }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def hkt_now():
    return datetime.now(HKT)


def fmt_tminus(delta):
    """Day-aware countdown: >1h -> 'T-16h 09m' (never looks like a clock time),
    last 60 minutes -> precise 'T-15:32' seconds countdown for T-15m/T-3m ops."""
    if delta is None:
        return "—"
    secs = int(delta.total_seconds())
    if secs <= 0:
        return "🏁 GO"
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"T-{d}d {h:02d}h {m:02d}m"
    if h >= 1:
        return f"T-{h}h {m:02d}m"
    return f"T-{m:02d}:{s:02d}"


def esc(x):
    return htmlmod.escape(str(x if x is not None else ""))


def num(v, fallback="—"):
    try:
        if v is None or pd.isna(v):
            return fallback
        return float(v)
    except (TypeError, ValueError):
        return fallback


def numf(v):
    """Numeric-only: returns float or None (NEVER a display string), safe for
    arithmetic/formatting. Use this everywhere math is applied."""
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def horse_label(r, html=True):
    """Saddlecloth law: horses are ALWAYS shown as 'NAME (#N)' or 'NAME (#2 [D12])'."""
    name = str(r.get('horse_name') or '—')
    no = r.get('horse_number')
    d = r.get('barrier_draw')
    nm = esc(name) if html else name
    try:
        no_s = f"#{int(no)}" if no is not None and not pd.isna(no) else "#?"
    except (TypeError, ValueError):
        no_s = "#?"
    d_s = ''
    if d is not None and not pd.isna(d):
        try:
            d_s = f" [D{int(d)}]"
        except (TypeError, ValueError):
            d_s = ''
    return f"{nm} ({no_s}{d_s})"


def chip_arrow(td):
    """Odds shortened (money in) -> green ↓; drifted -> red ↑; unchanged -> ''."""
    if td is None:
        return ''
    if td < -0.001:
        return '<span class="dn"> ↓</span>'
    if td > 0.001:
        return '<span class="up"> ↑</span>'
    return ''


def flow_badge(signal, smart):
    s = num(smart, 50.0)
    if signal in ("🔥 STEAMER",) or (s is not None and s >= 75):
        return f'<span class="flow" style="color:{STEAM_COLOR};">🔥 STEAMER</span>'
    if signal in ("⚠️ DRIFTER",) or (s is not None and s < 40):
        return f'<span class="flow" style="color:{DRIFT_COLOR};">⚠️ DRIFTER</span>'
    return f'<span class="flow" style="color:{STABLE_COLOR};">➖ STABLE</span>'


def win_prime(r) -> bool:
    """Prime Win (dual-mode guardrails):
      P_final >= 0.16 AND EV_win >= 1.15 AND O_W in [2.2, 14.0] AND S >= 50.
    Longshots (O_W > 14.0) with EV >= 1.25 are NEVER surfaced as straight WIN.
    """
    odds = numf(r.get('win_odds'))
    prob = numf(r.get('prob'))
    ev = numf(r.get('ev'))
    smart = numf(r.get('smart_money_score')) or 50.0
    if odds is None or prob is None or ev is None:
        return False
    return (prob >= 0.16 and (ev + 1.0) >= 1.15
            and 2.2 <= odds <= 14.0 and smart >= 50.0)


def place_prime(r) -> bool:
    """Prime Place (conservative Henery calibration):
      O_P in [1.8, 4.0] -> EV_place >= 1.15
      O_P > 4.0 (longshot) -> EV_place >= 1.30
    plus S >= 50. numf: NEVER compare a display string with a float."""
    po = numf(r.get('place_odds'))
    pev = numf(r.get('place_ev'))
    smart = numf(r.get('smart_money_score')) or 50.0
    if po is None or pev is None:
        return False
    if po > 4.0:
        return pev >= 1.30 and smart >= 50
    return 1.8 <= po <= 4.0 and pev >= 1.15 and smart >= 50


def verdict_of(r):
    if bool(r.get('race_closed')):
        return 'closed', '🏁 CLOSED'
    if bool(r.get('unposted')):
        return 'preopen', '⏳ PRE-OPEN'
    pw = win_prime(r)
    pp = place_prime(r)
    evr = numf(r.get('ev'))
    odds = numf(r.get('win_odds'))
    # Longshot value trap -> exotics-only channel (PLACE / QP / TIERCE anchor)
    if (not pw and pp and evr is not None and odds is not None
            and (evr + 1.0) >= 1.25 and odds > 14.0):
        return 'prime', '🎯 EXOTIC (P/QP/TIERCE)'
    if bool(r.get('divergence_trap')) and pp:
        return 'prime', '🎯 EXOTIC (P/QP/TIERCE)'
    if pw and pp:
        return 'prime', '🎯 DUAL VALUE'
    if pw:
        return 'prime', '🎯 PRIME W'
    if pp:
        return 'prime', '🎯 PRIME P'
    return 'pass', 'PASS'


def tick_arrow(delta):
    """Per-tick move marker: odds SHORTENED (money in) -> green down-arrow,
    odds DRIFTED (money out) -> red up-arrow."""
    if delta is None:
        return ''
    if delta < -0.001:
        return '<span class="qt-up">↓</span>'
    if delta > 0.001:
        return '<span class="qt-down">↑</span>'
    return ''


def tick_cls(delta):
    if delta is None:
        return ''
    if delta < -0.001:
        return ' down'
    if delta > 0.001:
        return ' up'
    return ''


def header_panel(date_str, venue, race_no, post_dt, status):
    now = hkt_now()
    meeting = f"{date_str.replace('-', '/')} · {venue}"
    clock = now.strftime("%H:%M:%S")
    if race_no:
        countdown = fmt_tminus(post_dt - now if post_dt else None)
        race_txt = f"R{race_no}"
    else:
        countdown = "—"
        race_txt = "OVERVIEW"
    st.markdown(
        f'<div class="qt-head qt-term">'
        f'<span class="qt-title">🏇 HKJC QUANT TERMINAL</span>'
        f'<span class="qt-cell">Meeting <b>{esc(meeting)}</b></span>'
        f'<span class="qt-cell">HKT <b>{clock}</b></span>'
        f'<span class="qt-cell">Focus <b>{race_txt}</b></span>'
        f'<span class="qt-cell">Countdown <b>{countdown}</b></span>'
        f'<span class="qt-cell">{status}</span>'
        f'</div>', unsafe_allow_html=True)


def kpi_strip(vals):
    cells = "".join(
        f'<div class="qt-kpi"><div class="k">{esc(k)}</div>'
        f'<div class="v">{esc(v)}</div></div>' for k, v in vals)
    st.markdown(f'<div class="qt-kpis qt-term">{cells}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Focus view (active race): 1-3 s loop, only this race is touched
# ----------------------------------------------------------------------
def render_focus(scored, race_no):
    scored = scored.copy()
    scored['_flag'] = scored.apply(flag_row, axis=1)
    scored['mkt_prob'] = de_vig_market_probs(scored['win_odds'])
    ev = pd.to_numeric(scored['ev'], errors='coerce')
    scored['_ord'] = ev.fillna(-1e9)
    scored = scored.sort_values('_ord', ascending=False).reset_index(drop=True)

    valid = scored[~scored['unposted']].copy()
    o = pd.to_numeric(valid['win_odds'], errors='coerce')
    o = o[o.notna() & (o >= 1.0)]   # keep a REAL 1.0 heavy favourite in the book
    overrun = float((1.0 / o).sum() - 1.0) * 100.0 if len(o) else float('nan')
    sm = pd.to_numeric(valid['smart_money_score'], errors='coerce')
    steam = int((sm >= 75).sum())
    drift = int((sm < 40).sum())
    top_ev = float(ev.max() + 1.0) if ev.notna().any() else float('nan')
    n_active = int((~scored['unposted']).sum())
    n_pre = int(scored['unposted'].sum())

    kpi_strip([
        ("Active Runners", f"{n_active}" + (f" · {n_pre} ⏳" if n_pre else "")),
        ("Market Overround", f"{overrun:+.1f}%" if not pd.isna(overrun) else "—"),
        ("Top EV Ratio", f"×{top_ev:.3f}" if not pd.isna(top_ev) else "—"),
        ("Steamers", f"{steam}"),
        ("Drifters", f"{drift}"),
    ])

    # --- Pace scenario matrix badge (lagged front-runner density) ---
    race_closed = bool(scored.iloc[0].get('race_closed')) if len(scored) else False
    exec_state = str(scored.iloc[0].get('exec_state') or '') if len(scored) else ''
    if exec_state == 'LOADING_DELAY':
        st.markdown('<span class="qt-gate qt-term">🟡 GATE LOADING — TURBO STREAM '
                    '(0.8s · W=0.85 pinned until freeze)</span>', unsafe_allow_html=True)
    elif exec_state == 'TURBO_APPROACH':
        st.markdown('<span class="qt-turbo qt-term">🚦 TURBO APPROACH — 0.8s STREAM '
                    '· W=0.85</span>', unsafe_allow_html=True)
    if race_closed:
        st.markdown('<span class="qt-pace lone qt-term">🏁 RACE CLOSED — odds frozen at '
                    'official close; no live flow / EV signals.</span>', unsafe_allow_html=True)
    else:
        pace_scn = str(scored.iloc[0].get('pace_scenario') or 'NORMAL')
        pace_n = num(scored.iloc[0].get('pace_n_leaders'), 0.0)
        if pace_scn == 'MELTDOWN':
            st.markdown(f'<span class="qt-pace meltdown qt-term">🔥 PACE MELTDOWN '
                        f'({pace_n:.0f} leaders) — front-runners penalised, closers boosted</span>',
                        unsafe_allow_html=True)
        elif pace_scn == 'LONE':
            st.markdown('<span class="qt-pace lone qt-term">🚀 LONE LEADER (slow bias) '
                        '— leader logit +0.20</span>', unsafe_allow_html=True)
        elif pace_scn == 'SLOW':
            st.markdown('<span class="qt-pace slow qt-term">🐢 SLOW PACE (0 leaders) '
                        '— HV wide-draw closers −0.15 logit · inside front-runners +0.10</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="qt-pace normal qt-term">⚖️ NORMAL PACE '
                        f'({pace_n:.0f} leaders)</span>', unsafe_allow_html=True)

    # --- Top-4 exotic order banner (model-projected 1st/2nd/3rd/4th) ---
    t4 = scored.dropna(subset=['prob'])
    if len(t4) >= 4:
        t4 = t4.nlargest(4, 'prob').reset_index(drop=True)
        rk = ['1st', '2nd', '3rd', '4th']
        parts = [
            f'<span class="r">{rk[i]}</span> {horse_label(t4.iloc[i])} '
            f'({float(t4.iloc[i]["prob"]) * 100:.1f}%)' for i in range(len(t4))]
        st.markdown(
            f'<div class="qt-top4 qt-term">🏆 MODEL ORDER&nbsp;&nbsp;'
            f'{"&nbsp;·&nbsp;".join(parts)}</div>', unsafe_allow_html=True)

    cols = ["HORSE", "MODEL / DE-VIG MKT", "WIN", "PLACE",
            "SMART FLOW", "JOCKEY / TRAINER", "LAST 3", "VERDICT"]
    hdr = "".join(f'<span>{c}</span>' for c in cols)
    st.markdown(f'<div class="qt-hdr qt-term">{hdr}</div>', unsafe_allow_html=True)

    thin = (n_active < 8) or pd.isna(overrun) or (overrun < 15.0)
    if thin:
        st.markdown(
            f'<div class="qt-note qt-term" style="color:#FFD700;">⚠️ THIN LIQUIDITY '
            f'(&lt;8 runners or overround &lt;15%) — EV shown for reference only, '
            f'no staking signal.</div>', unsafe_allow_html=True)

    for _, r in scored.iterrows():
        kind = verdict_of(r)
        row_cls = "qt-row preopen" if kind[0] == 'preopen' else ("qt-row prime" if kind[0] == 'prime' else "qt-row")
        odds = num(r.get('win_odds'))
        place = num(r.get('place_odds'))
        prob = numf(r.get('prob'))
        mkt = numf(r.get('mkt_prob'))
        evr = numf(r.get('ev'))
        evx = evr + 1.0 if evr is not None else None
        pev = numf(r.get('place_ev'))
        p3 = numf(r.get('p_top3'))
        smart = numf(r.get('smart_money_score'))
        if smart is None:
            smart = 50.0
        mp = f"{prob * 100:.1f}%" if prob is not None else "—"
        kp = f"{mkt * 100:.1f}%" if mkt is not None else "—"
        t3pct = f"{p3 * 100:.1f}%" if p3 is not None else "—"
        mbar = min(100.0, (prob or np.nan) * 100) if prob is not None else 0.0
        kbar = min(100.0, (mkt or np.nan) * 100) if mkt is not None else 0.0
        evtxt = f"×{evx:.3f}" if evx is not None else "—"
        pevtxt = f"×{pev:.3f}" if pev is not None else "—"
        w_td = numf(r.get('tick_delta'))
        p_td = numf(r.get('tick_delta_place'))
        w_flag = win_prime(r)
        pp_flag = place_prime(r)
        vt_cls = "prime" if kind[0] == 'prime' else ("closed" if kind[0] == 'closed'
                  else ("preopen" if kind[0] == 'preopen' else "pass"))
        vtxt = kind[1]
        flow_html = ('<span style="color:#6b7a99;">—</span>' if kind[0] == 'closed'
                     else flow_badge(r.get("flow_signal"), smart))
        if kind[0] == 'prime' and ('PRIME W' in kind[1] or 'DUAL' in kind[1]):
            kr = numf(r.get('kelly'))
            if kr is not None:
                vtxt += f' ({kr * 100:.1f}% BR)'
        jt = str(r.get('jockey') or '—').strip()
        tr = str(r.get('trainer') or '—').strip()
        f3 = str(r.get('last3_form') or '—').strip()
        st.markdown(
            f'<div class="{row_cls} qt-term">'
            f'<div class="qt-horse">{horse_label(r)}</div>'
            f'<div class="qt-bars">'
            f'<span class="lbl">MODEL</span><span class="qt-bar model"><i style="width:{mbar:.0f}%"></i></span>{mp}'
            f'<br><span class="lbl">MKT</span><span class="qt-bar mkt"><i style="width:{kbar:.0f}%"></i></span>{kp}'
            f'<br><span class="lbl">TOP3</span><span class="qt-bar place"><i style="width:{min(100.0, (p3 or 0.0) * 100):.0f}%"></i></span>{t3pct}'
            f'</div>'
            f'<div class="qt-pool">'
            f'<span class="qt-chip w">W {odds if odds is not None else "—"}{chip_arrow(w_td)}</span>'
            f'<span class="qt-ev-chip {"prime" if w_flag else ""}">{evtxt}</span>'
            f'</div>'
            f'<div class="qt-pool">'
            f'<span class="qt-chip p">P {place if place is not None else "—"}{chip_arrow(p_td)}</span>'
            f'<span class="qt-ev-chip {"prime" if pp_flag else ""}">{pevtxt}</span>'
            f'</div>'
            f'<div class="qt-smart">S {smart:.0f} {flow_html}</div>'
            f'<div class="qt-meta"><span class="jt">{esc(jt)}</span> / <span class="jt">{esc(tr)}</span></div>'
            f'<div class="qt-meta"><span class="f3">{esc(f3)}</span></div>'
            f'<div class="qt-verdict {vt_cls}">{vtxt}</div>'
            f'</div>', unsafe_allow_html=True)

    track = scored[scored['_ord'] > -1e9]
    if len(track):
        top3m = track.nlargest(3, 'prob').index.tolist()
        top3k = track.nsmallest(3, 'win_odds').index.tolist()
        lm = [horse_label(track.loc[i]) for i in top3m]
        lk = [horse_label(track.loc[i]) for i in top3k]
        div = [n for n in top3m if n not in top3k]
        ld = [horse_label(track.loc[i]) for i in div]
        st.markdown(
            f'<div class="qt-note qt-term">🔬 Model vs market divergence: '
            f'model top-3 = {", ".join(lm)} · market top-3 = {", ".join(lk)}'
            + (f' · divergent: {", ".join(ld)}' if div else '') + '</div>',
            unsafe_allow_html=True)

    # --- LLM export (full race intel as one Markdown file) ---
    date_str = str(scored.iloc[0].get('race_date')) if len(scored) else ''
    venue = str(scored.iloc[0].get('venue')) if len(scored) else ''
    if date_str and venue:
        report = build_llm_report(scored, date_str, venue, race_no)
        st.download_button(
            "📄 Download LLM Report (.md)",
            data=report,
            file_name=f"hkjc_{date_str}_{venue}_Race{race_no}.md",
            mime="text/markdown",
        )


def build_llm_report(scored, date_str, venue, race_no) -> str:
    """One Markdown file with EVERYTHING the board knows about the race
    (SpeedPRO energy, form remarks, draw stats, jockey/trainer, model/market
    probabilities, smart flow) - ready to upload to an external LLM."""
    now = hkt_now()
    meta = race_meta(date_str, venue, int(race_no))
    title = re.sub(r'\s+', ' ', str((meta or {}).get('title') or '')).strip()
    post = race_post_time(date_str, venue, int(race_no))
    countdown = fmt_tminus((post - now) if post is not None else None)
    n = len(scored)
    exec_state = str(scored.iloc[0].get('exec_state') or '') if n else ''
    race_closed = bool(scored.iloc[0].get('race_closed')) if n else False
    pace = str(scored.iloc[0].get('pace_scenario') or 'NORMAL') if n else 'NORMAL'
    n_lead = numf(scored.iloc[0].get('pace_n_leaders')) if n else None
    valid = scored[~scored['unposted']] if 'unposted' in scored else scored
    overrun = None
    if len(valid):
        o = pd.to_numeric(valid['win_odds'], errors='coerce')
        o = o[o.notna() & (o >= 1.0)]
        if len(o):
            overrun = float((1.0 / o).sum() - 1.0) * 100.0
    thin = (race_closed or len(valid) < 8 or overrun is None or overrun < 15.0)
    extras = _extras_cache.get((date_str, venue, int(race_no))) or {}
    wpq = str(extras.get('wpq_str') or '').strip()
    n_img = len(extras.get('speedpro_images') or [])
    vn = {'ST': 'Sha Tin', 'HV': 'Happy Valley'}.get(str(venue).upper(), str(venue))

    def pct(x) -> str:
        v = numf(x)
        return f"{v * 100:.1f}%" if v is not None else '—'

    L: list = []
    L.append(f"# HKJC Race Report — {date_str} {vn} R{race_no}")
    L.append("")
    L.append(f"- **Race title**: {title or '—'}")
    L.append(f"- **Venue**: {vn} | **Race no**: {race_no}")
    L.append(f"- **Post time (HKT)**: {post.strftime('%H:%M') if post else '—'} | "
             f"**Generated**: {now.strftime('%Y-%m-%d %H:%M:%S')} HKT | **Countdown**: {countdown}")
    L.append(f"- **Execution state**: {exec_state} | **Race closed**: {race_closed}")
    L.append(f"- **Valid runners**: {len(valid)}/{n} | **Overround**: "
             f"{overrun:+.1f}%" if overrun is not None else "- **Valid runners**: "
             f"{len(valid)}/{n} | **Overround**: —")
    L[-1] += f" | **Pace**: {pace}" + (f" ({n_lead:.0f} leaders)" if n_lead is not None else "")
    L.append(f"- **Thin liquidity**: {thin}")
    L.append("")

    order = scored.copy()
    order['_prob'] = pd.to_numeric(order.get('prob'), errors='coerce').fillna(0.0)
    order = order.sort_values('_prob', ascending=False)
    for i, (_, r) in enumerate(order.iterrows(), 1):
        name = str(r.get('horse_name') or '—')
        no = r.get('horse_number')
        d = r.get('barrier_draw')
        no_s = f"#{int(no)}" if pd.notna(no) else "#?"
        d_s = f"[D{int(d)}]" if pd.notna(d) else "[D?]"
        unposted = bool(r.get('unposted'))
        jt = str(r.get('jockey') or '—')
        tr = str(r.get('trainer') or '—')
        wt = numf(r.get('weight_carried'))
        w = numf(r.get('win_odds'))
        p = numf(r.get('place_odds'))
        w_td = numf(r.get('tick_delta'))
        p_td = numf(r.get('tick_delta_place'))
        sp = numf(r.get('speedpro_energy'))
        frm = str(r.get('formguide_remarks') or '').strip()
        dw = numf(r.get('draw_win_pct'))
        dp = numf(r.get('draw_place_pct'))
        smart = numf(r.get('smart_money_score'))
        ev = numf(r.get('ev'))
        pev = numf(r.get('place_ev'))
        kelly = numf(r.get('kelly'))
        f3 = str(r.get('last3_form') or '—')
        flow = str(r.get('flow_signal') or 'STABLE')
        flags = []
        if bool(r.get('syndicate_steam')):
            flags.append('SYNDICATE_STEAM')
        if bool(r.get('divergence_trap')):
            flags.append('DIVERGENCE_TRAP')
        if bool(r.get('smart_place_absorption')):
            flags.append('SMART_PLACE_ABSORPTION')
        L.append(f"## {i}. {name} ({no_s} {d_s})" + (" — PRE-OPEN" if unposted else ""))
        L.append(f"- **Jockey**: {jt} | **Trainer**: {tr} | "
                 f"**Weight**: {f'{wt:.0f}' if wt is not None else '—'} | **Draw**: {d_s}")
        L.append(f"- **Win odds**: {w if w is not None else '—'}"
                 + (f" (Δ {w_td:+.1f})" if w_td is not None else "")
                 + f" | **Place odds**: {p if p is not None else '—'}"
                 + (f" (Δ {p_td:+.2f})" if p_td is not None else ""))
        L.append(f"- **SpeedPRO Energy**: {f'{sp:.0f}' if sp is not None else '—'} | "
                 f"**Draw stats**: win {f'{dw:.1f}%' if dw is not None else '—'} / "
                 f"place {f'{dp:.1f}%' if dp is not None else '—'}")
        if frm:
            L.append(f"- **Form remarks**: {frm}")
        L.append(f"- **Model prob**: {pct(r.get('prob'))} | "
                 f"**Market (de-vig)**: {pct(r.get('mkt_prob'))} | "
                 f"**Top3**: {pct(r.get('p_top3'))} | "
                 f"**Place EV**: ×{(pev if pev is not None else float('nan')):.3f}" if pev is not None
                 else f"- **Model prob**: {pct(r.get('prob'))} | "
                 f"**Market (de-vig)**: {pct(r.get('mkt_prob'))} | "
                 f"**Top3**: {pct(r.get('p_top3'))} | **Place EV**: —")
        L.append(f"- **EV ratio**: ×{(ev + 1.0) if ev is not None else float('nan'):.3f}" if ev is not None
                 else "- **EV ratio**: —")
        L.append(f"- **Kelly**: {f'{kelly * 100:.2f}%' if kelly is not None else '—'} | "
                 f"**Smart money**: S {f'{smart:.0f}' if smart is not None else '—'} {flow}")
        L.append(f"- **Verdict**: {verdict_of(r)[1]}"
                 + (f" | **Flags**: {', '.join(flags)}" if flags else ""))
        L.append(f"- **Last 3 runs**: {f3}")
        L.append("")

    if wpq:
        L.append("## Win / Place Quinella (WPQ)")
        L.append("")
        L.append(wpq)
        L.append("")
    if n_img:
        L.append(f"- SpeedPRO chart images captured on the board: {n_img} "
                 f"(base64, not embedded in this file)")
    L.append("---")
    L.append("Generated by HKJC Quant Terminal — educational use only.")
    return "\n".join(L)


# ----------------------------------------------------------------------
# Today's buy-list card (cold: persisted snapshots + model only, no scrape)
# ----------------------------------------------------------------------
def _min_win_odds(p, tgt: float = 1.15):
    """Decay-aware minimum W odds so EV_ratio = p*O*min((C/O)^gamma,1) >= tgt."""
    p = numf(p)
    if p is None or not (0.0 < p < 1.0):
        return None
    if p * EDGE_DECAY_C >= tgt:                      # O <= C: decay clamped to 1
        return tgt / p
    # O > C: ratio = p * C^gamma * O^(1-gamma)
    return float((tgt / (p * EDGE_DECAY_C ** EDGE_DECAY_GAMMA)) ** (1.0 / (1.0 - EDGE_DECAY_GAMMA)))


def daily_card_rows(date_str: str, venue: str) -> list:
    """All actionable plays for the day (prime verdicts, non-thin, open races).

    SELF-SEEDING: races without persisted snapshots are polled ONCE with the
    light scan (no SpeedPRO extras) - real pools only. Cache TTL = DISCOVER_TTL
    (5 min), refreshed on each card render.

    Each entry: tier/conf, race, play (W / P / WP / EX), horse, live odds,
    EV ratios, decay-aware buy thresholds, smart score and reasons."""
    store = snap_store(date_str, venue)
    nos = set()
    for rid in store['race_id'].dropna().unique():
        m = re.match(rf"{re.escape(date_str)}_Race(\d+)$", str(rid))
        if m:
            nos.add(int(m.group(1)))
    # Seed races with no snapshot yet (real pools, light scan)
    empty_run = 0
    for rn in range(1, MAX_RACES + 1):
        if rn in nos:
            continue
        live = poll_race(date_str, venue, rn, ttl=DISCOVER_TTL, skip_extras=True)
        if live is not None and len(live) > 0:
            nos.add(rn)
            empty_run = 0
        else:
            empty_run += 1
            if empty_run >= 2:
                break
    picks = []
    for rn in sorted(nos):
        live = poll_race(date_str, venue, rn, ttl=DISCOVER_TTL, skip_extras=True)
        sc = score_race(live, date_str, venue, rn) if live is not None else None
        if sc is None:
            # fall back to the latest persisted frame for this race
            sub = store[store['race_id'] == f"{date_str}_Race{rn}"]
            e = pd.to_numeric(sub['epoch'], errors='coerce').max()
            if pd.isna(e):
                continue
            frame = (sub[sub['epoch'] == e]
                     [['horse_number', 'horse_name', 'win_odds', 'place_odds']].copy())
            sc = score_race(frame, date_str, venue, rn)
        if sc is None or len(sc) == 0 or bool(sc.iloc[0].get('race_closed')):
            continue
        valid = sc[~sc['unposted']]
        o = pd.to_numeric(valid['win_odds'], errors='coerce')
        o = o[o.notna() & (o >= 1.0)]
        overrun = float((1.0 / o).sum() - 1.0) * 100.0 if len(o) else float('nan')
        thin = (len(valid) < 8) or pd.isna(overrun) or (overrun < 15.0)
        if thin:
            continue
        for _, r in valid.iterrows():
            kind = verdict_of(r)
            if kind[0] != 'prime':
                continue
            odds = numf(r.get('win_odds'))
            po = numf(r.get('place_odds'))
            prob = numf(r.get('prob'))
            place_prob = numf(r.get('place_prob'))
            ev_w = numf(r.get('ev'))
            ev_p = numf(r.get('place_ev'))
            smart = numf(r.get('smart_money_score')) or 50.0
            if odds is None or prob is None:
                continue
            txt = kind[1]
            if 'DUAL' in txt:
                play = 'WP'
            elif 'EXOTIC' in txt:
                play = 'EX'
            elif 'PRIME W' in txt:
                play = 'W'
            elif 'PRIME P' in txt:
                play = 'P'
            else:
                continue
            steamish = (smart >= 70.0) or bool(r.get('syndicate_steam'))
            if play in ('W', 'WP'):
                cpct = prob * 100.0
                tier = 'HIGH' if (cpct >= 20.0 or steamish) else 'MED'
            else:
                cpct = (place_prob or 0.0) * 100.0
                tier = 'HIGH' if (cpct >= 40.0 or steamish) else 'MED'
            buy_w = _min_win_odds(prob, 1.15) if play in ('W', 'WP') else None
            tgt_p = 1.30 if (po is not None and po > 4.0) else 1.15
            buy_p = (tgt_p / place_prob) if (play in ('P', 'WP', 'EX')
                                             and place_prob and place_prob > 0) else None
            reasons = []
            if play in ('W', 'WP') and ev_w is not None:
                reasons.append(f"EV-W ×{ev_w + 1.0:.2f}")
            if play in ('P', 'WP', 'EX') and ev_p is not None:
                reasons.append(f"EV-P ×{ev_p:.2f}")
            if bool(r.get('syndicate_steam')):
                reasons.append('STEAM')
            if bool(r.get('smart_place_absorption')):
                reasons.append('PLACE-ABSORB')
            if bool(r.get('divergence_trap')):
                reasons.append('TRAP')
            if smart >= 75.0:
                reasons.append('S-HOT')
            picks.append({
                'tier': tier,
                'conf': f"{cpct:.0f}%",
                'race': rn,
                'play': play,
                'horse': horse_label(r, html=False),
                'w_odds': odds,
                'p_odds': po,
                'ev_w': ev_w + 1.0 if ev_w is not None else None,
                'ev_p': ev_p,
                'buy_w': buy_w,
                'buy_p': buy_p,
                's': smart,
                'why': ' · '.join(reasons),
            })
    picks.sort(key=lambda x: (0 if x['tier'] == 'HIGH' else 1, -(x['ev_w'] or 0.0)))
    return picks


def build_daily_card_md(date_str: str, venue: str, picks=None) -> str:
    """Professional Markdown of the day's betting plan (downloadable)."""
    if picks is None:
        picks = daily_card_rows(date_str, venue)
    vn = {'ST': 'Sha Tin', 'HV': 'Happy Valley'}.get(str(venue).upper(), str(venue))
    now = hkt_now()

    def f(v, d=1):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"

    L = [f"# HKJC Daily Betting Card — {date_str} {vn}", ""]
    L.append(f"- **Generated**: {now:%Y-%m-%d %H:%M:%S} HKT")
    L.append("- **Play types**: W = Win only · P = Place only · "
             "WP = Win + Place · EX = Exotics (Place / Quinella-Place / Tierce anchor)")
    L.append("- **Buy W ≥ / Buy P ≥**: minimum odds to still hold the EV edge "
             "(favourite-longshot decay adjusted).")
    L.append("")
    if not picks:
        L.append("No actionable plays yet — snapshots/models warming up.")
    else:
        n_high = sum(1 for p in picks if p['tier'] == 'HIGH')
        L.append(f"## {len(picks)} Actionable Plays — {n_high} HIGH conviction")
        L.append("")
        L.append("| Conf | Race | Play | Horse | W odds | P odds | EV(W) | EV(P) | "
                 "Buy W ≥ | Buy P ≥ | S | Why |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in picks:
            L.append(f"| {p['tier']} | R{p['race']} | {p['play']} | {p['horse']} | "
                     f"{f(p['w_odds'])} | {f(p['p_odds'])} | {f(p['ev_w'], 2)} | "
                     f"{f(p['ev_p'], 2)} | {f(p['buy_w'])} | {f(p['buy_p'])} | "
                     f"{p['s']:.0f} | {p['why']} |")
        L.append("")
        L.append("### HIGH conviction plays")
        L.append("")
        for p in [x for x in picks if x['tier'] == 'HIGH']:
            L.append(f"- **R{p['race']} {p['play']}** — {p['horse']}  "
                     f"(odds {f(p['w_odds'])}/{f(p['p_odds'])}, confidence {p['conf']}, {p['why']})")
        L.append("")
        L.append("### Staking guardrails")
        L.append("")
        L.append("- WIN only when P_final ≥ 0.16 AND EV ≥ 1.15 AND O_W ∈ [2.2, 14.0].")
        L.append("- Longshots (O_W > 14) with EV ≥ 1.25 are NEVER straight WIN — "
                 "they route to PLACE / QP / TIERCE only.")
        L.append("- Thin pools (< 8 runners or overround < 15%) carry no staking signal.")
    L.append("")
    L.append("---")
    L.append("Generated by HKJC Quant Terminal — educational use only.")
    return "\n".join(L)


def render_card(date_str: str, venue: str):
    """Professional daily buy-list summary (cold, snapshot-driven)."""
    st.markdown('<div class="qt-top4 qt-term" style="margin-top:6px;">'
                '🎯 TODAY\'S BUY LIST — quant signals · real pools · auto-seeded · 5m refresh</div>',
                unsafe_allow_html=True)
    picks = daily_card_rows(date_str, venue)
    if not picks:
        st.info("No actionable plays yet — pools may still be unformed (⏳); "
                "the card auto-polls every 5 minutes and fills as real odds post.")
        return
    n_high = sum(1 for p in picks if p['tier'] == 'HIGH')
    kpi_strip([
        ("Actionable Plays", f"{len(picks)}"),
        ("HIGH Conviction", f"{n_high}"),
        ("Races Covered", f"{len(set(p['race'] for p in picks))}"),
    ])
    highs = [p for p in picks if p['tier'] == 'HIGH']
    if highs:
        parts = []
        for p in highs[:6]:
            col = PRIME_COLOR if p['play'] in ('W', 'WP') else '#00E5FF'
            ow = p['w_odds'] if p['play'] in ('W', 'WP') else p['p_odds']
            parts.append(f'<span class="r">R{p["race"]}</span> {p["play"]} '
                         f'<b>{esc(p["horse"])}</b> '
                         f'<span style="color:{col};">O {f"{ow:.1f}" if ow is not None else "—"}</span> '
                         f'<span style="color:#8ea2c0;">{p["conf"]}</span>')
        st.markdown(f'<div class="qt-top4 qt-term" style="font-size:12px;">'
                    f'💎 HIGH CONVICTION&nbsp;&nbsp;{"&nbsp;·&nbsp;".join(parts)}</div>',
                    unsafe_allow_html=True)
    df = pd.DataFrame(picks)
    show = df[['tier', 'race', 'play', 'horse', 'w_odds', 'p_odds', 'ev_w', 'ev_p',
               'buy_w', 'buy_p', 's', 'why']].copy()
    show.columns = ['Conf', 'Race', 'Play', 'Horse', 'W odds', 'P odds',
                    'EV (W)', 'EV (P)', 'Buy W ≥', 'Buy P ≥', 'S', 'Why']
    st.dataframe(show, hide_index=True, width='stretch',
                 column_config={
                     'Horse': st.column_config.TextColumn(width='large'),
                     'Why': st.column_config.TextColumn(width='large')})
    st.download_button("📄 Download Today's Card (.md)",
                       data=build_daily_card_md(date_str, venue, picks),
                       file_name=f"hkjc_{date_str}_{venue}_daily_plan.md",
                       mime="text/markdown")


# ----------------------------------------------------------------------
# Overview (cold, 60 s only)
# ----------------------------------------------------------------------
def render_overview(date_str, venue):
    """Cold summary: reads the PERSISTED snapshot store (real data, zero
    scraping) so it never competes with the active race loop (Law 1)."""
    store = snap_store(date_str, venue)
    if len(store) == 0 or 'race_id' not in store.columns:
        st.info("No snapshots yet — pick a race tab to start polling.")
        return
    nos = []
    for rid in store['race_id'].dropna().unique():
        m = re.match(rf"{re.escape(date_str)}_Race(\d+)$", str(rid))
        if m:
            nos.append(int(m.group(1)))
    if not nos:
        st.info("No snapshots yet — pick a race tab to start polling.")
        return
    n = max(nos)
    rows = []
    for rn in range(1, n + 1):
        sub = store[store['race_id'] == f"{date_str}_Race{rn}"]
        if len(sub) == 0:
            continue
        e = pd.to_numeric(sub['epoch'], errors='coerce').max()
        frame = (sub[sub['epoch'] == e]
                 [['horse_number', 'horse_name', 'win_odds', 'place_odds']].copy())
        sc = score_race(frame, date_str, venue, rn)
        if sc is None:
            continue
        valid = sc[~sc['unposted']].copy()
        o = pd.to_numeric(valid['win_odds'], errors='coerce')
        o = o[o.notna() & (o >= 1.0)]   # keep a REAL 1.0 heavy favourite in the book
        overrun = float((1.0 / o).sum() - 1.0) * 100.0 if len(o) else float('nan')
        sm = pd.to_numeric(valid['smart_money_score'], errors='coerce')
        n_valid = int((~sc['unposted']).sum())
        n_pre = int(sc['unposted'].sum())
        closed_flag = bool(sc.get('race_closed', pd.Series([False])).iloc[0]) \
            if len(sc) and 'race_closed' in sc.columns else False
        thin = closed_flag or (n_valid < 8) or pd.isna(overrun) or (overrun < 15.0)
        # signal counts: steamers + primes across BOTH win and place pools
        primes = int(sum(1 for _, rr in valid.iterrows() if verdict_of(rr)[0] == 'prime'))
        # BEST EV across win & place (ratio form)
        cands = []
        for _, rr in valid.iterrows():
            evr = numf(rr.get('ev'))
            pev = numf(rr.get('place_ev'))
            if evr is not None:
                cands.append((float(evr) + 1.0, 'W', rr))
            if pev is not None:
                cands.append((float(pev), 'P', rr))
        if cands and not thin:
            best_val, best_pool, best_row = max(cands, key=lambda c: c[0])
            best_txt = f"{best_pool} ×{best_val:.3f}"
            top_txt = (f"{horse_label(best_row)} "
                       f'<span style="color:#6b7a99;">{best_pool}×{best_val:.2f}</span>')
        elif thin:
            best_txt = '<span style="color:#9E9E9E;">N/A ⏳</span>'
            top_txt = '—'
        else:
            best_txt = '—'
            top_txt = '—'
        # TOP 4 QUANT SELECTIONS: model rank order, saddlecloth law applied
        t4 = valid.dropna(subset=['prob']).nlargest(4, 'prob')
        if len(t4):
            rk = ['1st', '2nd', '3rd', '4th']
            cls = ['qt-r1', 'qt-r2', 'qt-r3', 'qt-r4']
            parts = [f'<span class="{cls[i]}">{rk[i]}: {horse_label(t4.iloc[i])}</span>'
                     for i in range(len(t4))]
            top4_txt = ' <span style="color:#44506a;">&gt;</span> '.join(parts)
        else:
            top4_txt = '—'
        rows.append((rn, (n_valid, n_pre), overrun, best_txt,
                     (int((sm >= 75).sum()), primes), top4_txt, top_txt, thin, closed_flag))

    hdr = "".join(f'<span>{c}</span>' for c in
                  ["RACE", "RUNNERS", "OVERRD.", "BEST EV", "SIGNALS",
                   "TOP 4 QUANT SELECTIONS", "TOP VALUE (OVERLAY)"])
    st.markdown(f'<div class="qt-ov-hdr qt-term">{hdr}</div>', unsafe_allow_html=True)
    for rn, (act, pre), overrun, best_txt, (steam, prime), top4_txt, top_txt, thin, closed_flag in rows:
        if pd.isna(overrun):
            over_txt = '<span>—</span>'
        else:
            over_txt = f'<span>{overrun:+.1f}%</span>'
        liq = ('<span class="qt-lq"> 🏁</span>' if closed_flag
               else ('<span class="qt-lq"> ⏳ LIQ</span>' if thin else ''))
        runner_txt = f'<span>{act}</span>' + (f'<span style="color:#6b7a99;"> {pre} ⏳</span>' if pre else '')
        st.markdown(
            f'<div class="qt-ov-row qt-term">'
            f'<span style="font-weight:700;">R{rn}</span>{liq}'
            + runner_txt + over_txt
            + f'<span>{best_txt}</span>'
            + f'<span>{steam} / {prime}</span>'
            + f'<span class="qt-ov-top4">{top4_txt}</span>'
            + f'<span style="color:{PRIME_COLOR};">{top_txt}</span>'
            + '</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Terminal shell
# ----------------------------------------------------------------------
ensure_data_loaded()

today = datetime.now(HKT)
with st.sidebar:
    st.markdown("### ⚙️ Terminal Controls")
    # Race-day default: TODAY (not tomorrow) - the terminal is for the live card.
    date_str = st.text_input("Race day (YYYY-MM-DD)",
                             value=today.strftime("%Y-%m-%d"))
    # Venue default follows the HKJC fixture convention: Wed night = Happy
    # Valley, Sun/Sat day = Sha Tin. Override anytime via the dropdown.
    default_venue = "HV" if today.weekday() == 2 else "ST"
    venue = st.selectbox("Venue", ["ST", "HV"],
                         index=0 if default_venue == "ST" else 1)
    poll_s = st.select_slider("Active race poll (s)", options=[1, 2], value=1)
    auto = st.checkbox("Auto-refresh", value=True)
    force = st.button("🔄 Poll Now", type="primary", width='stretch')
    st.divider()
    st.caption("Law 1: only the selected race is scraped/inferred per tick.")
    st.caption("Law 2: real snapshots only · ⏳ when a card is unformed (whole field ≤ 1.01).")
    st.caption(f"Snapshots: data/odds_snapshots/{date_str.replace('-', '')}_{venue}.csv")

options = ["🎯 Today's Card", "📊 Overview"] + [f"R{i}" for i in range(1, 13)]
mode = st.radio("View", options, horizontal=True, label_visibility="collapsed", key="qt_view")

if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        interval = 60_000 if mode in ("🎯 Today's Card", "📊 Overview") else int(poll_s * 1000)
        st_autorefresh(interval=interval, key="qt_auto")
    except Exception:
        pass

if mode == "🎯 Today's Card":
    st.session_state['qt_active_race'] = None
    header_panel(date_str, venue, None, None,
                 '<span style="color:#4dabf7;">🎯 DAILY PLAN · REAL POLL · 5m</span>')
    render_card(date_str, venue)
    st.stop()

if mode == "📊 Overview":
    st.session_state['qt_active_race'] = None
    header_panel(date_str, venue, None, None,
                 '<span style="color:#9E9E9E;">🌙 COLD · 60s</span>')
    render_overview(date_str, venue)
    st.stop()

race_no = int(mode[1:])
st.session_state['qt_active_race'] = race_no

post_dt = race_post_time(date_str, venue, race_no)
if not force and post_dt is not None:
    poll_ttl = effective_poll_ttl(post_dt, float(poll_s))
else:
    poll_ttl = 0.0 if force else float(poll_s)
live = poll_race(date_str, venue, race_no, ttl=poll_ttl)
ok = live is not None and len(live) > 0
st.session_state['qt_last_poll'] = time.time()
st.session_state['qt_last_ok'] = ok
status = ('<span style="color:#00E676;">🟢 LIVE</span>' if ok else
          '<span style="color:#FF5252;">🔴 NO DATA</span>')

# post time may have just been captured by the scrape above
post_dt = post_dt or race_post_time(date_str, venue, race_no)
header_panel(date_str, venue, race_no, post_dt, status)

scored = score_race(live, date_str, venue, race_no)
if scored is not None:
    render_focus(scored, race_no)
else:
    st.info(f"R{race_no}: odds not posted yet — polling every {poll_s}s.")


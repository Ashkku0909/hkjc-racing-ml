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

import web_live
from web_live import (
    ensure_data_loaded, poll_race, card_size, score_race, flag_row,
    de_vig_market_probs, race_post_time, snap_store, SLOW_TTL,
)

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
    """Prime Win: EV_win >= 1.22 & win_odds in [4.5, 8.0] & S >= 50."""
    odds = num(r.get('win_odds'))
    ev = num(r.get('ev'))
    smart = num(r.get('smart_money_score'), 50.0)
    if odds is None or ev is None:
        return False
    return 4.5 <= odds <= 8.0 and (ev + 1.0) >= 1.22 and smart >= 50


def place_prime(r) -> bool:
    """Prime Place (conservative Henery calibration):
      O_P in [1.8, 4.0] -> EV_place >= 1.15
      O_P > 4.0 (longshot) -> EV_place >= 1.30
    plus S >= 50."""
    po = num(r.get('place_odds'))
    pev = num(r.get('place_ev'))
    smart = num(r.get('smart_money_score'), 50.0)
    if po is None or pev is None:
        return False
    if po > 4.0:
        return 1.8 < po and pev >= 1.30 and smart >= 50
    return 1.8 <= po <= 4.0 and pev >= 1.15 and smart >= 50


def verdict_of(r):
    if bool(r.get('unposted')):
        return 'preopen', '⏳ PRE-OPEN'
    pw = win_prime(r)
    pp = place_prime(r)
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
    pace_scn = str(scored.iloc[0].get('pace_scenario') or 'NORMAL')
    pace_n = num(scored.iloc[0].get('pace_n_leaders'), 0.0)
    if pace_scn == 'MELTDOWN':
        st.markdown(f'<span class="qt-pace meltdown qt-term">🔥 PACE MELTDOWN '
                    f'({pace_n:.0f} leaders) — front-runners penalised, closers boosted</span>',
                    unsafe_allow_html=True)
    elif pace_scn == 'LONE':
        st.markdown('<span class="qt-pace lone qt-term">🚀 LONE LEADER (slow bias) '
                    '— leader logit +0.20</span>', unsafe_allow_html=True)
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
        prob = num(r.get('prob'))
        mkt = num(r.get('mkt_prob'))
        evr = num(r.get('ev'))
        evx = evr + 1.0 if evr is not None else None
        pev = num(r.get('place_ev'))
        p3 = num(r.get('p_top3'))
        smart = num(r.get('smart_money_score'), 50.0)
        mp = f"{prob * 100:.1f}%" if prob is not None else "—"
        kp = f"{mkt * 100:.1f}%" if mkt is not None else "—"
        t3pct = f"{p3 * 100:.1f}%" if p3 is not None else "—"
        mbar = min(100.0, (prob or np.nan) * 100) if prob is not None else 0.0
        kbar = min(100.0, (mkt or np.nan) * 100) if mkt is not None else 0.0
        evtxt = f"×{evx:.3f}" if evx is not None else "—"
        pevtxt = f"×{pev:.3f}" if pev is not None else "—"
        w_td = num(r.get('tick_delta'))
        p_td = num(r.get('tick_delta_place'))
        w_flag = win_prime(r)
        pp_flag = place_prime(r)
        vt_cls = "prime" if kind[0] == 'prime' else ("preopen" if kind[0] == 'preopen' else "pass")
        vtxt = kind[1]
        if kind[0] == 'prime' and ('PRIME W' in kind[1] or 'DUAL' in kind[1]):
            kr = num(r.get('kelly'))
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
            f'<div class="qt-smart">S {smart:.0f} {flow_badge(r.get("flow_signal"), smart)}</div>'
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
        thin = (n_valid < 8) or pd.isna(overrun) or (overrun < 15.0)
        # signal counts: steamers + primes across BOTH win and place pools
        primes = int(sum(1 for _, rr in valid.iterrows() if verdict_of(rr)[0] == 'prime'))
        # BEST EV across win & place (ratio form)
        cands = []
        for _, rr in valid.iterrows():
            evr = num(rr.get('ev'))
            pev = num(rr.get('place_ev'))
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
                     (int((sm >= 75).sum()), primes), top4_txt, top_txt, thin))

    hdr = "".join(f'<span>{c}</span>' for c in
                  ["RACE", "RUNNERS", "OVERRD.", "BEST EV", "SIGNALS",
                   "TOP 4 QUANT SELECTIONS", "TOP VALUE (OVERLAY)"])
    st.markdown(f'<div class="qt-ov-hdr qt-term">{hdr}</div>', unsafe_allow_html=True)
    for rn, (act, pre), overrun, best_txt, (steam, prime), top4_txt, top_txt, thin in rows:
        if pd.isna(overrun):
            over_txt = '<span>—</span>'
        else:
            over_txt = f'<span>{overrun:+.1f}%</span>'
        liq = '<span class="qt-lq"> ⏳ LIQ</span>' if thin else ''
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
    date_str = st.text_input("Race day (YYYY-MM-DD)",
                             value=(today + timedelta(days=1)).strftime("%Y-%m-%d"))
    venue = st.selectbox("Venue", ["ST", "HV"], index=0)
    poll_s = st.select_slider("Active race poll (s)", options=[1, 2], value=1)
    auto = st.checkbox("Auto-refresh", value=True)
    force = st.button("🔄 Poll Now", type="primary", use_container_width=True)
    st.divider()
    st.caption("Law 1: only the selected race is scraped/inferred per tick.")
    st.caption("Law 2: real snapshots only · ⏳ when a card is unformed (whole field ≤ 1.01).")
    st.caption(f"Snapshots: data/odds_snapshots/{date_str.replace('-', '')}_{venue}.csv")

options = ["📊 Overview"] + [f"R{i}" for i in range(1, 13)]
mode = st.radio("View", options, horizontal=True, label_visibility="collapsed", key="qt_view")

if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        interval = 60_000 if mode == "📊 Overview" else int(poll_s * 1000)
        st_autorefresh(interval=interval, key="qt_auto")
    except Exception:
        pass

if mode == "📊 Overview":
    st.session_state['qt_active_race'] = None
    header_panel(date_str, venue, None, None,
                 '<span style="color:#9E9E9E;">🌙 COLD · 60s</span>')
    render_overview(date_str, venue)
    st.stop()

race_no = int(mode[1:])
st.session_state['qt_active_race'] = race_no

post_dt = race_post_time(date_str, venue, race_no)
ttl = 0.0 if force else float(poll_s)
live = poll_race(date_str, venue, race_no, ttl=ttl)
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


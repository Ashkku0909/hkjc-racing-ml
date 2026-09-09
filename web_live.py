"""web_live.py — persistent live-engine helpers for the Streamlit EV board.

IMPORTANT (Streamlit semantics): the app script re-executes on every rerun but
MODULE-LEVEL state in an imported module persists for the process lifetime.
All caches / the asyncio loop / odds snapshots live HERE, never in app.py.

REAL DATA ONLY: polls bet.hkjc.com wp pages; every poll is persisted to
data/odds_snapshots via scraping.live_scraper.
"""
import asyncio
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from scraping.live_scraper import scrape_live_odds, load_odds_snapshots, RACE_META
from bot.analyzer_service import (
    get_data, merge_live_odds_with_predictions,
    calculate_smart_money_metrics,
    get_flow_signal, SMART_MONEY_CENTER,
)
from modeling.model_training import EDGE_DECAY_C, EDGE_DECAY_GAMMA
from modeling.exotics_pricing import (
    henery_gamma_for_field,
    smart_place_absorption_mask,
)

KELLY_MIN_EV = 0.22
KELLY_MIN_ODDS = 4.5
KELLY_MAX_ODDS = 8.0
KELLY_FRACTION = 0.15
KELLY_MAX_BET = 0.02
# Live-field fallback temperature: last-start true_probs (e.g. VERBIER 0.617
# from a 1.8 favourite start) are past-race values, NOT today's probabilities;
# an untempered softmax turns them into a 76% monopoly. logit / TAU flattens the
# distribution (p^(1/tau)) so no single horse dominates an 8-14 runner card.
LIVE_FIELD_TAU = 1.75
# Henery discount (order statistics): conditional place probs use p^gamma /
# sum p^gamma (gamma = 0.81) - fixes Harville's systematic OVER-estimation of
# place chances for extreme longshots (O >= 10).
HENERY_GAMMA = 0.81
# Rolling smart-money window: flow is also measured against the snapshot closest
# to (now - RECENT_FLOW_WINDOW_S) so mid-session surges trigger regardless of
# when the board booted (day-open baseline alone stays flat if the app started
# mid-market, e.g. the 09:14 UTC store captured after the overnight move).
RECENT_FLOW_WINDOW_S = 600.0
# A race is only CLOSED when its nominal post time has passed AND the quotes
# have stopped moving for this long. Races run late (13:00 post often starts
# 13:01+), so a still-ticking market must stay OPEN to capture final flow.
CLOSED_FROZEN_SEC = 60.0
# --- Dynamic post-time execution state machine ---------------------------
# PRE_POST -> TURBO_APPROACH (T-90s) -> LOADING_DELAY (past nominal, ticking)
#                                      -> OFFICIAL_CLOSED (frozen >= 60s)
EXEC_STATES = ("PRE_POST", "TURBO_APPROACH", "LOADING_DELAY", "OFFICIAL_CLOSED")
TURBO_LEAD_S = 90.0          # TURBO_APPROACH starts 90s before nominal post
TURBO_TTL_S = 0.8            # throttled polling cadence inside the final window
T_TILDE_MIN = 20.0           # effective time-to-jump floor (never decays to 0)
T_TILDE_MAX = 45.0           # effective time-to-jump cap in the final band
W_TILDE_EARLY = 0.25         # asymptote far from post (model-dominated)
W_TILDE_TURBO = 0.85         # asymptote at post (market-dominated, pinned in TURBO)
# Smooth continuous Bayesian post-time weight:
#   w(T~) = w_max - (w_max - w_min) / (1 + exp(-kappa * (T~ - T0)))
#   w_max = 0.85 (market), w_min = 0.25 (model), T0 = 360s, kappa = 0.02.
W_TILDE_T0 = 360.0
W_TILDE_KAPPA = 0.02
FLOW_WINDOW_S = 90.0         # relative-velocity lookback window
FLOW_STEAM_V = -0.15         # reference: -15% win price over the window
FLOW_STEAM_LOGIT = math.log(1.0 + FLOW_STEAM_V)   # -0.1625 in logit(odds) space
FLOW_DELTA_MIN = 0.10        # logit booster for a detected steam
FLOW_DELTA_MAX = 0.35
TRAP_MODEL_MIN = 0.30        # divergence-trap defense (adverse selection)
TRAP_MARKET_MAX = 0.09
TRAP_T_TILDE_S = 120.0       # trap armed only inside the final 2 minutes
TRAP_LOGIT_PENALTY = -1.5    # heavy penalty on the fused logit
# Dual-mode bet filter (win vs exotics routing)
WIN_MIN_PROB = 0.16
WIN_MIN_EV = 1.15
WIN_MIN_ODDS = 2.2
WIN_MAX_ODDS = 14.0
EXOTIC_EV = 1.25
SCRAPE_TIMEOUT = 45.0
MAX_RACES = 12
FOCUS_TTL = 5.0
SLOW_TTL = 60.0
DISCOVER_TTL = 300.0

# --- persistent asyncio loop + browser (module-level = process-lifetime) ---
_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True, name="hkjc-asyncio").start()
        _loop = loop
    return _loop


def run_async(coro):
    """Schedules on the persistent loop; returns a concurrent Future."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())


# --- persistent caches ---
_poll_cache: dict = {}
_snap_cache: dict = {}
_card_cache: dict = {}
_inflight: dict = {}     # (date, venue, race) -> (ts, Future)  single-flight guard
_data_loaded: bool = False
# --- live-validation audit trail (Master-Rule observation, printed once) ---
_engine_state_prev: dict = {}     # race_id -> last printed execution state
_steam_seen: set = set()          # (race_id, sorted steamer names) seen


def ensure_data_loaded() -> None:
    """Loads the model prediction store ONCE per process (not per rerun)."""
    global _data_loaded
    if not _data_loaded:
        from bot.analyzer_service import load_data
        load_data()
        _data_loaded = True


def _decay(odds: float) -> float:
    # Favorite-longshot decay is a penalty (<=1) for odds > C; for short prices
    # (e.g. a 1.0 heavily-banked favourite) (C/o)^gamma would inflate EV, so clamp
    # to 1.0 and use the raw EV there.
    if not odds or odds <= 0:
        return 0.0
    return min((EDGE_DECAY_C / float(odds)) ** EDGE_DECAY_GAMMA, 1.0)


def race_meta(date_str: str, venue: str, race_no: int) -> dict:
    """Header metadata ({'title', 'post_hhmm', ...}) captured by the last scrape."""
    return RACE_META.get((date_str, venue, int(race_no))) or {}


def race_post_time(date_str: str, venue: str, race_no: int):
    """HKT-aware post datetime parsed from the WP page header, or None."""
    meta = race_meta(date_str, venue, race_no)
    hhmm = meta.get('post_hhmm')
    if not hhmm:
        return None
    try:
        dt = datetime.strptime(f"{date_str} {hhmm}", "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=timezone(timedelta(hours=8)))
    except Exception:
        return None


def de_vig_market_probs(win_odds: pd.Series) -> pd.Series:
    """Power-law de-vig (find alpha with sum(p^alpha) = 1, brentq) - same model
    as modeling.apply_power_law. Returns normalized fair market probabilities.

    Includes a REAL 1.0 favourite (an open pool legitimately shows one - e.g.
    2026-09-06 ST R3, ~97% of the win pool). With p=[1.0, tiny...] the power-law
    has no finite root (sum >= 1 always), so the normalized fallback takes over
    and correctly assigns ~0.997 to the favourite instead of spreading the
    longshots to 100%."""
    from scipy.optimize import brentq
    o = pd.to_numeric(win_odds, errors='coerce')
    idx = o[o.notna() & (o >= 1.0) & (o > 0)].index
    impl = 1.0 / o.loc[idx].to_numpy(dtype=float)
    out = pd.Series(np.nan, index=win_odds.index)
    if len(impl) < 2:
        return out

    def obj(a: float) -> float:
        return float(np.sum(np.power(impl, a)) - 1.0)

    try:
        alpha = brentq(obj, 0.01, 2.0)
        p = np.power(impl, alpha)
    except Exception:
        p = impl / impl.sum()
    out.loc[idx] = p
    return out


def poll_race(date_str: str, venue: str, race_no: int, ttl: float,
              skip_extras: bool = False):
    """Real poll with per-race TTL + SINGLE-FLIGHT: concurrent reruns share the
    same in-flight scrape instead of stacking duplicate browser sessions.

    LOW-LATENCY: while a scrape is already in flight, reruns serve the LAST
    GOOD frame immediately instead of blocking on the network (the in-flight
    call is the only one that waits, and it refreshes the cache on completion).
    skip_extras=True is the light scan used by the Today's Buy-List (no
    SpeedPRO/FormGuide/WPQ/DrawStats gather per race).
    """
    key = (date_str, venue, race_no)
    now = time.time()
    hit = _poll_cache.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]

    inf = _inflight.get(key)
    if inf is not None and now - inf[0] < 45.0:
        # a poll is already running - never block the UI rerun on it
        try:
            return hit[1] if hit is not None else inf[1].result(timeout=SCRAPE_TIMEOUT)
        except Exception:
            return hit[1] if hit is not None else None

    fut = run_async(scrape_live_odds(date_str, venue, race_no, skip_extras=skip_extras))
    _inflight[key] = (now, fut)
    try:
        df = fut.result(timeout=SCRAPE_TIMEOUT)
    except Exception:
        df = None
    _poll_cache[key] = (time.time(), df)
    _inflight.pop(key, None)
    return df


def snap_store(date_str: str, venue: str) -> pd.DataFrame:
    key = (date_str, venue)
    now = time.time()
    hit = _snap_cache.get(key)
    if hit is not None and now - hit[0] < 30.0:
        return hit[1]
    df = load_odds_snapshots(date_str, venue)
    _snap_cache[key] = (now, df)
    return df


def card_size(date_str: str, venue: str) -> int:
    """Full-card size: fast-path from persisted snapshots, else probe races
    until 2 consecutive empty (cached 90s)."""
    key = (date_str, venue)
    now = time.time()
    hit = _card_cache.get(key)
    if hit is not None and now - hit[0] < 90.0:
        return hit[1]
    # Fast path: already-persisted snapshots tell us the real card size without
    # any extra scraping (Law 1 - Overview stays cold/light).
    store = snap_store(date_str, venue)
    if len(store) and 'race_id' in store.columns:
        nos = []
        for rid in store['race_id'].dropna().unique():
            m = re.match(rf"{re.escape(str(date_str))}_Race(\d+)$", str(rid))
            if m:
                nos.append(int(m.group(1)))
        if nos:
            n = max(nos)
            _card_cache[key] = (now, n)
            return n
    n, empty_run = 0, 0
    for r in range(1, MAX_RACES + 1):
        df = poll_race(date_str, venue, r, ttl=DISCOVER_TTL)
        if df is not None and len(df) > 0:
            n = r
            empty_run = 0
        else:
            empty_run += 1
            if empty_run >= 2:
                break
    _card_cache[key] = (now, n)
    return n


def _valid_odds_mask(win: pd.Series, place: pd.Series = None,
                     open_race: Optional[bool] = None) -> pd.Series:
    """Real-data pre-open sentry (race-level, not per-row).

    Verified against live HKJC behaviour (2026-09-06 ST R3, the Chief
    Executive's Cup): an OPEN win pool can legitimately show a 1.0 favourite
    (~97% of the win pool) alongside win 197-490 / place 6-24 longshots.
    The old 'win > 1.01' and 'win > 5*place' rules wrongly killed that card.

    A *pre-open* card instead shows 1.0 for EVERY runner. So:
      - if no horse posts a real price (>1.01)  -> whole race unposted
      - otherwise EVERY horse with a real (non-NaN) win price is valid,
        including a 1.0 favourite.
    """
    win = pd.to_numeric(win, errors='coerce')
    if open_race is None:
        open_race = bool(((win > 1.01).any())) if len(win) else False
    if not open_race:
        return pd.Series(False, index=win.index)
    return pd.Series(win.notna().to_numpy(dtype=bool), index=win.index)


def _first_valid_odds_map(store_race: pd.DataFrame):
    """TRUE BASELINE: per horse, the FIRST VALID POST-OPEN snapshot odds.

    'Open' is decided RACE-LEVEL: the first epoch where ANY horse posts a real
    win price (>1.01). A horse whose own price is 1.0 (heavy favourite) is still
    a valid baseline once the race is open.
    """
    base_rows = []
    sub = store_race.sort_values('epoch')
    open_epoch = None
    for e, g in sub.groupby('epoch'):
        wins = pd.to_numeric(g['win_odds'], errors='coerce')
        if bool((wins > 1.01).any()):
            open_epoch = e
            break
    if open_epoch is None:
        return pd.DataFrame(base_rows)
    for hn, g in sub.groupby('horse_number'):
        g = g[g['epoch'] >= open_epoch]
        ok = _valid_odds_mask(pd.Series(g['win_odds']),
                              pd.Series(g.get('place_odds')), open_race=True)
        if ok.any():
            r = g[ok.values].iloc[0]
            base_rows.append({'horse_number': hn,
                              'horse_name': r.get('horse_name', ''),
                              'win_odds': float(r['win_odds'])})
    return pd.DataFrame(base_rows)


def _field_model_probs(live_df: pd.DataFrame, all_df: pd.DataFrame) -> pd.Series:
    """Leak-free live-card model probabilities from the real store.

    Uses each horse's MOST RECENT walk-forward true_prob (last start), then a
    per-race softmax over the covered runners. When fewer than half the field
    has model history (e.g. debutants / early card), returns NaN for everyone:
    the board then shows market-only, never inflated pseudo-probabilities.
    """
    if all_df is None or len(all_df) == 0 or 'true_prob' not in all_df.columns:
        return pd.Series(np.nan, index=live_df.index)
    hist = all_df[['horse_name', 'race_date', 'true_prob']].copy()
    hist['_key'] = hist['horse_name'].astype(str).str.upper()
    hist['race_date'] = pd.to_datetime(hist['race_date'], errors='coerce')
    lastp = (hist.sort_values('race_date')
             .drop_duplicates('_key', keep='last')
             .set_index('_key')['true_prob'])
    probe = live_df['horse_name'].astype(str).str.upper().map(lastp)
    probe = pd.to_numeric(probe, errors='coerce')
    covered = probe.notna().sum() / max(len(live_df), 1)
    if covered < 0.5:
        return pd.Series(np.nan, index=live_df.index)
    p = np.clip(probe.fillna(np.nan), 1e-12, 1.0)
    # temperature-scaled softmax over model log-probs: exp(log(p)/tau) = p^(1/tau)
    # flattens the last-start prior so a single 0.6 horse cannot monopolise a
    # 14-runner race (VERBIER 76.2% -> ~25% at tau=1.75).
    logit = np.log(p) / LIVE_FIELD_TAU
    valid = ~np.isnan(logit)
    if not valid.any():
        return pd.Series(np.nan, index=live_df.index)
    e = np.where(valid, np.exp(logit - np.nanmax(logit)), 0.0)
    return pd.Series(e / e.sum(), index=live_df.index)


_pace_cache = None


def _pace_profiles() -> dict:
    """horse_name (upper) -> (run_style, avg_early_pct_last_3, avg_late_speed_last_3)
    from each horse's MOST RECENT run in the feature store (lagged, leak-free)."""
    global _pace_cache
    if _pace_cache is None:
        d = {}
        try:
            f = pd.read_csv('data/model_features.csv',
                            usecols=['race_date', 'horse_name', 'run_style',
                                     'avg_early_pct_last_3', 'avg_late_speed_last_3'],
                            low_memory=False)
            f['_key'] = f['horse_name'].astype(str).str.upper()
            f['race_date'] = pd.to_datetime(f['race_date'], errors='coerce')
            f = f.sort_values('race_date').drop_duplicates('_key', keep='last')
            for _, r in f.iterrows():
                d[str(r['_key'])] = (str(r.get('run_style') or ''),
                                     float(r['avg_early_pct_last_3'])
                                     if pd.notna(r.get('avg_early_pct_last_3')) else np.nan,
                                     float(r['avg_late_speed_last_3'])
                                     if pd.notna(r.get('avg_late_speed_last_3')) else np.nan)
        except Exception as e:
            print(f"pace profile load failed: {e}")
            d = {}
        _pace_cache = d
    return _pace_cache


def _apply_pace_adjustment(probs: pd.Series, runner_df) -> tuple:
    """Pace Scenario Matrix (lagged front-runner density, leak-free).

      Pace Meltdown (>= 3 Leaders): front-runner logits -0.15, closers +0.10
      Lone Leader / Slow Bias (exactly 1 Leader): leader logit +0.20
    Returns (adjusted probs Series, (scenario, n_leaders)).
    """
    profiles = _pace_profiles()

    def style_of(name):
        return profiles.get(str(name).upper(), ('', np.nan, np.nan))[0]

    styles = runner_df['horse_name'].astype(str).map(style_of)
    leaders = (styles == 'Leader').to_numpy(dtype=bool)
    closers = (styles == 'Closer').to_numpy(dtype=bool)
    backmarkers = (styles.isin(['Closer', 'Backmarker'])).to_numpy(dtype=bool)
    front = (styles.isin(['Leader', 'Presser'])).to_numpy(dtype=bool)
    n_lead = int(leaders.sum())
    p = pd.to_numeric(probs, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(p)
    scenario = 'NORMAL'
    # Master Rule 2 (pace clashing): N_leaders >= 3 -> front-runners clash,
    # positive drift applied to closers/backmarkers.
    if n_lead >= 3:
        scenario = 'MELTDOWN'
        adj = np.where(leaders, -0.15, np.where(closers, 0.10, 0.0))
        if np.any(adj[valid]):
            pv = np.clip(p, 1e-12, 1.0)
            logit = np.log(pv / (1.0 - pv)) + adj
            p = 1.0 / (1.0 + np.exp(-logit))
            p[~valid] = np.nan
            if np.nansum(p) > 0:
                p = p / np.nansum(p)
    elif n_lead == 1:
        scenario = 'LONE'
        adj = np.where(leaders, 0.20, 0.0)
        if np.any(adj[valid]):
            pv = np.clip(p, 1e-12, 1.0)
            logit = np.log(pv / (1.0 - pv)) + adj
            p = 1.0 / (1.0 + np.exp(-logit))
            p[~valid] = np.nan
            if np.nansum(p) > 0:
                p = p / np.nansum(p)
    elif n_lead == 0:
        # Slow pace / zero pace-setters: on the narrow Happy Valley course a
        # hold-up runner parked wide is nearly a death sentence, while an
        # inside gate (<=4) front-runner/presser can steal a slow race.
        scenario = 'SLOW'
        is_hv = False
        if 'venue' in runner_df.columns and len(runner_df):
            try:
                is_hv = str(pd.Series(runner_df['venue']).iloc[0]).upper() == 'HV'
            except Exception:
                is_hv = False
        draw = pd.to_numeric(runner_df.get('barrier_draw'), errors='coerce')
        if is_hv:
            dv = draw.to_numpy(dtype=float)
            adj = np.where(backmarkers & (dv >= 8.0), -0.15, 0.0)          # wide closer
            adj = np.where(front & (dv >= 1.0) & (dv <= 4.0), adj + 0.10, adj)  # rail front
            if np.any(adj[valid]):
                pv = np.clip(p, 1e-12, 1.0)
                logit = np.log(pv / (1.0 - pv)) + adj
                p = 1.0 / (1.0 + np.exp(-logit))
                p[~valid] = np.nan
                if np.nansum(p) > 0:
                    p = p / np.nansum(p)
    return pd.Series(p, index=probs.index), (scenario, n_lead)


def _market_frozen(sub: pd.DataFrame, frozen_sec: float = CLOSED_FROZEN_SEC) -> bool:
    """True iff no win quote has changed for frozen_sec (pool truly closed).

    Nominal post times run early (a 13:00 card often starts 13:01+), so a market
    that is still ticking must NOT be flagged closed - that would discard the
    final smart-money surge. Only a frozen pool is 'RACE CLOSED'.
    """
    try:
        eras = sorted(pd.to_numeric(sub['epoch'], errors='coerce').dropna().unique())
        if len(eras) < 2:
            return False
        last_move = float(eras[0])
        for i in range(len(eras) - 1, 0, -1):
            e0, e1 = eras[i - 1], eras[i]
            f0 = sub[sub['epoch'] == e0].set_index('horse_number')['win_odds']
            f1 = sub[sub['epoch'] == e1].set_index('horse_number')['win_odds']
            common = f0.index.intersection(f1.index)
            if len(common) and (pd.to_numeric(f1.loc[common], errors='coerce') !=
                                pd.to_numeric(f0.loc[common], errors='coerce')).any():
                last_move = float(e1)
                break
        return (time.time() - last_move) >= frozen_sec
    except Exception:
        return False


# =====================================================================
# Dynamic post-time execution state machine (HKJC off-time slippage)
# =====================================================================
HKT_TZ = timezone(timedelta(hours=8))


def _now_hkt() -> datetime:
    return datetime.now(HKT_TZ)


def t_tilde_seconds(post_dt) -> Optional[float]:
    """Effective time-to-jump in seconds.

    Real seconds to nominal post, but inside the final 45s band the value is
    STRETCHED into the rolling [T-45s, T-20s] interval so the fusion weight
    never collapses to zero before the actual jump (HKJC posts run late).
    """
    if post_dt is None:
        return None
    s = (post_dt - _now_hkt()).total_seconds()
    if s > T_TILDE_MAX:
        return float(s)
    return float(min(max(s, T_TILDE_MIN), T_TILDE_MAX))


def execution_state(post_dt, sub: Optional[pd.DataFrame] = None,
                    frozen_sec: float = CLOSED_FROZEN_SEC,
                    now: Optional[datetime] = None) -> str:
    """Adaptive 4-state execution lifecycle.

    PRE_POST          t < nominal - 90s
    TURBO_APPROACH    nominal - 90s <= t < nominal
    LOADING_DELAY     t >= nominal while the pool is STILL ticking (gate
                      loading / vet checks push the actual jump 1-3 min late)
    OFFICIAL_CLOSED   past nominal AND zero odds change for >= frozen_sec,
                      OR the scraper saw an official freeze banner.

    `sub` is the persisted snapshot store for the race (epoch + win_odds);
    when absent, past-nominal is conservatively LOADING_DELAY (never closed).
    """
    now = now or _now_hkt()
    if post_dt is None:
        return "PRE_POST"
    s = (post_dt - now).total_seconds()
    if s >= TURBO_LEAD_S:
        return "PRE_POST"
    if s >= 0:
        return "TURBO_APPROACH"
    if sub is not None and len(sub) and _market_frozen(sub, frozen_sec):
        return "OFFICIAL_CLOSED"
    return "LOADING_DELAY"


def fusion_weight(post_dt, state: Optional[str] = None) -> float:
    """Smooth continuous Bayesian post-time market weight w(T~).

        w(T~) = w_max - (w_max - w_min) / (1 + exp(-kappa * (T~ - T0)))
        w_max = 0.85, w_min = 0.25, T0 = 360 s, kappa = 0.02

    Pinned at peak (0.85) throughout TURBO_APPROACH and LOADING_DELAY so
    last-second syndicate flows are always absorbed (Master Rule 1 / 2).
    """
    if state is None:
        state = execution_state(post_dt)
    if state in ("TURBO_APPROACH", "LOADING_DELAY"):
        return W_TILDE_TURBO
    tt = t_tilde_seconds(post_dt)
    if tt is None:
        return W_TILDE_EARLY
    z = math.exp(-W_TILDE_KAPPA * (tt - W_TILDE_T0))
    return float(W_TILDE_TURBO - (W_TILDE_TURBO - W_TILDE_EARLY) / (1.0 + z))


def effective_poll_ttl(post_dt, base_ttl: float) -> float:
    """Throttle the poll cadence: 0.8s inside the final window, base otherwise."""
    st = execution_state(post_dt)
    if st in ("TURBO_APPROACH", "LOADING_DELAY"):
        return TURBO_TTL_S
    return float(base_ttl)


def bayesian_fusion(p_model: np.ndarray, p_market: np.ndarray, w: float,
                    delta_flow: Optional[np.ndarray] = None) -> np.ndarray:
    """Time-varying logit fusion:

        logit(P_final) = (1-w) * logit(P_model) + w * logit(P_market)
                         + delta_flow

    followed by a field softmax so sum(P_final) == 1. NaN-safe per runner:
    a missing model prob falls back to the market component and vice versa.
    """
    pm = np.asarray(p_model, dtype=float)
    pk = np.asarray(p_market, dtype=float)
    n = max(len(pm), len(pk))
    pm = np.resize(pm, n)
    pk = np.resize(pk, n)
    df_ = np.zeros(n) if delta_flow is None else np.asarray(delta_flow, dtype=float)

    def _logit(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, 1e-12, 1.0 - 1e-12)
        out = np.log(pc / (1.0 - pc))
        out[~np.isfinite(p)] = np.nan
        return out

    lm = _logit(pm)
    lk = _logit(pk)
    fused = np.full(n, np.nan)
    for i in range(n):
        parts: list = []
        weights: list = []
        if np.isfinite(lm[i]):
            parts.append(lm[i])
            weights.append(1.0 - w)
        if np.isfinite(lk[i]):
            parts.append(lk[i])
            weights.append(w)
        if weights:
            ws = sum(weights)
            fused[i] = sum(x * wt for x, wt in zip(parts, weights)) / ws + df_[i]
    valid = np.isfinite(fused)
    if not valid.any():
        return np.full(n, np.nan)
    e = np.where(valid, np.exp(fused - np.nanmax(fused)), 0.0)
    return e / e.sum()


def _odds_velocity(sub: pd.DataFrame, window_s: float = FLOW_WINDOW_S) -> dict:
    """horse_number -> {'v90': raw rel. odds change over the window,
                         'logit_vel': logit(odds) velocity, 'wp_contract': bool}.

    Microstructure is measured in LOG-ODDS space (Master Rule 1): the implied
    logit move dlogit(O) = ln(O_now / O_base) is scale-free, so a steam on a
    3.0 favourite and one on a 60.0 longshot are comparable (raw % changes
    would otherwise carry odds-band scale bias). A sharp drop
    (logit_vel <= FLOW_STEAM_LOGIT, i.e. <= -15% price) with a contracting
    O_W/O_P ratio is an active syndicate steam.
    """
    out: dict = {}
    try:
        s = sub.copy()
        s['epoch'] = pd.to_numeric(s['epoch'], errors='coerce')
        s = s.dropna(subset=['epoch'])
        if len(s) < 2:
            return out
        eras = sorted(s['epoch'].unique())
        cur_e = eras[-1]
        cutoff = time.time() - window_s
        past = [e for e in eras if e <= cutoff]
        base_e = past[-1] if past else eras[0]
        if base_e == cur_e:
            return out

        def frame(e: float) -> dict:
            f = s[s['epoch'] == e]
            m = {}
            for _, r in f.iterrows():
                try:
                    hn = int(r['horse_number'])
                except (TypeError, ValueError):
                    continue
                w = pd.to_numeric(pd.Series([r.get('win_odds')]), errors='coerce').iloc[0]
                p = pd.to_numeric(pd.Series([r.get('place_odds')]), errors='coerce').iloc[0]
                m[hn] = (w, p)
            return m

        base = frame(base_e)
        cur = frame(cur_e)
        for hn, (ow, op) in cur.items():
            if hn not in base or pd.isna(ow) or ow <= 0:
                continue
            bw, bp = base[hn]
            if pd.isna(bw) or bw <= 0:
                continue
            v90 = (float(ow) - float(bw)) / float(bw)
            logit_vel = float(math.log(float(ow) / float(bw)))
            wp_contract = False
            if pd.notna(op) and pd.notna(bp) and op > 0 and bp > 0:
                wp_contract = (float(ow) / float(op)) < (float(bw) / float(bp))
            out[hn] = {'v90': v90, 'logit_vel': logit_vel, 'wp_contract': wp_contract}
    except Exception as e:
        print(f"odds velocity failed: {e}")
    return out


_form_cache = None


def _henery_rank(p: np.ndarray, r: int, gamma: float = HENERY_GAMMA) -> np.ndarray:
    """Exact Henery order-statistic marginal P(rank = r) per runner (r = 1..4).

    Conditional place probabilities are discounted by gamma:
      P(2nd=j | 1st=i)   = p_j^g / sum_{k!=i} p_k^g
      P(3rd=m | 1st=i,2nd=j) = p_m^g / sum_{k!=i,j} p_k^g
    (Harville = gamma 1.0; Henery gamma 0.81 corrects longshot place bias.)
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < r:
        return np.full(n, np.nan)
    if r == 1:
        return p.copy()
    g = np.power(np.clip(p, 1e-12, 1.0), gamma)
    res = np.zeros(n)
    if r == 2:
        for j in range(n):
            s = 0.0
            for i in range(n):
                if i != j:
                    s_i = g[i]
                    den = 0.0
                    for k in range(n):
                        if k != i:
                            den += g[k]
                    if den > 1e-12:
                        s += p[i] * (g[j] / den)
            res[j] = s
    elif r == 3:
        for k in range(n):
            s = 0.0
            for i in range(n):
                if i == k:
                    continue
                s_i = 0.0
                for jj in range(n):
                    if jj != i:
                        s_i += g[jj]
                if s_i <= 1e-12:
                    continue
                for j in range(n):
                    if j == i or j == k:
                        continue
                    s_ij = 0.0
                    for m in range(n):
                        if m != i and m != j:
                            s_ij += g[m]
                    if s_ij <= 1e-12:
                        continue
                    s += p[i] * (g[j] / s_i) * (g[k] / s_ij)
            res[k] = s
    elif r == 4:
        for l in range(n):
            s = 0.0
            for i in range(n):
                if i == l:
                    continue
                s_i = 0.0
                for jj in range(n):
                    if jj != i:
                        s_i += g[jj]
                if s_i <= 1e-12:
                    continue
                for j in range(n):
                    if j == i or j == l:
                        continue
                    s_ij = 0.0
                    for m in range(n):
                        if m != i and m != j:
                            s_ij += g[m]
                    if s_ij <= 1e-12:
                        continue
                    for k in range(n):
                        if k == i or k == j or k == l:
                            continue
                        s_ijk = 0.0
                        for m in range(n):
                            if m != i and m != j and m != k:
                                s_ijk += g[m]
                        if s_ijk <= 1e-12:
                            continue
                        s += p[i] * (g[j] / s_i) * (g[k] / s_ij) * (g[l] / s_ijk)
            res[l] = s
    return res


def rank_order_probs(p: np.ndarray, gamma: Optional[float] = None) -> dict:
    """Exact Henery rank-order marginals for a calibrated win vector.

    gamma (order-statistic discount) is chosen DYNAMICALLY from field size in
    [0.75, 0.88] when not supplied explicitly (henery_gamma_for_field).

    Returns dict of (n,) arrays (NaN where undefined, e.g. n < rank):
      p1, p2, p3, p4        = P(finish exactly 1st/2nd/3rd/4th)
      top2, top3, top4      = P(finish in the top 2/3/4)
    The field-size rule for the PLACE pool is applied by the caller:
      n >= 7  -> place pays TOP 3, 4-6 -> TOP 2, n < 4 -> place pool closed.
    """
    p = np.asarray(p, dtype=float)
    if gamma is None:
        gamma = henery_gamma_for_field(len(p))
    finite = np.isfinite(p) & (p > 0)
    pv = np.where(finite, p, 0.0)
    if pv.sum() > 0:
        pv = pv / pv.sum()
    n = len(p)
    p1 = _henery_rank(pv, 1)
    p2 = _henery_rank(pv, 2) if n >= 2 else np.full(n, np.nan)
    p3 = _henery_rank(pv, 3) if n >= 3 else np.full(n, np.nan)
    p4 = _henery_rank(pv, 4) if n >= 4 else np.full(n, np.nan)
    top2 = p1 + p2 if n >= 2 else np.full(n, np.nan)
    top3 = top2 + p3 if n >= 3 else np.full(n, np.nan)
    top4 = top3 + p4 if n >= 4 else np.full(n, np.nan)
    return {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
            'top2': top2, 'top3': top3, 'top4': top4}


def plackett_luce_sim(p: np.ndarray, n_iter: int = 5000, seed: int = 20260906) -> dict:
    """Method B: Monte-Carlo Plackett-Luce order statistics.

    score_i = -log(U_i) / p_i (exponential clocks); argsort = finishing order.
    Empirical marginals match the exact Harville values; kept as a cross-check.
    """
    p = np.asarray(p, dtype=float)
    pv = np.where(np.isfinite(p) & (p > 0), p, 0.0)
    if pv.sum() > 0:
        pv = pv / pv.sum()
    rng = np.random.default_rng(seed)
    u = rng.random((n_iter, len(pv)))
    scores = -np.log(np.clip(u, 1e-12, 1.0)) / np.maximum(pv, 1e-12)
    order = np.argsort(scores, axis=1)
    n = len(pv)

    def marg(k):
        m = np.zeros(n)
        for h in range(n):
            m[h] = np.mean((order[:, :k] == h).any(axis=1))
        return m

    return {'top1': marg(1), 'top2': marg(2), 'top3': marg(3), 'top4': marg(4)}



def _last3_form() -> dict:
    """horse_name (upper) -> '1-2-4' from the last 3 runs in the feature store."""
    global _form_cache
    if _form_cache is None:
        fm = {}
        try:
            f = pd.read_csv('data/model_features.csv',
                            usecols=['race_date', 'horse_name', 'finish_position'],
                            low_memory=False)
            f['_key'] = f['horse_name'].astype(str).str.upper()
            f['race_date'] = pd.to_datetime(f['race_date'], errors='coerce')
            f['finish_position'] = pd.to_numeric(f['finish_position'], errors='coerce')
            f = f.dropna(subset=['race_date', 'finish_position']).sort_values('race_date')
            for k, g in f.groupby('_key'):
                vals = [int(v) for v in g['finish_position'].tail(3)
                        if 1 <= v <= 20]
                if vals:
                    fm[k] = '-'.join(str(v) for v in vals)
        except Exception as e:
            print(f"last3 form load failed: {e}")
            fm = {}
        _form_cache = fm
    return _form_cache


def score_race(live_df, date_str: str, venue: str, race_no: int):
    """LGBM probs -> rigorous smart money -> Bayesian live prob -> EV + flags.

    Causality & sentries:
      - Withdrawn / unposted runners (invalid win odds) are EXCISED before
        softmax, smart-money and Kelly computations (shown as ⏳ pre-open).
      - TRUE baseline = per horse, the FIRST VALID POST-OPEN snapshot odds.
    """
    if live_df is None or len(live_df) == 0:
        return None

    all_df = get_data()
    race_id = f"{date_str}_Race{race_no}"
    merged = live_df.copy()
    if all_df is not None and len(all_df) > 0:
        pred_df = all_df[all_df['race_id'] == race_id]
        if len(pred_df) > 0:
            merged = merge_live_odds_with_predictions(merged, pred_df)
    if 'true_prob' not in merged.columns:
        merged['true_prob'] = np.nan
    # Last-3 form string ('1-2-4') from the feature store (real data, lazy load)
    try:
        merged['last3_form'] = (merged['horse_name'].astype(str).str.upper()
                                .map(_last3_form()))
    except Exception:
        merged['last3_form'] = np.nan

    # --- TRUE baseline: first valid post-open snapshot per horse ---
    # Market-close sentinel: nominal post times run early (a 13:00 card often
    # starts 13:01+), so 'closed' = post time passed AND quotes frozen for
    # CLOSED_FROZEN_SEC. A still-ticking market stays OPEN to keep the last
    # smart-money surges visible (flagged after `sub` is built below).
    try:
        _post = race_post_time(date_str, venue, race_no)
    except Exception:
        _post = None
    store = snap_store(date_str, venue)
    sub = pd.DataFrame()
    exec_state = execution_state(_post)
    baseline, polls_n, open_age_min = None, 0, None
    recent_base = None
    prev_w_map, prev_p_map = {}, {}
    if len(store):
        sub = store[store['race_id'] == race_id].copy()
        sub['epoch'] = pd.to_numeric(sub['epoch'], errors='coerce')
        if len(sub):
            polls_n = int(sub['horse_number'].notna().sum())
            baseline = _first_valid_odds_map(sub)
            if len(baseline):
                try:
                    open_age_min = (time.time() - float(sub['epoch'].min())) / 60.0
                except Exception:
                    open_age_min = None
            # rolling baseline: snapshot closest to (now - window); falls back to
            # the earliest epoch when the store is younger than the window.
            eras = sorted(sub['epoch'].dropna().unique())
            cutoff = time.time() - RECENT_FLOW_WINDOW_S
            eras_in = [e for e in eras if e >= cutoff]
            anchor = eras_in[0] if eras_in else (eras[0] if eras else None)
            if anchor is not None:
                rows = []
                fr = sub[sub['epoch'] == anchor]
                for _, r in fr.iterrows():
                    try:
                        hn = int(r['horse_number'])
                    except (TypeError, ValueError):
                        continue
                    if pd.notna(r.get('win_odds')):
                        rows.append({'horse_number': hn,
                                     'horse_name': r.get('horse_name', ''),
                                     'win_odds': float(r['win_odds'])})
                recent_base = pd.DataFrame(rows)
            # tick-delta sources: the PREVIOUS poll (second-latest epoch) so the
            # terminal can flash on every tick: delta = O_t - O_{t-1}
            if len(eras) >= 2:
                prev_frame = sub[sub['epoch'] == eras[-1 - 1]]
                for _, r in prev_frame.iterrows():
                    try:
                        hn = int(r['horse_number'])
                    except (TypeError, ValueError):
                        continue
                    w = r.get('win_odds')
                    p = r.get('place_odds')
                    if pd.notna(w):
                        prev_w_map[hn] = float(w)
                    if pd.notna(p):
                        prev_p_map[hn] = float(p)
            # CLOSED only when post time passed AND the pool stopped ticking
            # (4-state lifecycle; a still-ticking market stays LOADING_DELAY).
            exec_state = execution_state(_post, sub)
    race_closed = (exec_state == "OFFICIAL_CLOSED")
    t_tilde = t_tilde_seconds(_post)
    w_fuse = fusion_weight(_post, exec_state)
    # audit: print once per execution-state transition (TURBO / LOADING / CLOSED)
    if exec_state in ('TURBO_APPROACH', 'LOADING_DELAY', 'OFFICIAL_CLOSED') \
            and _engine_state_prev.get(race_id) != exec_state:
        print(f"[engine-audit] {race_id} state -> {exec_state}  "
              f"w={w_fuse:.3f}  T~={round(t_tilde, 0) if t_tilde is not None else None}s")
        _engine_state_prev[race_id] = exec_state

    # --- Excise invalid runners BEFORE softmax / smart money / Kelly ---
    valid = _valid_odds_mask(merged['win_odds'], merged.get('place_odds'))
    work = merged[valid].copy()
    out = merged.copy()
    out['race_closed'] = race_closed

    if len(work) == 0:
        out['prob'] = np.nan
        out['win_odds'] = pd.to_numeric(out['win_odds'], errors='coerce')
        out['ev'] = np.nan
        out['kelly'] = 0.0
        out['smart_money_score'] = SMART_MONEY_CENTER
        out['flow_signal'] = '➖ STABLE'
        out['unposted'] = True
        out['odds_move_pct'] = np.nan
        out['prev_win_odds'] = np.nan
        out['prev_place_odds'] = np.nan
        out['tick_delta'] = np.nan
        out['tick_delta_place'] = np.nan
        for c in ['p_rank1', 'p_rank2', 'p_rank3', 'p_rank4',
                  'p_top2', 'p_top3', 'p_top4', 'place_prob', 'place_ev',
                  'pace_scenario', 'pace_n_leaders',
                  'fused_weight', 'delta_flow', 't_tilde',
                  'syndicate_steam', 'divergence_trap',
                  'smart_place_absorption']:
            out[c] = np.nan
        out['exec_state'] = exec_state
        out['polls_n'] = polls_n
        out['open_age_min'] = open_age_min
        return out

    if 'true_prob' not in work.columns or work['true_prob'].isna().all():
        if all_df is not None and len(all_df) > 0:
            work['true_prob'] = _field_model_probs(work, all_df)
        else:
            work['true_prob'] = np.nan

    # --- Pace scenario matrix (lagged run-style density; penalise/boost logits) ---
    pace_scn, pace_n = 'NORMAL', 0
    try:
        work['true_prob'], (pace_scn, pace_n) = _apply_pace_adjustment(
            work['true_prob'], work)
    except Exception as e:
        print(f"pace adjustment failed: {e}")

    if baseline is not None and len(baseline):
        scored = calculate_smart_money_metrics(work, baseline)
        # rolling-window overlay: keep the stronger real signal so mid-session
        # surges trigger (choose the score with the larger |S - 50|).
        if recent_base is not None and len(recent_base) and len(recent_base) >= 2:
            try:
                scored_recent = calculate_smart_money_metrics(work, recent_base)
                s_tot = scored['smart_money_score'].to_numpy(dtype=float)
                s_rec = scored_recent['smart_money_score'].to_numpy(dtype=float)
                use_rec = np.abs(s_rec - SMART_MONEY_CENTER) > np.abs(s_tot - SMART_MONEY_CENTER)
                scored['smart_money_score'] = np.where(use_rec, s_rec, s_tot)
                scored['odds_prob_delta'] = np.where(
                    use_rec,
                    scored_recent['odds_prob_delta'].to_numpy(dtype=float),
                    scored['odds_prob_delta'].to_numpy(dtype=float))
                scored['flow_signal'] = [get_flow_signal(x) for x in scored['smart_money_score']]
            except Exception as e:
                print(f"recent-flow overlay failed: {e}")
        scored.index = work.index          # merge() resets index - restore so the
        work['smart_money_score'] = scored['smart_money_score'].to_numpy(dtype=float)
        work['flow_signal'] = scored['flow_signal'].astype(object).to_numpy(dtype=object)
    else:
        work['smart_money_score'] = SMART_MONEY_CENTER
        work['flow_signal'] = '➖ STABLE'

    # --- Time-varying Bayesian fusion engine ---------------------------
    # logit(P_final) = (1-w) logit(P_model) + w logit(P_market) + delta_flow
    p_model = pd.to_numeric(work.get('true_prob'), errors='coerce')
    p_mkt = de_vig_market_probs(work['win_odds'])
    vel = _odds_velocity(sub) if len(sub) else {}
    work['delta_flow'] = 0.0
    work['syndicate_steam'] = False
    work['divergence_trap'] = False
    delta_flow = np.zeros(len(work))
    for pos, r in work.iterrows():
        try:
            hn = int(r['horse_number'])
        except (TypeError, ValueError):
            continue
        v = vel.get(hn)
        if not v:
            continue
        # logit(odds) velocity: scale-free microstructure (Master Rule 1)
        lv = v.get('logit_vel')
        lv = float(lv) if lv is not None else float(v.get('v90'))
        if lv <= FLOW_STEAM_LOGIT and bool(v.get('wp_contract')):
            base_ref = float(v.get('v90')) if v.get('v90') is not None else lv
            strength = min(abs(base_ref) / abs(FLOW_STEAM_V), 1.0)
            delta_flow[work.index.get_loc(pos)] = (FLOW_DELTA_MIN
                + (FLOW_DELTA_MAX - FLOW_DELTA_MIN) * strength)
            work.at[pos, 'syndicate_steam'] = True
    pm = p_model.to_numpy(dtype=float)
    pk = p_mkt.to_numpy(dtype=float)
    if t_tilde is not None and t_tilde <= TRAP_T_TILDE_S:
        trap = (pm > TRAP_MODEL_MIN) & np.isfinite(pk) & (pk < TRAP_MARKET_MAX)
        work['divergence_trap'] = trap
        delta_flow = np.where(trap, delta_flow + TRAP_LOGIT_PENALTY, delta_flow)
    fused = bayesian_fusion(pm, pk, w_fuse, delta_flow)
    work['delta_flow'] = delta_flow
    # audit: print once per distinct syndicate-steam episode (logit-velocity)
    try:
        if (work['syndicate_steam']).any():
            steamers = tuple(sorted(work.loc[work['syndicate_steam'], 'horse_name']
                                    .astype(str).tolist()))
            key = (race_id, steamers)
            if key not in _steam_seen:
                print(f"[engine-audit] STEAM {race_id}: {list(steamers)}  "
                      f"logit_vel<={FLOW_STEAM_LOGIT:.3f} (≤-15% price / 90s)")
                _steam_seen.add(key)
    except Exception:
        pass
    work['live_prob'] = fused
    work['fused_weight'] = w_fuse
    work['exec_state'] = exec_state
    work['t_tilde'] = t_tilde if t_tilde is not None else np.nan
    odds_w = pd.to_numeric(work['win_odds'], errors='coerce')
    decay_w = odds_w.apply(_decay)
    work['live_expected_value'] = fused * odds_w * decay_w - 1.0
    prob_col, ev_col = 'live_prob', 'live_expected_value'

    # --- Place (top 2/3) / rank-order marginals (Harville / Plackett-Luce) ---
    # Field-size rule: >=7 runners -> place pays TOP 3; 4-6 -> TOP 2; <4 -> closed.
    pcol = prob_col if prob_col in work.columns else 'true_prob'
    probs = pd.to_numeric(work.get(pcol), errors='coerce')
    work['p_rank1'] = np.nan
    work['p_rank2'] = np.nan
    work['p_rank3'] = np.nan
    work['p_rank4'] = np.nan
    work['p_top2'] = np.nan
    work['p_top3'] = np.nan
    work['p_top4'] = np.nan
    work['place_prob'] = np.nan
    work['place_ev'] = np.nan
    work['pace_scenario'] = pace_scn
    work['pace_n_leaders'] = pace_n
    work['smart_place_absorption'] = False
    if probs.notna().sum() >= 2:
        pm = rank_order_probs(probs.to_numpy(dtype=float),
                              gamma=henery_gamma_for_field(len(work)))
        for c, key in [('p_rank1', 'p1'), ('p_rank2', 'p2'), ('p_rank3', 'p3'),
                       ('p_rank4', 'p4'), ('p_top2', 'top2'), ('p_top3', 'top3'),
                       ('p_top4', 'top4')]:
            work[c] = pm[key]
        n_r = len(work)
        if n_r >= 7:
            work['place_prob'] = pm['top3']
        elif n_r >= 4:
            work['place_prob'] = pm['top2']
        po = pd.to_numeric(work['place_odds'], errors='coerce')
        work['place_ev'] = work['place_prob'] * po     # EV_place ratio
        # Win/Place pool-ratio anomaly: heavy place-pool hedging vs the field
        work['smart_place_absorption'] = smart_place_absorption_mask(
            probs.to_numpy(dtype=float), po.to_numpy(dtype=float))

    # Merge scored rows back; invalid rows carry no model info.
    # NOTE: create the target column as OBJECT dtype first - assigning a string
    # ArrowStringArray (flow_signal) over a float64 NaN column raises TypeError
    # and killed the whole card after the focus race.
    for col in ['true_prob', 'live_prob', 'live_expected_value',
                'smart_money_score', 'flow_signal',
                'p_rank1', 'p_rank2', 'p_rank3', 'p_rank4',
                'p_top2', 'p_top3', 'p_top4', 'place_prob', 'place_ev',
                'pace_scenario', 'pace_n_leaders',
                'fused_weight', 'delta_flow', 't_tilde',
                'syndicate_steam', 'divergence_trap',
                'smart_place_absorption', 'exec_state']:
        if col in work.columns:
            out[col] = np.full(len(out), np.nan, dtype=object)
            out.loc[work.index, col] = work[col].astype(object).to_numpy(dtype=object)

    out['prob'] = pd.to_numeric(out.get(prob_col), errors='coerce')
    odds = pd.to_numeric(out['win_odds'], errors='coerce')
    place = pd.to_numeric(out.get('place_odds'), errors='coerce')
    unposted = ~_valid_odds_mask(odds, place if place is not None else None)
    out['unposted'] = unposted
    out['win_odds'] = odds.where(~unposted)
    odds = out['win_odds']
    decay = odds.apply(_decay)

    if ev_col is not None and ev_col in out.columns:
        out['ev'] = pd.to_numeric(out[ev_col], errors='coerce')
        out['ev'] = out['ev'].where(odds.notna())
    else:
        out['ev'] = out['prob'] * odds * decay - 1.0
    out['kelly'] = np.where(
        odds > 1.0,
        np.clip((out['prob'] * odds * decay - 1.0) / (odds - 1.0) * KELLY_FRACTION, 0.0, KELLY_MAX_BET),
        0.0)
    out['kelly'] = out['kelly'].where(odds.notna(), 0.0)
    out['decay'] = decay

    open_map = {}
    if baseline is not None:
        for _, r in baseline.iterrows():
            try:
                open_map[int(r['horse_number'])] = float(r['win_odds'])
            except (TypeError, ValueError):
                pass
    out['open_odds'] = out['horse_number'].map(open_map)
    out['odds_move_pct'] = np.where(
        out['open_odds'].notna() & (out['open_odds'] > 0) & odds.notna(),
        (odds - out['open_odds']) / out['open_odds'] * 100.0, np.nan)
    prev = out['horse_number'].map(prev_w_map)
    out['prev_win_odds'] = prev
    out['tick_delta'] = np.where(prev.notna() & odds.notna(), odds - prev, np.nan)
    prev_p = out['horse_number'].map(prev_p_map)
    out['prev_place_odds'] = prev_p
    out['tick_delta_place'] = np.where(prev_p.notna() & place.notna(), place - prev_p, np.nan)
    out['polls_n'] = polls_n
    out['open_age_min'] = open_age_min
    return out


def flag_row(row) -> str:
    """Dual-mode betting guardrails (Law 2 - no longshot value traps).

    WIN requires  P_final >= 0.16 AND EV >= 1.15 AND O_W in [2.2, 14.0].
    EV >= 1.25 with O_W > 14.0 is suppressed as a WIN and routed to the
    exotics channel (PLACE / QP / TIERCE anchor) instead.
    """
    def f(v):
        try:
            x = float(v)
            return x if not np.isnan(x) else None
        except (TypeError, ValueError):
            return None

    prob = f(row.get('prob'))
    odds = f(row.get('win_odds'))
    ev = f(row.get('ev'))
    smart = f(row.get('smart_money_score')) or SMART_MONEY_CENTER
    if prob is None or odds is None or ev is None:
        return '—'
    ev_ratio = ev + 1.0
    if (prob >= WIN_MIN_PROB and ev_ratio >= WIN_MIN_EV
            and WIN_MIN_ODDS <= odds <= WIN_MAX_ODDS and smart >= 50.0):
        return '🎯 PRIME W'
    if ev_ratio >= EXOTIC_EV and odds > WIN_MAX_ODDS:
        return '🎯 EXOTIC'
    if ev > 0.0 and smart < 40.0:
        return '⚠️ DRIFT'
    return '—'

"""
course_rail.py — Course & Rail-Setting Draw Bias (Master Rules Task E)
=======================================================================
Parameterizes barrier-draw multipliers by venue (HV / ST) and, when supplied,
rail setting (Course A / B / C / C+3), grounded in REAL historical stats from
the feature store (data/model_features.csv).

Two layers:
  1. Empirical venue bias  — per (venue, draw) smoothed WIN/PLACE rate from real
     finished runs in the feature store, expressed as a multiplier relative to
     the venue's average draw win rate. This is data, not opinion.
  2. Rail modulation      — a parameterized per-venue table of draw multipliers
     per rail setting. Because no historic rail column exists in the store yet,
     the default tables are NEUTRAL (1.0) and only activate when a rail is
     actually known for the meeting (data/rail_settings.csv or rail_for()).
     Once per-rail calibration stats exist, fill RAIL_DRAW_MULT below.

Usage in web_live._apply_pace_adjustment:
    from modeling.course_rail import logit_scale
    adj[i] *= logit_scale(venue, rail, draw[i], run_style[i], distance)
(always within a safe clamp so a rule never flips sign or blows up).
"""
import os
import threading

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(HERE, 'data', 'model_features.csv')
RAILS_PATH = os.path.join(HERE, 'data', 'rail_settings.csv')

# Venue names as stored in the feature store (track column)
_VENUE_ALIAS = {'HV': 'HV', 'ST': 'ST', 'HAPPY VALLEY': 'HV', 'SHA TIN': 'ST'}

# --- Parameter tables (CALIBRATE with per-rail historical stats when available).
# Structure: RAIL_DRAW_MULT[venue][rail][draw] -> factor. Identity (all 1.0)
# until calibrated; logit_scale() falls back to the empirical venue bias only.
RAIL_DRAW_MULT = {
    'HV': {rail: {d: 1.0 for d in range(1, 13)} for rail in ('A', 'B', 'C', 'C+3')},
    'ST': {rail: {d: 1.0 for d in range(1, 15)} for rail in ('A', 'B', 'C', 'C+3')},
}

# Safe bounds for the combined multiplier (never flips a pace adjustment).
LOGIT_SCALE_MIN = 0.5
LOGIT_SCALE_MAX = 2.0
# Blend of empirical venue bias when rail data is absent (0 -> pure rule base).
# 0.5 = half-weight towards the venue's observed draw profile; conservative so
# an unvalidated layer cannot override the Master-Rules pace logits.
EMPIRICAL_BLEND = 0.5

_lock = threading.Lock()
_cache = None            # dict: (venue, distance) -> DataFrame indexed by draw


def _venue_key(venue) -> str:
    return _VENUE_ALIAS.get(str(venue).strip().upper(),
                            str(venue).strip().upper())


def load_venue_draw_stats(venue: str, force: bool = False) -> pd.DataFrame:
    """Smoothed per-draw WIN/PLACE rate for a venue from the feature store.

    Returns a DataFrame indexed by barrier_draw with columns
      n, win_rate, place_rate, mult_win, mult_place
    (multiplier = rate / mean over draws; NaN when the venue has < 60 samples).
    """
    global _cache
    vk = _venue_key(venue)
    with _lock:
        if _cache is None or force:
            d = {}
            try:
                f = pd.read_csv(FEATURES_PATH,
                                usecols=['race_date', 'track', 'distance',
                                         'barrier_draw', 'finish_position'],
                                low_memory=False)
                f['_venue'] = f['track'].astype(str).str.strip().str.upper().map(_VENUE_ALIAS)
                f = f.dropna(subset=['_venue', 'barrier_draw', 'finish_position'])
                f['barrier_draw'] = f['barrier_draw'].astype(int)
                f['finish_position'] = pd.to_numeric(f['finish_position'],
                                                     errors='coerce')
                f = f[f['finish_position'] >= 1]
                f['_placed'] = (f['finish_position'] <= 3).astype(float)
                f['_won'] = (f['finish_position'] == 1).astype(float)
                for v, g in f.groupby('_venue'):
                    for dist, gg in g.groupby('distance'):
                        tbl = gg.groupby('barrier_draw').agg(
                            n=('_won', 'size'),
                            win_rate=('_won', 'mean'),
                            place_rate=('_placed', 'mean')).reset_index()
                        tbl = tbl.set_index('barrier_draw')
                        d[(v, dist)] = tbl
            except Exception as e:  # never crash the live engine
                print(f"course_rail stats load failed: {e}")
                d = {}
            _cache = d
        tbl = _cache.get((vk, None)) or pd.DataFrame()
        if tbl.empty:
            # aggregate across distances when a venue has no distance table yet
            allrows = []
            for (v, _dist), t in _cache.items():
                if v == vk:
                    t2 = t.copy()
                    allrows.append(t2)
            if allrows:
                tbl = pd.concat(allrows).groupby(level=0).sum()
        if not len(tbl):
            return pd.DataFrame()
        total_n = int(tbl['n'].sum())
        if total_n < 60:
            return pd.DataFrame()
        avg_w = float((tbl['win_rate'] * tbl['n']).sum()) / total_n
        avg_p = float((tbl['place_rate'] * tbl['n']).sum()) / total_n
        out = tbl.copy()
        out['mult_win'] = out['win_rate'] / avg_w if avg_w > 0 else 1.0
        out['mult_place'] = out['place_rate'] / avg_p if avg_p > 0 else 1.0
        return out


def rail_for(date_str: str, venue: str):
    """Rail setting for a meeting from data/rail_settings.csv (if maintained).

    CSV columns: date (YYYY-MM-DD), venue (HV/ST), rail (A/B/C/C+3). Returns
    None when unknown -> neutral rail modulation."""
    try:
        if not os.path.exists(RAILS_PATH):
            return None
        r = pd.read_csv(RAILS_PATH, dtype=str)
        vk = _venue_key(venue)
        m = r[(r['venue'].str.strip().str.upper().map(_VENUE_ALIAS) == vk)
              & (r['date'].str.strip() == str(date_str))]
        if len(m):
            return str(m.iloc[0]['rail']).strip().upper()
    except Exception:
        return None
    return None


def logit_scale(venue: str, rail, draw, distance=None, kind='win') -> float:
    """Combined draw-multiplier for a pace-rule logit adjustment.

    multiplier = rail_factor(draw) x empirical_venue_factor(draw), clipped to
    [LOGIT_SCALE_MIN, LOGIT_SCALE_MAX] so adjustments never flip sign/explode.
    Neutral (1.0) whenever there is no data or no rail calibration.
    """
    vk = _venue_key(venue)
    try:
        d = int(round(float(draw)))
    except (TypeError, ValueError):
        return 1.0
    rail_factor = 1.0
    if rail:
        rs = str(rail).strip().upper()
        tbl = RAIL_DRAW_MULT.get(vk, {}).get(rs)
        if tbl and d in tbl:
            rail_factor = float(tbl[d])
    emp = 1.0
    try:
        stats = load_venue_draw_stats(vk)
        if len(stats) and d in stats.index:
            col = 'mult_win' if kind == 'win' else 'mult_place'
            v = float(stats.loc[d, col])
            if np.isfinite(v):
                emp = v
    except Exception:
        emp = 1.0
    mult = rail_factor * (1.0 + EMPIRICAL_BLEND * (emp - 1.0))
    return float(np.clip(mult, LOGIT_SCALE_MIN, LOGIT_SCALE_MAX))

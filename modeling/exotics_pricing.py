"""
HKJC Exotics Pricing Engine — Harville / Luce Model
====================================================
Derives multi-horse outcome probabilities (Place / Quinella / Quinella Place)
from single-horse win probabilities, scans live pools for overlays, and settles
tickets against REAL scraped dividends.

⚠️ EDUCATIONAL USE ONLY — statistical modeling study, not gambling advice.
"""

import itertools
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# HKJC pool takeout approximations (fraction of pool removed before dividends)
WIN_TAKEOUT = 0.175
PLACE_TAKEOUT = 0.175
QIN_TAKEOUT = 0.25   # Quinella
QPL_TAKEOUT = 0.25   # Quinella Place

DEFAULT_PLACES = 3

# Henery order-statistic discount: bounded [0.75, 0.88] and chosen DYNAMICALLY
# from field size (small fields -> lower gamma, longshot place bias is sharper).
HENERY_GAMMA_MIN = 0.75
HENERY_GAMMA_MAX = 0.88


def henery_gamma_for_field(field_size: int) -> float:
    """Field-size-calibrated Henery gamma in [0.75, 0.88].

    Larger fields need a higher discount because the rank-2/3/4 denominators
    pool more longshot mass; small fields (4-6 runners) sit at the 0.75 floor.
    """
    n = max(4, int(field_size))
    frac = min(max((n - 4) / (14 - 4), 0.0), 1.0)
    return HENERY_GAMMA_MIN + frac * (HENERY_GAMMA_MAX - HENERY_GAMMA_MIN)


def smart_place_absorption_mask(win_probs, place_odds) -> np.ndarray:
    """Flag runners whose normalized implied PLACE probability significantly
    exceeds their de-vigged WIN probability vs the field median ratio
    (>= 1.5x median) - i.e. smart money hedging the place pool.
    """
    wp = np.asarray(win_probs, dtype=float)
    po = np.asarray(place_odds, dtype=float)
    n = len(wp)
    mask = np.zeros(n, dtype=bool)
    ok = np.isfinite(wp) & (wp > 0) & np.isfinite(po) & (po > 1.0)
    if int(ok.sum()) < 2:
        return mask
    place_impl = np.where(ok, 1.0 / np.where(po > 0, po, np.nan), np.nan)
    place_norm = place_impl / np.nansum(place_impl)
    win_norm = np.where(ok, wp, np.nan)
    win_norm = win_norm / np.nansum(win_norm)
    ratio = np.where(ok & (win_norm > 0), place_norm / win_norm, np.nan)
    med = float(np.nanmedian(ratio))
    if np.isfinite(med) and med > 0:
        mask = ok & (ratio >= 1.5 * med)
    return mask


def _normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-9, 1.0)
    total = probs.sum()
    return probs / total if total > 0 else probs


def harville_place_probs(win_probs, n_places: int = DEFAULT_PLACES) -> np.ndarray:
    """P(Horse i finishes in top `n_places`) under the Harville model.

    P(top 3) = p_i
             + sum_{j!=i} p_j * p_i / (1 - p_j)
             + sum_{j!=i} sum_{k!=i,j} p_j * p_k/(1-p_j) * p_i/(1-p_j-p_k)
    """
    p = _normalize(win_probs)
    n = len(p)
    place = np.zeros(n)
    for i in range(n):
        total = p[i]
        # second place for i
        for j in range(n):
            if j == i:
                continue
            total += p[j] * p[i] / (1.0 - p[j])
        # third place for i
        if n_places >= 3:
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    total += p[j] * (p[k] / (1.0 - p[j])) * (p[i] / (1.0 - p[j] - p[k]))
        place[i] = total
    return place


def quinella_prob(p_i: float, p_j: float) -> float:
    """P(exact quinella (i,j) in either order):
    P = p_i * p_j/(1-p_i) + p_j * p_i/(1-p_j)
    """
    return p_i * p_j / (1.0 - p_i) + p_j * p_i / (1.0 - p_j)


def quinella_place_prob(win_probs, i: int, j: int, n_places: int = DEFAULT_PLACES) -> float:
    """P(both horses i and j finish inside the top `n_places`).

    Enumerates every top-3 ordering containing both horses and sums the
    Harville chain probabilities p[x1] * p[x2]/(1-p[x1]) * p[x3]/(1-p[x1]-p[x2]).
    """
    p = _normalize(win_probs)
    n = len(p)
    if n_places != 3:
        raise NotImplementedError("quinella_place_prob currently supports top-3 only")
    others = [k for k in range(n) if k != i and k != j]
    total = 0.0
    for third in others:
        for a, b, c in ((i, j, third), (j, i, third), (i, third, j),
                        (j, third, i), (third, i, j), (third, j, i)):
            total += p[a] * (p[b] / (1.0 - p[a])) * (p[c] / (1.0 - p[a] - p[b]))
    return total


def exotic_prob_matrix(win_probs):
    """Returns (place_probs, quinella_matrix, qp_matrix) for a race."""
    p = _normalize(win_probs)
    n = len(p)
    place = harville_place_probs(p)
    q = np.zeros((n, n))
    qp = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        v = quinella_prob(p[i], p[j])
        q[i, j] = q[j, i] = v
        w = quinella_place_prob(p, i, j)
        qp[i, j] = qp[j, i] = w
    return place, q, qp


def market_win_probs(win_odds):
    """Market-implied win probabilities normalized to 1 (strips overround)."""
    imp = 1.0 / np.asarray(win_odds, dtype=float)
    return _normalize(imp)


def place_dividend_estimate(win_odds, takeout: float = PLACE_TAKEOUT, n_places: int = DEFAULT_PLACES):
    """Approximates the $10 place dividend from win odds.

    Uses Harville to move from market win probabilities to place probabilities
    and applies the place-pool takeout. Returns dividend per $1 staked
    (HKJC displays dividends per $10, hence a factor of 10 is NOT applied here;
    callers that want the $10 display multiply by 10).
    """
    p = market_win_probs(win_odds)
    place = harville_place_probs(p, n_places)
    pay = (1.0 - takeout) / np.clip(place, 1e-9, None)
    return pay  # profit multiplier per $1 (includes stake return)


def scan_exotic_overlays(race_df, core_odds_band=(4.5, 8.0), intent_col='trainer_urgency_index',
                         qp_min_edge=0.15, takeout_qpl=QPL_TAKEOUT, takeout_place=PLACE_TAKEOUT):
    """Flags high-edge Place / QPL combinations for one race.

    race_df must contain: win_odds, true_prob (model win probs), horse_name,
    and optionally the intent column (e.g. trainer_urgency_index or
    jockey_rode_trackwork_count_14d).

    A "core horse" is a solid model pick within the 4.5-8.0 odds band;
    an "intent horse" is the highest-scoring horse on the intent feature.
    The scanner returns combos whose model QP probability beats the market
    QP probability by `qp_min_edge` (relative), plus the top Place overlays.
    """
    df = race_df.reset_index(drop=True).copy()
    df = df[df['win_odds'].notna() & df['true_prob'].notna()]
    if len(df) < 4:
        return pd.DataFrame()

    p_model = _normalize(df['true_prob'].values)
    p_market = market_win_probs(df['win_odds'].values)
    n = len(df)

    place_model = harville_place_probs(p_model)
    place_market = harville_place_probs(p_market)
    place_pay = (1.0 - takeout_place) / np.clip(place_market, 1e-9, None)

    df['model_place_prob'] = place_model
    df['market_place_prob'] = place_market
    df['est_place_div'] = place_pay
    df['place_edge'] = place_model - place_market

    # Core horses: solid model probability within the sweet-spot odds band
    core_mask = df['win_odds'].between(*core_odds_band)
    core_idx = df.index[core_mask].tolist()

    # Intent horse: highest score on the intent feature (fallback: highest place edge)
    if intent_col in df.columns:
        intent_idx = df[intent_col].idxmax()
    else:
        intent_idx = df['place_edge'].idxmax()

    rows = []
    for i in core_idx:
        j = intent_idx
        if i == j:
            continue
        qp_model = quinella_place_prob(p_model, i, j)
        qp_market = quinella_place_prob(p_market, i, j)
        if qp_market <= 0:
            continue
        edge = qp_model / qp_market - 1.0
        if edge >= qp_min_edge:
            rows.append({
                'combo': f"{df.loc[i, 'horse_name']} / {df.loc[j, 'horse_name']}",
                'core_horse': df.loc[i, 'horse_name'],
                'core_odds': df.loc[i, 'win_odds'],
                'intent_horse': df.loc[j, 'horse_name'],
                'intent_score': (df.loc[j, intent_col] if intent_col in df.columns else np.nan),
                'qp_model_prob': qp_model,
                'qp_market_prob': qp_market,
                'qp_edge': edge,
                'est_qpl_div': (1.0 - takeout_qpl) / np.clip(qp_market, 1e-9, None),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Sanity self-test: uniform win probs -> place probs sum to 3, QP pairs sum to 3
    p = np.array([0.25, 0.2, 0.15, 0.1, 0.3])
    place = harville_place_probs(p)
    assert abs(place.sum() - 3.0) < 1e-9, place.sum()
    _, q, qp = exotic_prob_matrix(p)
    assert abs(np.triu(qp, 1).sum() - 3.0) < 1e-9, np.triu(qp, 1).sum()
    q12 = quinella_prob(p[0], p[1])
    assert abs(q12 - q[0, 1]) < 1e-12
    print("Harville self-tests passed.")
    print("place probs:", np.round(place, 4))
    print("QP pair sum (should be 3.0):", np.triu(qp, 1).sum())


# ================================================================
# REAL-DIVIDEND SETTLEMENT ENGINE (Task B)
# ================================================================

def load_dividend_store(path: str = "data/historical_dividends.csv",
                       legacy_path: str = "data/dividends.csv"):
    """Loads REAL scraped dividends into a settlement store.

    Reads the primary historical store and the legacy store (if present) and
    merges them with deduplication. HKJC dividends are per HK$10 -> ticket
    multiplier = dividend / 10.
    Returns (df, place_map, qpl_map, qin_map) where:
      place_map[(race_id, horse_code)] -> multiplier
      qpl_map[(race_id, frozenset({c1, c2}))] -> multiplier
      qin_map[(race_id, frozenset({c1, c2}))] -> multiplier
    Returns (None, {}, {}, {}) when no dividend data exists yet.
    """
    frames = []
    for p in (path, legacy_path):
        if os.path.exists(p):
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                logger.warning("Could not read dividend store %s: %s", p, e)
    if not frames:
        logger.warning("Dividend stores not found (%s / %s) - exotics backtest cannot "
                       "settle against real payouts.", path, legacy_path)
        return None, {}, {}, {}
    df = pd.concat(frames, ignore_index=True)
    if 'combo' in df.columns:
        df = df.drop_duplicates(subset=['race_id', 'pool', 'combo'])
    if len(df) == 0:
        return None, {}, {}, {}
    place_map, qpl_map, qin_map = {}, {}, {}
    for _, r in df.iterrows():
        codes = [c for c in str(r.get('combo_codes', '')).split('/') if c]
        race_id = str(r['race_id'])
        mult = float(r['dividend']) / 10.0
        pool = str(r['pool'])
        if pool == 'PLACE' and len(codes) == 1:
            place_map[(race_id, codes[0])] = mult
        elif pool == 'QPL' and len(codes) == 2:
            qpl_map[(race_id, frozenset(codes))] = mult
        elif pool == 'QUINELLA' and len(codes) == 2:
            qin_map[(race_id, frozenset(codes))] = mult
    return df, place_map, qpl_map, qin_map


def settle_ticket(pool_map, race_id, codes, stake: float = 10.0) -> float:
    """Ticket Return = stake * actual dividend multiplier if the combination hit,
    else 0. Uses REAL scraped dividends only."""
    if isinstance(codes, (list, tuple, set, frozenset)):
        codes = list(codes)
        key = (str(race_id), frozenset(codes)) if len(codes) > 1 else (str(race_id), codes[0])
    else:
        key = (str(race_id), codes)
    mult = pool_map.get(key, 0.0)
    return stake * mult


# ================================================================
# CORE-TO-LONGSHOT OVERLAY STRATEGY (Task C)
# ================================================================

def build_core_satellite_bets(race_df,
                              banker_tier1=(4.0, 8.0, 0.22),   # (odds_lo, odds_hi, min EV-1)
                              banker_tier2=(3.0, 9.0, 0.12),   # secondary tier
                              satellite_min_odds=7.0, max_satellites=2,
                              place_ev_threshold=1.15, takeout_place=PLACE_TAKEOUT,
                              urgency_col='trainer_urgency_index', urgency_threshold=0.40):
    """Structured two-tier Banker (膽) + Satellite (腳) selection for one race.

    Tier 1 Banker: win odds in [4.0, 8.0] AND decay-adjusted EV >= 1.22
    Tier 2 Banker: win odds in [3.0, 9.0] AND decay-adjusted EV >= 1.12
    Satellite:     win odds >= 7.0 AND (trainer_urgency_index >= 0.40
                   OR is_forgive_run == 1 OR trial_won_before_race == 1)

    Returns dict:
      'qpl_pairs': list of (banker_code, satellite_code, banker_tier)
      'pla_bets':  list of horse_codes whose model Place EV (vs market proxy) > threshold
      'bankers':   list of (horse_code, tier)
    """
    df = race_df.reset_index(drop=True).copy()
    df = df[df['win_odds'].notna() & df['true_prob'].notna()]
    if len(df) < 2:
        return {'qpl_pairs': [], 'pla_bets': [], 'bankers': []}

    if 'horse_code' not in df.columns:
        df['horse_code'] = df['horse_name'].astype(str)

    # Decay-adjusted EV (odds-ratio minus 1); recompute if the column is absent
    if 'expected_value' not in df.columns:
        if 'decay_factor' in df.columns:
            df['expected_value'] = df['true_prob'] * df['win_odds'] * df['decay_factor'] - 1.0
        else:
            df['expected_value'] = df['true_prob'] * df['win_odds'] - 1.0

    # --- Two-tier bankers ---
    t1 = df[(df['win_odds'].between(banker_tier1[0], banker_tier1[1]))
            & (df['expected_value'] >= banker_tier1[2])].copy()
    t1['_tier'] = 1
    t2 = df[(df['win_odds'].between(banker_tier2[0], banker_tier2[1]))
            & (df['expected_value'] >= banker_tier2[2])].copy()
    t2['_tier'] = 2
    bankers = pd.concat([t1, t2]).drop_duplicates(subset=['horse_code'])
    bankers = bankers.sort_values(['_tier', 'expected_value'], ascending=[True, False])

    # --- Satellites ---
    if urgency_col in df.columns:
        df['_intent'] = df[urgency_col].fillna(0)
    else:
        df['_intent'] = 0.0
    intent_flags = (df['_intent'] >= urgency_threshold)
    if 'is_forgive_run' in df.columns:
        intent_flags |= (df['is_forgive_run'].fillna(0) == 1)
    if 'trial_won_before_race' in df.columns:
        intent_flags |= (df['trial_won_before_race'].fillna(0) == 1)
    satellites = df[(df['win_odds'] >= satellite_min_odds) & intent_flags]
    satellites = satellites.sort_values('_intent', ascending=False)

    # --- QPL pairs: banker x top-N satellites ---
    qpl_pairs = []
    for _, b in bankers.iterrows():
        taken = 0
        for _, s in satellites.iterrows():
            if taken >= max_satellites:
                break
            if s['horse_code'] == b['horse_code']:
                continue
            qpl_pairs.append((b['horse_code'], s['horse_code'], int(b['_tier'])))
            taken += 1

    # --- PLA candidates: model Place EV vs market-implied proxy dividend ---
    p_model = _normalize(df['true_prob'].values)
    p_market = market_win_probs(df['win_odds'].values)
    place_model = harville_place_probs(p_model)
    place_market = harville_place_probs(p_market)
    proxy_mult = (1.0 - takeout_place) / np.clip(place_market, 1e-9, None)
    place_ev = place_model * proxy_mult
    pla_mask = (place_ev > place_ev_threshold) & (df['win_odds'] >= 3.0)
    pla_bets = df.loc[pla_mask, 'horse_code'].tolist()

    return {'qpl_pairs': qpl_pairs, 'pla_bets': pla_bets,
            'bankers': list(zip(bankers['horse_code'], bankers['_tier']))}


# ================================================================
# MASTER-RULES TASK C — Exotics Overlay Pricing Matrix (Q / QP)
# ================================================================
# Edge = (P_model / P_market) - 1.0 on QUINELLA and QUINELLA-PLACE pairs.
# Pair probabilities use HENERY order statistics (gamma in [0.75, 0.88] by
# field size) applied to BOTH the model and de-vigged market side so the edge
# is apples-to-apples. Filter: edge >= min_edge AND at least one runner has
# Smart Money S >= min_smart. Returns the top-N combinations.


def henery_quinella_matrix(p, gamma=None):
    """P(exact quinella (i,j), either order) for every pair, under Henery
    order statistics: conditional 2nd odds are g_j/(S - g_i), g = p^gamma.

    Vectorised: Q[i,j] = p_i*g_j/(S-g_i) + p_j*g_i/(S-g_j)   (i != j, Q=0 diag)
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < 2:
        return np.zeros((n, n))
    if gamma is None:
        gamma = henery_gamma_for_field(n)
    pv = _normalize(p)
    g = np.clip(pv, 1e-12, 1.0) ** gamma
    S = float(g.sum())
    den = np.maximum(S - g, 1e-12)
    Q = (pv[:, None] * g[None, :] / den[:, None]
         + pv[None, :] * g[:, None] / den[None, :])
    np.fill_diagonal(Q, 0.0)
    return Q


def henery_qp_matrix(p, gamma=None, n_places=3):
    """P(both horses i and j finish inside the top `n_places`) for EVERY pair,
    under Henery order statistics (Master Rules Task C: vectorized pairs).

    n_places in (2, 3): HKJC QP pays the first 3 (first 2 in small fields).
    Implementation is vectorised over runner pairs by enumerating the position
    of the extra horse x for each fixed x (O(n^3) total, exact chain sums):
      both-top2 term + {x 1st / x 2nd / x 3rd} chain contributions.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < 2:
        return np.zeros((n, n))
    if gamma is None:
        gamma = henery_gamma_for_field(n)
    pv = _normalize(p)
    g = np.clip(pv, 1e-12, 1.0) ** gamma
    S = float(g.sum())
    k = int(n_places)
    if k > n:
        k = n
    if k < 2:
        return np.zeros((n, n))
    den_i = np.maximum(S - g, 1e-12)
    # P(both i,j finish in the top 2, either order) = A + A.T.
    # For top-3, the 'x third' orderings (i,j first two, x third) collapse over
    # all x != i,j EXACTLY back to A + A.T (sum_x g_x = S - g_i - g_j), so the
    # per-x loop only needs the x-1st and x-2nd chain contributions.
    A = pv[:, None] * g[None, :] / den_i[:, None]
    np.fill_diagonal(A, 0.0)
    if k == 2:
        return A + A.T
    # --- top-3: extra horse x is 1st or 2nd; the x-3rd part collapses to A+A.T
    QP = np.zeros((n, n))
    for x in range(n):
        gx = g[x]
        s_gx = S - gx
        # x first, i second, j third   +   x first, j second, i third:
        #   p_x * g_i/(S-g_x) * g_j/(S-g_x-g_i)  +  mirrored
        if s_gx > 1e-12:
            d_x_i = np.maximum(s_gx - g[:, None], 1e-12)   # S - g_x - g_i (row)
            termA = (pv[x] * g[:, None] / s_gx) * (g[None, :] / d_x_i)
            x1 = termA + termA.T
        else:
            x1 = np.zeros((n, n))
        # i first, x second, j third   +   j first, x second, i third:
        #   p_i * g_x/(S-g_i) * g_j/(S-g_i-g_x)  +  mirrored
        d_i_x = np.maximum(S - g[:, None] - gx, 1e-12)    # S - g_i - g_x (row)
        termC = (pv[:, None] * gx / den_i[:, None]) * (g[None, :] / d_i_x)
        x2 = termC + termC.T
        # mask rows/cols == x (x must differ from both i and j)
        x1[x, :] = 0.0
        x1[:, x] = 0.0
        x2[x, :] = 0.0
        x2[:, x] = 0.0
        QP = QP + x1 + x2
    QP = QP + A + A.T
    np.fill_diagonal(QP, 0.0)
    return QP


def overlay_pricing_matrix(race_df, prob_col='prob', odds_col='win_odds',
                           smart_col='smart_money_score', gamma=None,
                           min_edge=0.25, min_smart=50.0, top_n=5,
                           n_places=None):
    """Master-Rules Task C: Quinella / Quinella-Place overlay matrix for a race.

    race_df must contain (per runner):
        prob_col   model fused win probability (float 0..1)
        odds_col   live win odds (float > 1)
        smart_col  Smart Money score S (float; absent -> treated as 50)
        horse_name (optional) / horse_number (optional) for display

    Edge = (P_model / P_market) - 1.0 computed on Henery Q & QP probabilities
    (same gamma on both sides). A pair is kept when its best edge >= min_edge
    AND at least one runner has S >= min_smart; the top `top_n` by best edge
    are returned (highest edge first).

    Returns a DataFrame (empty when nothing qualifies) with columns:
      rank, horse_i, horse_j, num_i, num_j, odds_i, odds_j, smart_i, smart_j,
      q_model_prob, q_market_prob, q_edge, qp_model_prob, qp_market_prob,
      qp_edge, best_edge
    """
    df = race_df.reset_index(drop=True).copy()
    need = [prob_col, odds_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"overlay_pricing_matrix missing columns: {missing}")
    pm = pd.to_numeric(df.get(prob_col), errors='coerce')
    po = pd.to_numeric(df.get(odds_col), errors='coerce')
    if smart_col in df.columns:
        sm = pd.to_numeric(df.get(smart_col), errors='coerce').fillna(min_smart)
    else:
        sm = pd.Series(min_smart, index=df.index)
    ok = pm.notna() & (pm > 0) & (pm < 1) & po.notna() & (po > 1.0)
    df = df[ok].reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame()
    p_model = _normalize(df[prob_col].astype(float).values)
    p_mkt = market_win_probs(df[odds_col].astype(float).values)
    n = len(df)
    if gamma is None:
        gamma = henery_gamma_for_field(n)
    k = int(n_places) if n_places else (3 if n >= 7 else 2)
    Qm = henery_quinella_matrix(p_model, gamma)
    Qk = henery_quinella_matrix(p_mkt, gamma)
    QPm = henery_qp_matrix(p_model, gamma, k)
    QPk = henery_qp_matrix(p_mkt, gamma, k)

    def name(i):
        return (str(df.loc[i, 'horse_name']) if 'horse_name' in df.columns
                else f"H{i + 1}")

    def num(i):
        return (df.loc[i, 'horse_number'] if 'horse_number' in df.columns
                else i + 1)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(sm.iloc[i]) < min_smart and float(sm.iloc[j]) < min_smart:
                continue
            q_edge = (Qm[i, j] / Qk[i, j] - 1.0) if Qk[i, j] > 1e-12 else 0.0
            qp_edge = (QPm[i, j] / QPk[i, j] - 1.0) if QPk[i, j] > 1e-12 else 0.0
            best = max(q_edge, qp_edge)
            if best < min_edge:
                continue
            rows.append({
                'horse_i': name(i), 'horse_j': name(j),
                'num_i': num(i), 'num_j': num(j),
                'odds_i': float(df.loc[i, odds_col]),
                'odds_j': float(df.loc[j, odds_col]),
                'smart_i': float(sm.iloc[i]), 'smart_j': float(sm.iloc[j]),
                'q_model_prob': Qm[i, j], 'q_market_prob': Qk[i, j],
                'q_edge': q_edge,
                'qp_model_prob': QPm[i, j], 'qp_market_prob': QPk[i, j],
                'qp_edge': qp_edge,
                'best_edge': best,
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values('best_edge', ascending=False).head(int(top_n)).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out

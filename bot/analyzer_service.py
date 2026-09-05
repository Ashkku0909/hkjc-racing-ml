import pandas as pd
import numpy as np
import os
import logging

from modeling.model_training import EDGE_DECAY_C, EDGE_DECAY_GAMMA

logger = logging.getLogger(__name__)

PREDICTIONS_FILE = "data/all_predictions.csv"
MODEL_FILE = "data/lgbm_model.txt"
CALIBRATOR_FILE = "data/calibrator.pkl"
INTENT_FILE = "data/model_features.csv"

_df = None
_model = None
_calibrator = None
_intent_df = None

def load_model():
    """Loads the LightGBM model and LogisticRegression calibrator with strict logging.

    Falls back gracefully: if either artifact is missing or fails to load, the
    bot continues to operate using the pre-computed predictions CSV.
    """
    global _model, _calibrator

    if not os.path.exists(MODEL_FILE):
        logger.warning(f"LightGBM model not found at {MODEL_FILE}. "
                       f"Run model_training.py to generate it; falling back to CSV predictions only.")
    else:
        try:
            import lightgbm as lgb
            _model = lgb.Booster(model_file=MODEL_FILE)
            logger.info(f"Loaded LightGBM model from {MODEL_FILE}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM model from {MODEL_FILE}: {e}")
            _model = None

    if not os.path.exists(CALIBRATOR_FILE):
        logger.warning(f"Calibrator not found at {CALIBRATOR_FILE}. "
                       f"Run model_training.py to generate it; falling back to CSV predictions only.")
    else:
        try:
            import pickle
            with open(CALIBRATOR_FILE, 'rb') as f:
                _calibrator = pickle.load(f)
            logger.info(f"Loaded calibrator from {CALIBRATOR_FILE}")
        except Exception as e:
            logger.error(f"Failed to load calibrator from {CALIBRATOR_FILE}: {e}")
            _calibrator = None

    if _model is None or _calibrator is None:
        logger.error("Model artifacts incomplete. Live probability estimation will fall back to historical pred_score.")

    return _model, _calibrator

def get_model():
    """Returns (model, calibrator) or (None, None) if artifacts could not be loaded."""
    return _model, _calibrator

def load_intent_features():
    """Loads trainer-intent columns (urgency / forgive / trial) from the feature store."""
    global _intent_df
    if not os.path.exists(INTENT_FILE):
        logger.warning(f"{INTENT_FILE} not found - QPL satellite intent scoring disabled.")
        _intent_df = None
        return
    try:
        cols = ['race_id', 'horse_name', 'trainer_urgency_index',
                'is_forgive_run', 'trial_won_before_race']
        _intent_df = pd.read_csv(INTENT_FILE, usecols=lambda c: c in set(cols))
        logger.info(f"Loaded intent features ({len(_intent_df)} rows) from {INTENT_FILE}")
    except Exception as e:
        logger.error(f"Failed to load intent features: {e}")
        _intent_df = None

def get_intent_features():
    """Returns the intent feature DataFrame (or None)."""
    return _intent_df

def load_data():
    global _df
    load_model()
    load_intent_features()
    if os.path.exists(PREDICTIONS_FILE):
        _df = pd.read_csv(PREDICTIONS_FILE)
        logger.info(f"Loaded {len(_df)} rows from {PREDICTIONS_FILE}")
    else:
        logger.warning(f"{PREDICTIONS_FILE} not found. Run model_training.py first.")

def get_data():
    return _df

def merge_live_odds_with_predictions(live_df, pred_df_subset):
    """Merges live odds with prediction subset based on horse name."""
    live_df = live_df.copy()
    pred_df_subset = pred_df_subset.copy()
    
    live_df['horse_name_upper'] = live_df['horse_name'].str.upper()
    pred_df_subset['horse_name_upper'] = pred_df_subset['horse_name'].str.upper()
    
    merged_df = pd.merge(
        live_df, 
        pred_df_subset[['horse_name_upper', 'true_prob', 'pred_score']], 
        on='horse_name_upper', 
        how='left'
    )
    
    has_odds_mask = merged_df['win_odds'].notna() & merged_df['true_prob'].notna()
    if has_odds_mask.any():
        # Apply non-linear penalty for live longshots to avoid favorite-longshot bias mapping errors
        # This mirrors the penalty used in model_training.py
        penalty_mask = merged_df.loc[has_odds_mask, 'win_odds'] > 8.0
        if penalty_mask.any():
            # Create a separate mask aligning to the whole dataframe
            full_penalty_mask = merged_df['win_odds'] > 8.0
            merged_df.loc[full_penalty_mask & has_odds_mask, 'true_prob'] = (
                merged_df.loc[full_penalty_mask & has_odds_mask, 'true_prob'] * 
                ((8.0 / merged_df.loc[full_penalty_mask & has_odds_mask, 'win_odds']) ** 2)
            )

        # Calculate standard EV calculation first
        merged_df.loc[has_odds_mask, 'expected_value'] = (merged_df.loc[has_odds_mask, 'true_prob'] * merged_df.loc[has_odds_mask, 'win_odds']) - 1
        
        # Calculate Market Overround normalized probability edge
        merged_df.loc[has_odds_mask, 'implied_prob'] = 1.0 / merged_df.loc[has_odds_mask, 'win_odds']
        # Compute sum overround over valid horses (assumes this DataFrame only contains ONE race)
        overround = merged_df.loc[has_odds_mask, 'implied_prob'].sum()
        
        if overround > 0:
            merged_df.loc[has_odds_mask, 'true_market_prob'] = merged_df.loc[has_odds_mask, 'implied_prob'] / overround
            merged_df.loc[has_odds_mask, 'prob_edge'] = merged_df.loc[has_odds_mask, 'true_prob'] - merged_df.loc[has_odds_mask, 'true_market_prob']

        # Favorite-longshot decay penalty + decay-adjusted EV (mirrors model_training.py)
        # Adjusted Edge = (true_prob - true_market_prob) * (C / odds) ** gamma
        merged_df.loc[has_odds_mask, 'decay_factor'] = (
            (EDGE_DECAY_C / merged_df.loc[has_odds_mask, 'win_odds']) ** EDGE_DECAY_GAMMA
        )
        merged_df.loc[has_odds_mask, 'prob_edge'] = (
            merged_df.loc[has_odds_mask, 'prob_edge'] * merged_df.loc[has_odds_mask, 'decay_factor']
        )
        merged_df.loc[has_odds_mask, 'expected_value'] = (
            merged_df.loc[has_odds_mask, 'true_prob'] * merged_df.loc[has_odds_mask, 'win_odds']
            * merged_df.loc[has_odds_mask, 'decay_factor'] - 1.0
        )

    merged_df = merged_df.drop(columns=['horse_name_upper'])
    return merged_df

def estimate_probabilities_from_history(live_df, all_df):
    """Estimates probabilities using horses' most recent historical data."""
    live_df = live_df.copy()
    live_df['horse_name_upper'] = live_df['horse_name'].str.upper()
    
    recent_horse_data = all_df.sort_values('race_date', ascending=False).drop_duplicates(subset=['horse_name'])
    recent_horse_data['horse_name_upper'] = recent_horse_data['horse_name'].str.upper()
    
    merged_df = pd.merge(
        live_df,
        recent_horse_data[['horse_name_upper', 'pred_score']],
        on='horse_name_upper',
        how='left'
    )
    
    if merged_df['pred_score'].notna().any():
        min_score = merged_df['pred_score'].min()
        if pd.isna(min_score):
            min_score = 0
        merged_df['pred_score'] = merged_df['pred_score'].fillna(min_score / 2.0)

        # Normalize probability output from the binary predictor
        pred_scores = merged_df['pred_score']
        merged_df['true_prob'] = pred_scores / (pred_scores.sum() + 1e-9)

        has_odds_mask = merged_df['win_odds'].notna()
        if has_odds_mask.any():
            # Apply longshot penalty
            full_penalty_mask = merged_df['win_odds'] > 8.0
            merged_df.loc[full_penalty_mask & has_odds_mask, 'true_prob'] = (
                merged_df.loc[full_penalty_mask & has_odds_mask, 'true_prob'] * 
                ((8.0 / merged_df.loc[full_penalty_mask & has_odds_mask, 'win_odds']) ** 2)
            )
            
    return merged_df


# =====================================================================
# Pure Quant Smart Money Flow Engine (no LLM, vectorized numpy/pandas)
# =====================================================================

# Odds velocity scoring: S = clip(50 + 200 * Δp, 0, 100)
SMART_MONEY_CENTER = 50.0
SMART_MONEY_GAIN = 200.0
# Bayesian prior-posterior weight for the smart-money signal
SMART_MONEY_UPDATE_WEIGHT = 0.20
# Flow status thresholds
SMART_STEAMER_THRESHOLD = 75.0
SMART_DRIFTER_THRESHOLD = 40.0


def get_flow_signal(score) -> str:
    """Maps a smart_money_score (0-100) to a flow signal label. NaN-safe."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return '➖ STABLE'
    if np.isnan(s):
        return '➖ STABLE'
    if s >= SMART_STEAMER_THRESHOLD:
        return '🔥 STEAMER'
    if s < SMART_DRIFTER_THRESHOLD:
        return '⚠️ DRIFTER'
    return '➖ STABLE'


def calculate_smart_money_metrics(current_odds_df, baseline_odds_df):
    """Smart money scoring from live vs baseline odds (single race, vectorized).

        Δp_i = (1/O_live,i − 1/O_init,i) − mean_race(1/O_live − 1/O_init)
        S_smart,i = clip(50 + 200·Δp_i, 0, 100)

    Joins on uppercased horse_name. NaN-safe:
      - identical baseline/live odds  -> every score = 50 (➖ STABLE)
      - missing baseline (new runner) -> score = 50 (no flow evidence)
      - missing live odds             -> score = 50 (not yet posted)

    Returns the current frame enriched with:
      smart_money_score, flow_signal, odds_prob_delta
    """
    cur = current_odds_df.copy()
    base = baseline_odds_df.copy()

    for frame in (cur, base):
        frame['_key'] = frame['horse_name'].astype(str).str.upper()

    merged = cur.merge(
        base[['_key', 'win_odds']].rename(columns={'win_odds': '_init_odds'}),
        on='_key', how='left')

    live = pd.to_numeric(merged['win_odds'], errors='coerce')
    init = pd.to_numeric(merged['_init_odds'], errors='coerce')

    p_live = np.where((live.notna() & (live > 1.0)).to_numpy(), 1.0 / live.to_numpy(dtype=float), np.nan)
    p_init = np.where((init.notna() & (init > 1.0)).to_numpy(), 1.0 / init.to_numpy(dtype=float), np.nan)

    delta = p_live - p_init
    valid = ~np.isnan(delta)
    n_valid = int(valid.sum())
    mean_delta = float(np.nanmean(delta)) if n_valid > 0 else 0.0
    delta_demeaned = np.where(valid, delta - mean_delta, 0.0)

    score = SMART_MONEY_CENTER + SMART_MONEY_GAIN * delta_demeaned
    score = np.clip(score, 0.0, 100.0)
    # Rows without a valid live-vs-init pair carry no flow evidence -> neutral 50
    score = np.where(valid, score, SMART_MONEY_CENTER)

    merged['odds_prob_delta'] = delta_demeaned
    merged['smart_money_score'] = score
    merged['flow_signal'] = [get_flow_signal(s) for s in score]
    merged = merged.drop(columns=['_key', '_init_odds'], errors='ignore')
    return merged


def apply_smart_money_bayesian_update(df, score_col='smart_money_score',
                                      prob_col='true_prob', race_col='race_id'):
    """Bayesian prior-posterior refinement of the model probability.

        logit(p_live,i) = logit(p_model,i) + 0.20 · ((S_smart,i − 50) / 50)

    followed by a per-race Softmax so Σ p_live = 1.0. Also recomputes the
    decay-adjusted live EV and the live probability edge vs the market.
    """
    out = df.copy()
    if prob_col not in out.columns:
        return out
    if score_col not in out.columns:
        out[score_col] = SMART_MONEY_CENTER

    eps = 1e-12
    p = np.clip(pd.to_numeric(out[prob_col], errors='coerce').fillna(0.0).to_numpy(), eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))

    scores = pd.to_numeric(out[score_col], errors='coerce').fillna(SMART_MONEY_CENTER).to_numpy()
    adjustment = SMART_MONEY_UPDATE_WEIGHT * (scores - SMART_MONEY_CENTER) / 50.0

    out['_live_logit'] = logit + adjustment

    key = out[race_col] if race_col in out.columns else pd.Series(0, index=out.index)

    def _softmax(g):
        s = g - g.max()
        e = np.exp(s)
        return e / e.sum()

    out['live_prob'] = out.groupby(key)['_live_logit'].transform(_softmax)
    out = out.drop(columns=['_live_logit'])

    if 'win_odds' in out.columns:
        odds = pd.to_numeric(out['win_odds'], errors='coerce')
        # Clamp the favorite-longshot decay at 1.0: (C/odds)^gamma > 1 for
        # odds < C would FABRICATE edge for short-priced favourites (e.g. a
        # 1.0 heavy favourite in an open pool is real - not a placeholder).
        raw = np.power(EDGE_DECAY_C / np.where(odds > 0, odds, np.nan), EDGE_DECAY_GAMMA)
        decay = np.minimum(raw, 1.0)
        out['live_expected_value'] = out['live_prob'] * odds * decay - 1.0
        impl = pd.Series(1.0 / np.where(odds > 0, odds, np.nan), index=out.index)
        mkt = impl.groupby(key).transform(lambda x: x / x.sum())
        out['live_prob_edge'] = out['live_prob'] - mkt

    return out

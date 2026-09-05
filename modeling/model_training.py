import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import os
import pickle
from typing import Tuple
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Probability sharpening is parameterized via Softmax temperature scaling.
# Default temperature lives in the recommended range [0.8, 1.5];
# higher tau -> smoother (less confident) per-race distribution.
DEFAULT_TEMPERATURE = 1.0

# --- Favorite-Longshot Bias Decay (Benter/Ziemba style) ---
# Adjusted Edge = (true_prob - true_market_prob) * (C / market_odds) ** gamma
EDGE_DECAY_C = 4.0
EDGE_DECAY_GAMMA = 1.25

# --- Kelly betting window (the sweet spot) ---
# Tuned via 2021-2026 walk-forward sensitivity sweep (data/kelly_sweep_results.csv):
# Pareto-optimal combo maximizing ROI & Sharpe with bet count > 50.
KELLY_MIN_ODDS = 4.5
KELLY_MAX_ODDS = 8.0
KELLY_MIN_EV = 1.22       # decay-adjusted odds-ratio EV threshold

# --- Simultaneous Fractional Kelly sizing ---
KELLY_FRACTION = 0.15       # f* = 0.15 x full Kelly
KELLY_MAX_BET_FRACTION = 0.02   # single-horse bankroll cap: 2.0%
KELLY_MAX_RACE_EXPOSURE = 0.06  # per-race total exposure cap: 6.0%

# --- LambdaMART ranking configuration ---
# NDCG evaluation cutoffs used during training and in the backtest report
RANKING_EVAL_AT = [1, 2, 3]
# Market baseline weight: log(true_market_prob) is added to the ranking score
# (identical to training with init_score = market log-odds). beta=1.0 makes the
# model fit the market-mispricing residual; 0.0 disables the market prior.
MARKET_BLEND_BETA = 1.0

# Task A/B feature block (claim imputation, implied rating, race-context
# relatives). Excluded when legacy_features=True (empirically the legacy set
# is the profitable production config for the binary model).
TASK_AB_FEATURES = [
    'jockey_allowance', 'declared_weight', 'effective_carried_weight',
    'implied_rating', 'rating_diff_vs_class_max',
    'rel_speed_to_race_mean', 'rel_speed_to_race_max', 'speed_rank_in_field',
    'weight_rel_to_top', 'weight_rel_to_mean',
    'race_front_runner_density', 'early_speed_vs_field_fastest',
]


def _dcg_at(gains, k: int) -> float:
    """Discounted cumulative gain over the first k items (log2 discount)."""
    g = np.asarray(gains[:k], dtype=float)
    if len(g) == 0:
        return 0.0
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def _ndcg_at_k(rel_sorted, k: int) -> float:
    """NDCG@k for one race given relevance grades in ranked order (2^r - 1 gains)."""
    rel = np.asarray(rel_sorted, dtype=float)
    gains = 2.0 ** rel - 1.0
    dcg = _dcg_at(gains, k)
    ideal = _dcg_at(np.sort(gains)[::-1], k)
    return float(dcg / ideal) if ideal > 0 else np.nan


class RacingPipeline:
    def __init__(self, features_csv="data/model_features.csv", temperature: float = DEFAULT_TEMPERATURE,
                 edge_decay_c: float = EDGE_DECAY_C, edge_decay_gamma: float = EDGE_DECAY_GAMMA,
                 mode: str = 'plackett_luce', market_blend_beta: float = MARKET_BLEND_BETA,
                 legacy_features: bool = False):
        self.features_csv = features_csv
        # Clamp to the recommended range [0.8, 1.5] to keep scaling stable
        self.temperature = float(np.clip(temperature, 0.8, 1.5))
        # Favorite-longshot decay parameters (tunable)
        self.edge_decay_c = float(edge_decay_c)
        self.edge_decay_gamma = float(edge_decay_gamma)
        # Model family:
        #   'plackett_luce' (PRIMARY - group-aware softmax cross-entropy),
        #   'lambdarank'    (ranking baseline), 'binary' (pooled BCE baseline)
        self.mode = mode if mode in ('plackett_luce', 'lambdarank', 'binary') else 'plackett_luce'
        self.market_blend_beta = float(market_blend_beta)
        # Drop the Task A/B feature block (restores the pre-refactor feature set)
        self.legacy_features = bool(legacy_features)

    @staticmethod
    def _finish_rank(series: pd.Series) -> pd.Series:
        """Parses finish_position ('1', '2 DH', '10') into an integer rank."""
        return pd.to_numeric(series.astype(str).str.extract(r'^(\d+)')[0], errors='coerce')

    @staticmethod
    def _relevance(rank_series: pd.Series) -> pd.Series:
        """Ranking relevance grades: winner=3, 2nd=2, 3rd=1, 4th+=0."""
        rel = np.zeros(len(rank_series), dtype=int)
        r = rank_series.fillna(99).astype(int).values
        rel[r == 1] = 3
        rel[r == 2] = 2
        rel[r == 3] = 1
        return pd.Series(rel, index=rank_series.index)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.features_csv}...")
        if not os.path.exists(self.features_csv):
            raise FileNotFoundError(f"Could not find {self.features_csv}.")
            
        df = pd.read_csv(self.features_csv)
        
        df['race_date'] = pd.to_datetime(df['race_date'])
        df['finishing_time'] = pd.to_numeric(df['finishing_time'], errors='coerce')
        df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
        df['is_win'] = pd.to_numeric(df['is_win'], errors='coerce').fillna(0)
        
        df = df.dropna(subset=['finishing_time', 'win_odds'])
        return df

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list):
        """Dispatches to the configured model family."""
        if self.mode == 'plackett_luce':
            return self._train_plackett_luce(train_df, val_df, features)
        if self.mode == 'lambdarank':
            return self._train_lambdarank(train_df, val_df, features)
        return self._train_binary(train_df, val_df, features)

    def _train_binary(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list):
        logger.info("Training LightGBM Binary Model to Predict Win Probability...")
        
        X_train = train_df[features]
        y_train = train_df['is_win']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        callbacks = []

        if len(val_df) > 0:
            X_val = val_df[features]
            y_val = val_df['is_win']
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))
            
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=150,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        
        # Fit calibrator on validation set if available, otherwise on train set
        calibrator = LogisticRegression()
        if len(val_df) > 0:
            val_preds = model.predict(val_df[features])
            calibrator.fit(val_preds.reshape(-1, 1), val_df['is_win'])
        else:
            train_preds = model.predict(train_df[features])
            calibrator.fit(train_preds.reshape(-1, 1), train_df['is_win'])
            
        return model, calibrator

    @staticmethod
    def _market_logit(df: pd.DataFrame) -> np.ndarray:
        """log(true_market_prob / (1 - true_market_prob)) per row."""
        p = pd.to_numeric(df['true_market_prob'], errors='coerce').fillna(0.5).values
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    def _train_lambdarank(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list):
        """LightGBM LambdaMART (objective='lambdarank', metric NDCG@1/2/3).

        Relevance grades: winner=3, 2nd=2, 3rd=1, 4th+=0 (label gains 2^r-1).
        The market-implied log-odds are supplied as init_score so the trees fit
        the residual vs the market; the same offset is re-added at inference.
        """
        logger.info("Training LightGBM LambdaMART ranking model (metric NDCG@1/2/3)...")

        def build_ranking_df(df, features):
            out = df.copy()
            out['_rank'] = self._finish_rank(out['finish_position'])
            out['_rel'] = self._relevance(out['_rank'])
            # Grouped by race for lambdarank; stable order inside groups
            out = out.sort_values(['race_id', '_rel'], ascending=[True, False])
            return out

        tr = build_ranking_df(train_df, features)
        groups = tr.groupby('race_id', sort=True).size().values.astype(int)
        X_train = tr[features]
        y_train = tr['_rel'].astype(int)

        init_score = None
        if 'true_market_prob' in tr.columns and tr['true_market_prob'].notna().all():
            init_score = self._market_logit(tr)

        def make_dataset(df_sorted, groups, ref=None):
            kwargs = {'label': df_sorted['_rel'].astype(int), 'group': groups}
            if init_score is not None:
                kwargs['init_score'] = self._market_logit(df_sorted)
            ds = lgb.Dataset(df_sorted[features], **kwargs)
            if ref is not None:
                ds = lgb.Dataset(df_sorted[features], reference=ref, **kwargs)
            return ds

        try:
            train_data = make_dataset(tr, groups)
        except Exception as e:
            logger.warning("init_score rejected for lambdarank (%s) - training without market init.", e)
            init_score = None
            train_data = make_dataset(tr, groups)

        valid_sets = [train_data]
        callbacks = []
        if len(val_df) > 0:
            va = build_ranking_df(val_df, features)
            v_groups = va.groupby('race_id', sort=True).size().values.astype(int)
            val_data = make_dataset(va, v_groups, ref=train_data)
            valid_sets.append(val_data)
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False, first_metric_only=True))

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': RANKING_EVAL_AT,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=250,
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        # Monotone recalibration of the final combined score (ranking score +
        # market baseline) onto a probability scale; keeps ranking order intact.
        calibrator = LogisticRegression()
        if len(val_df) > 0:
            raw_val = model.predict(val_df[features])
            combined_val = raw_val + self.market_blend_beta * self._market_logit(val_df)
            calibrator.fit(combined_val.reshape(-1, 1), val_df['is_win'])
        else:
            raw_tr = model.predict(train_df[features])
            combined_tr = raw_tr + self.market_blend_beta * self._market_logit(train_df)
            calibrator.fit(combined_tr.reshape(-1, 1), train_df['is_win'])

        return model, calibrator

    def _train_plackett_luce(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list):
        """PRIMARY objective: group-aware Softmax Cross-Entropy (Plackett-Luce).

        Uses the FULL finishing order per race. For a race with runners placed
        h_1..h_n the negative Plackett-Luce log-likelihood is

            L = -Σ_{t=1..n} ( s_{h_t} - log Σ_{j∈A_t} exp(s_j) )

        where A_t = runners not yet placed before step t, and the full logit is
        s_i = f_i + β·logit(true_market_prob_i) — the trees fit the market
        pricing residual. Being a per-race categorical likelihood it is
        inherently group-aware: varying field sizes (4-14) need no prior
        adjustment, unlike pooled binary CE.

        Row-wise gradient / Hessian for horse i placed at step t_i:
            g_i = -1 + exp(s_i)·Σ_{t≤t_i} exp(-S_t),   S_t = logsumexp(A_t)
            h_i = exp(s_i)·Σ_{t≤t_i} exp(-S_t)
                  - exp(2·s_i)·Σ_{t≤t_i} exp(-2·S_t)
        Dead-heats are broken deterministically by win_odds (rare in HK).
        """
        logger.info("Training LightGBM Plackett-Luce group softmax-CE (custom fobj)...")

        tr = train_df.copy()
        tr['_rank'] = self._finish_rank(tr['finish_position'])
        # Deterministic order inside a race: finishing position, price as tie-break
        tr = tr.sort_values(['race_id', '_rank', 'win_odds'],
                            ascending=[True, True, True]).reset_index(drop=True)
        sizes = tr.groupby('race_id', sort=True).size()
        tr = tr[tr['race_id'].isin(sizes[sizes >= 2].index)].reset_index(drop=True)

        offset = np.zeros(len(tr))
        if 'true_market_prob' in tr.columns and tr['true_market_prob'].notna().all():
            offset = self.market_blend_beta * self._market_logit(tr)

        # Contiguous race boundaries (tr is sorted race-major, rows in finish order)
        codes, _ = pd.factorize(tr['race_id'], sort=True)
        boundaries = np.flatnonzero(np.diff(codes) != 0) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(tr)]])

        def pl_fobj(preds: np.ndarray, _dataset) -> Tuple[np.ndarray, np.ndarray]:
            """(grad, hess) of the full-order Plackett-Luce NLL per row."""
            s = np.asarray(preds, dtype=np.float64) + offset
            g = np.empty_like(s)
            h = np.empty_like(s)
            for a, b in zip(starts, ends):
                sl = np.clip(s[a:b], -30.0, 30.0)
                e = np.exp(sl)
                # S_t = log(Σ_{j≥t} e_j) via reversed cumulative sum
                S = np.log(np.maximum(np.cumsum(e[::-1])[::-1], 1e-300))
                C = np.cumsum(np.exp(-S))            # prefix Σ_{t≤i} exp(-S_t)
                D = np.cumsum(np.exp(-2.0 * S))      # prefix Σ_{t≤i} exp(-2S_t)
                eC = e * C
                g[a:b] = eC - 1.0
                h[a:b] = eC - e * e * D
            return g, np.maximum(h, 1e-6)

        X_train = tr[features].reset_index(drop=True)
        train_data = lgb.Dataset(X_train, label=np.zeros(len(tr)))

        params = {
            # LightGBM 4.x: custom objectives are passed as a callable via params
            'objective': pl_fobj,
            'metric': 'None',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        model = lgb.train(params, train_data, num_boost_round=200)

        # Monotone recalibration of the combined (tree + market baseline) score
        def combined(df_in: pd.DataFrame) -> np.ndarray:
            return model.predict(df_in[features]) + self.market_blend_beta * self._market_logit(df_in)

        calibrator = LogisticRegression()
        if len(val_df) > 0:
            calibrator.fit(combined(val_df).reshape(-1, 1), val_df['is_win'])
        else:
            calibrator.fit(combined(train_df).reshape(-1, 1), train_df['is_win'])

        return model, calibrator

    def apply_power_law(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing True Market Overround via Power Law...")
        
        df['implied_prob'] = 1.0 / df['win_odds']
        
        def calculate_true_market_probs(implied):
            def objective(n):
                return np.sum(implied ** n) - 1.0
            try:
                from scipy.optimize import brentq
                n_opt = brentq(objective, 0.01, 2.0)
                return implied ** n_opt
            except:
                return implied / implied.sum()

        df['true_market_prob'] = df.groupby('race_id')['implied_prob'].transform(calculate_true_market_probs)
        return df

    def convert_to_probabilities(self, df: pd.DataFrame, models: tuple, features: list, temperature: float = None) -> pd.DataFrame:
        """Converts raw LightGBM scores into per-race win probabilities.

        Calibrated probabilities are mapped back to log-odds and passed through
        a temperature-scaled Softmax within each race:

            P(y_i = 1 | Race k) = exp(z_i / tau) / sum_j exp(z_j / tau)

        where z_i is the calibrated logit (log-odds) and tau the temperature.
        """
        if len(df) == 0:
            return df

        if temperature is None:
            temperature = self.temperature

        model, calibrator = models

        # Market-implied probabilities are the baseline offset for lambdarank
        # (and the EV reference in both modes) -> compute them first.
        if 'true_market_prob' not in df.columns or df['true_market_prob'].isna().all():
            df = self.apply_power_law(df)

        raw_score = model.predict(df[features])
        if self.mode in ('lambdarank', 'plackett_luce'):
            # Final score = ranking/residual score + beta * logit(true_market_prob).
            # This mirrors training with a market baseline offset: the trees
            # predict the residual vs the market.
            combined = raw_score + self.market_blend_beta * self._market_logit(df)
            df['pred_score'] = combined
            calibrated_prob = calibrator.predict_proba(combined.reshape(-1, 1))[:, 1]
        else:
            df['pred_score'] = raw_score
            # Calibrate the raw scores using LogisticRegression for a reliable probability scale
            calibrated_prob = calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]

        # Convert calibrated probabilities to log-odds (logits) for Softmax scaling
        eps = 1e-12
        p_clipped = np.clip(calibrated_prob, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        logit_series = pd.Series(logits, index=df.index)

        def softmax_with_temperature(group_logits):
            scaled = group_logits / temperature
            shifted = scaled - scaled.max()  # numerically stable softmax
            exps = np.exp(shifted)
            return exps / exps.sum()

        df['calibrated_prob'] = calibrated_prob
        df['true_prob'] = logit_series.groupby(df['race_id']).transform(softmax_with_temperature)

        df = self.apply_power_law(df)

        # Favorite-longshot bias decay: non-linear odds penalty on the probability edge.
        # Adjusted Edge = (true_prob - true_market_prob) * (C / odds) ** gamma
        df['decay_factor'] = (self.edge_decay_c / df['win_odds']) ** self.edge_decay_gamma
        df['prob_edge'] = (df['true_prob'] - df['true_market_prob']) * df['decay_factor']

        # Decay-adjusted expected value (odds-ratio minus 1, so > 0 means profit)
        df['expected_value'] = df['true_prob'] * df['win_odds'] * df['decay_factor'] - 1.0
        
        return df

    def _ranking_metrics(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Adds ranking-eval columns (rank, relevance) for NDCG / logloss / Brier."""
        out = test_df[['race_id', 'finish_position', 'is_win', 'true_prob', 'true_market_prob']].copy()
        out['rank'] = self._finish_rank(out['finish_position'])
        out['relevance'] = self._relevance(out['rank'])
        return out

    def simultaneous_kelly(self, group_df, fraction=KELLY_FRACTION,
                          max_bet_fraction=KELLY_MAX_BET_FRACTION,
                          max_race_exposure=KELLY_MAX_RACE_EXPOSURE,
                          min_ev=KELLY_MIN_EV, odds_min=KELLY_MIN_ODDS, odds_max=KELLY_MAX_ODDS):
        """Simultaneous Multi-Horse Fractional Kelly (mutually exclusive outcomes).

        Entry rules:
          - Decay-adjusted EV = true_prob * win_odds * (C/odds)^gamma > min_ev (1.18)
          - Overlay odds band: odds_min <= win_odds <= odds_max (2.2..8.5)
        Sizing:
          - Fractional Kelly: fraction * f_full (default 15%)
          - Single-horse bankroll cap: 2.0%
          - Per-race total exposure cap: 6.0% (all stakes scaled proportionally)
        """
        df = group_df.copy()

        df = df[df['win_odds'].notna() & df['true_prob'].notna()]
        df['decay_factor'] = (self.edge_decay_c / df['win_odds']) ** self.edge_decay_gamma
        df['ev'] = df['true_prob'] * df['win_odds'] * df['decay_factor']

        # Pure positive edge within the sweet-spot overlay band
        df = df[(df['win_odds'] >= odds_min) & (df['win_odds'] <= odds_max) & (df['ev'] > min_ev)]

        race_df = df.sort_values('ev', ascending=False)
        fractions = pd.Series(0.0, index=group_df.index)

        if len(race_df) == 0:
            return fractions

        R_S = 1.0
        O_S = 1.0
        S = []

        for idx, row in race_df.iterrows():
            if row['ev'] > (R_S / O_S):
                S.append(idx)
                R_S -= row['true_prob']
                O_S -= 1.0 / row['win_odds']
            else:
                break

        if O_S <= 0:
            return fractions

        # Full Kelly stake per selected horse, then apply Fractional Kelly
        raw = []
        for idx in S:
            f_full = race_df.loc[idx, 'true_prob'] - (R_S / O_S) * (1.0 / race_df.loc[idx, 'win_odds'])
            raw.append((idx, max(f_full * fraction, 0.0)))

        # Single-horse cap
        capped = [(idx, min(f, max_bet_fraction)) for idx, f in raw]

        # Per-race exposure cap: scale all stakes proportionally when exceeded
        total_exposure = sum(f for _, f in capped)
        if total_exposure > max_race_exposure:
            scale = max_race_exposure / total_exposure
            capped = [(idx, f * scale) for idx, f in capped]

        for idx, f in capped:
            fractions[idx] = f

        return fractions

    def _prepare_features(self, df: pd.DataFrame):
        """Shared feature list / NaN handling / categorical conversion for walk-forward runs."""
        features = [
            'distance', 'barrier_draw', 'weight_carried',
            'last_speed_figure', 'speed_figure_change', 'speed_figure_ewm',
            'horse_win_rate_50', 'horse_place_rate_50', 'horse_win_rate_ewm', 'horse_place_rate_ewm',
            'jockey_win_rate_50', 'trainer_win_rate_50', 'jockey_roi_50', 'trainer_roi_50',
            'weight_diff', 'horse_weight_diff', 'avg_early_pct_last_3', 'avg_late_speed_last_3',
            'course_type', 'track_condition', 'race_class', 'track', 'jockey', 'trainer',
            'days_since_last_race', 'rest_period', 'jockey_trainer_win_rate_50', 'barrier_win_rate_50',
            'jockey_trainer_roi_50', 'is_high_value_combo', 'run_style', 'num_front_runners', 'pace_advantage', 'pace_scenario',
            'race_class_diff', 'distance_diff', 'last_finish_pos', 'career_starts', 'is_unexposed_star',
            # --- Newly activated features (already present in model_features.csv) ---
            # past_speed_figure: horse's speed figure from its previous start (0 for first-timers)
            # past_standardized_early_speed: leak-free lagged early-speed z-score.
            #   (the raw standardized_early_speed is intentionally NOT stored in the CSV
            #    so the current race's sectional data cannot leak into predictions)
            # early_pressure_index: race-level pace metric from the top-3 early runners
            'past_speed_figure', 'past_standardized_early_speed', 'early_pressure_index',
            # --- Claim imputation / implied rating (Task A) ---
            'jockey_allowance', 'declared_weight', 'effective_carried_weight',
            'implied_rating', 'rating_diff_vs_class_max',
            # --- Race-context relative features (Task B) ---
            'rel_speed_to_race_mean', 'rel_speed_to_race_max', 'speed_rank_in_field',
            'weight_rel_to_top', 'weight_rel_to_mean',
            'race_front_runner_density', 'early_speed_vs_field_fastest',
        ]

        features = [f for f in features if f in df.columns]

        # Legacy production feature set (pre Task A/B) - empirically the
        # profitable config for the binary model (+34% ROI walk-forward).
        if self.legacy_features:
            features = [f for f in features if f not in TASK_AB_FEATURES]

        # Robust NaN handling for the new features (first-time runners, missing sectional data).
        # Feature engineering already fills these with 0.0; this is a safety net so training
        # never sees NaN where no history is available.
        for col in ['past_speed_figure', 'past_standardized_early_speed', 'early_pressure_index',
                    'jockey_allowance', 'declared_weight', 'effective_carried_weight',
                    'implied_rating', 'rating_diff_vs_class_max',
                    'rel_speed_to_race_mean', 'rel_speed_to_race_max', 'speed_rank_in_field',
                    'weight_rel_to_top', 'weight_rel_to_mean',
                    'race_front_runner_density', 'early_speed_vs_field_fastest']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        categorical_cols = ['course_type', 'track_condition', 'race_class', 'run_style', 'rest_period', 'pace_scenario', 'track', 'jockey', 'trainer']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        if 'horse_weight' in df.columns:
            features.append('horse_weight')
            df['horse_weight'] = df['horse_weight'].fillna(df['horse_weight'].median())

        return df, features

    def run_walk_forward_predictions(self, cache_path="data/walk_forward_preds.csv"):
        """Runs the monthly-retrain walk-forward loop WITHOUT betting and caches the
        out-of-sample predictions. This lets the Kelly grid search replay instantly."""
        df = self.load_data()
        df, features = self._prepare_features(df)
        # Market-implied probs are the ranking baseline / EV reference
        df = self.apply_power_law(df)

        logger.info("Caching walk-forward out-of-sample predictions (no betting)...")
        start_date = df['race_date'].min() + pd.DateOffset(months=12)
        end_date = df['race_date'].max()

        current_date = start_date
        cache_cols = ['race_date', 'race_id', 'horse_name', 'is_win', 'win_odds',
                      'true_prob', 'true_market_prob', 'prob_edge', 'decay_factor',
                      'expected_value', 'pred_score']
        cached_frames = []

        while current_date < end_date:
            month_end = current_date + pd.offsets.MonthEnd(0)
            if month_end > end_date:
                month_end = end_date

            train_mask = df['race_date'] < current_date
            test_mask = (df['race_date'] >= current_date) & (df['race_date'] <= month_end)
            train_df = df[train_mask]
            test_df = df[test_mask].copy()

            if len(test_df) == 0:
                current_date = month_end + pd.DateOffset(days=1)
                continue

            split_idx = int(len(train_df) * 0.8)
            t_df = train_df.iloc[:split_idx]
            v_df = train_df.iloc[split_idx:]

            models = self.train_model(t_df, v_df, features)
            test_df = self.convert_to_probabilities(test_df, models, features)

            avail = [c for c in cache_cols if c in test_df.columns]
            cached_frames.append(test_df[avail].copy())
            current_date = month_end + pd.DateOffset(days=1)

        cache_df = pd.concat(cached_frames, ignore_index=True)
        cache_df.to_csv(cache_path, index=False)
        logger.info(f"Cached {len(cache_df)} prediction rows to {cache_path}")
        return cache_df

    def run_walk_forward_backtest(self):
        df = self.load_data()
        df, features = self._prepare_features(df)
        # Market-implied probs are the ranking baseline / EV reference
        df = self.apply_power_law(df)

        logger.info(f"Starting Walk-Forward Backtesting optimizing ROI using Simultaneous Kelly "
                    f"(mode={self.mode}, softmax tau={self.temperature:.2f})...")
        
        # Start backtesting allowing roughly 1 year for the initial training set
        start_date = df['race_date'].min() + pd.DateOffset(months=12) 
        end_date = df['race_date'].max()
        
        current_date = start_date
        initial_bankroll = 10000.0
        current_bankroll = initial_bankroll
        
        total_invested = 0.0
        total_returned = 0.0
        total_bets = 0
        monthly_results = []
        equity_curve = []  # bankroll snapshot per race date for ROI curve / drawdown
        bet_records = []   # per-bet ledger for win rate & odds band breakdown
        eval_frames = []   # per-row ranking eval data (NDCG / logloss / Brier)

        while current_date < end_date:
            month_end = current_date + pd.offsets.MonthEnd(0)
            if month_end > end_date:
                month_end = end_date
                
            train_mask = df['race_date'] < current_date
            test_mask = (df['race_date'] >= current_date) & (df['race_date'] <= month_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask].copy()
            
            if len(test_df) == 0:
                current_date = month_end + pd.DateOffset(days=1)
                continue
                
            # Internal Val split for early stopping
            split_idx = int(len(train_df) * 0.8)
            t_df = train_df.iloc[:split_idx]
            v_df = train_df.iloc[split_idx:]
            
            models = self.train_model(t_df, v_df, features)
            
            # Predict and convert to probabilities via temperature-scaled Softmax.
            # The temperature is shared with backtesting.py through DEFAULT_TEMPERATURE.
            test_df = self.convert_to_probabilities(test_df, models, features)

            # Ranking-quality eval rows for this month (NDCG / logloss / Brier)
            eval_frames.append(self._ranking_metrics(test_df))
            
            monthly_invested = 0.0
            monthly_returned = 0.0
            daily_bets = 0
            month_start_bankroll = current_bankroll
            
            for date, group in test_df.groupby('race_date'):
                if current_bankroll < 10.0: break
                
                # Sizing based on simultaneous kelly solver per race
                for race_id, race_group in group.groupby('race_id'):
                    # Simultaneous Fractional Kelly: 15% Kelly, 2.0% per-horse cap,
                    # 6.0% per-race exposure cap, decay-adjusted EV > 1.18, odds 2.2-8.5
                    fractions = self.simultaneous_kelly(race_group)
                    
                    race_invested = 0.0
                    race_returned = 0.0
                    for idx, f in fractions.items():
                        if f > 0:
                            # Per-horse bankroll cap already applied inside simultaneous_kelly
                            f = min(f, KELLY_MAX_BET_FRACTION)
                            bet_amt = current_bankroll * f
                            
                            if bet_amt >= 5.0: # Minimum bet threshold
                                race_invested += bet_amt
                                total_bets += 1
                                daily_bets += 1
                                won = int(race_group.loc[idx, 'is_win'] == 1)
                                ret = bet_amt * race_group.loc[idx, 'win_odds'] if won else 0.0
                                race_returned += ret
                                bet_records.append({
                                    'date': date,
                                    'race_id': race_id,
                                    'horse_name': (race_group.loc[idx, 'horse_name']
                                                   if 'horse_name' in race_group.columns else ''),
                                    'odds': race_group.loc[idx, 'win_odds'],
                                    'stake': bet_amt,
                                    'won': won,
                                    'return': ret,
                                })
                    
                    # Update bankroll after each race so it recalculates sizing properly
                    current_bankroll = current_bankroll - race_invested + race_returned
                    monthly_invested += race_invested
                    monthly_returned += race_returned

                equity_curve.append({'date': date, 'bankroll': current_bankroll})
            
            profit = monthly_returned - monthly_invested
            roi = (profit / monthly_invested * 100) if monthly_invested > 0 else 0
            
            logger.info(f"Month: {current_date.strftime('%Y-%m')} | Bets: {daily_bets} | Invested: ${monthly_invested:.2f} | Profit: ${profit:.2f} | ROI: {roi:.2f}%")
            
            monthly_results.append({
                'Month': current_date.strftime('%Y-%m'),
                'Invested': monthly_invested,
                'Profit': profit,
                'ROI': roi,
                'StartBankroll': month_start_bankroll
            })
            
            total_invested += monthly_invested
            total_returned += monthly_returned
            current_date = month_end + pd.DateOffset(days=1)

        total_profit = total_returned - total_invested
        total_roi = (total_profit / total_invested * 100) if total_invested > 0 else 0

        # --- Equity curve, max drawdown & Sharpe ratio ---
        eq_df = pd.DataFrame(equity_curve)
        max_drawdown = 0.0
        if len(eq_df) > 1:
            eq_df['date'] = pd.to_datetime(eq_df['date'])
            eq_df['roi_pct'] = (eq_df['bankroll'] / initial_bankroll - 1.0) * 100.0
            eq_df['peak'] = eq_df['bankroll'].cummax()
            eq_df['drawdown'] = (eq_df['bankroll'] - eq_df['peak']) / eq_df['peak']
            max_drawdown = eq_df['drawdown'].min() * 100.0
            try:
                eq_df.to_csv('data/backtest_equity_curve.csv', index=False)
            except Exception as e:
                logger.warning(f"Could not save equity curve: {e}")

        sharpe = 0.0
        mres = pd.DataFrame(monthly_results)
        if len(mres) > 1 and (mres['StartBankroll'] > 0).all():
            monthly_returns = mres['Profit'] / mres['StartBankroll']
            std = monthly_returns.std()
            if std and std > 0:
                sharpe = monthly_returns.mean() / std * np.sqrt(12)  # annualized (monthly returns)

        # --- Bet-level statistics: win rate, Calmar ratio & odds band breakdown ---
        br = pd.DataFrame(bet_records)
        win_rate = (br['won'].sum() / len(br) * 100.0) if len(br) else 0.0

        # Persist the real bet ledger (used by the ROI bootstrap / yearly audit)
        try:
            if len(br):
                br['profit'] = br['return'] - br['stake']
                br.to_csv('data/backtest_bets_log.csv', index=False)
                logger.info("Bet ledger saved to data/backtest_bets_log.csv (%d bets)",
                            len(br))
        except Exception as e:
            logger.warning(f"Could not save bet ledger: {e}")

        calmar = 0.0
        if len(eq_df) > 1 and max_drawdown < 0 and current_bankroll > 0:
            years = (eq_df['date'].max() - eq_df['date'].min()).days / 365.25
            if years > 0:
                annualized_return = (current_bankroll / initial_bankroll) ** (1.0 / years) - 1.0
                calmar = annualized_return / abs(max_drawdown / 100.0)

        band_rows = []
        if len(br):
            for lo, hi, label in [(1.5, 3.0, '[1.5-3.0]'), (3.0, 5.0, '(3.0-5.0]'),
                                  (5.0, 8.0, '(5.0-8.0]'), (8.0, 999.0, '(8.0+]')]:
                sub = br[(br['odds'] > lo) & (br['odds'] <= hi)]
                if len(sub):
                    inv = sub['stake'].sum()
                    ret = sub['return'].sum()
                    band_rows.append((label, len(sub), inv, ret - inv,
                                      ((ret - inv) / inv * 100.0) if inv > 0 else 0.0))

        logger.info("=======================================================")
        logger.info(" WALK-FORWARD BACKTEST SUMMARY (FRACTIONAL KELLY & SOFTMAX) ")
        logger.info("=======================================================")
        logger.info(f"Total Bets:        {total_bets}")
        logger.info(f"Win Rate:          {win_rate:.2f}%")
        logger.info(f"Final Bankroll:    ${current_bankroll:.2f}")
        logger.info(f"Net PnL:           ${total_profit:.2f} ({initial_bankroll + total_profit - initial_bankroll:+.2f})")
        logger.info(f"Total Invested:    ${total_invested:.2f}")
        logger.info(f"Total Returned:    ${total_returned:.2f}")
        logger.info(f"Overall ROI:       {total_roi:.2f}%")
        logger.info(f"Max Drawdown:      {max_drawdown:.2f}%")
        logger.info(f"Calmar Ratio:      {calmar:.3f}")
        logger.info(f"Sharpe (annual.):  {sharpe:.2f}")
        if len(eq_df) > 1:
            logger.info(f"Equity Peak ROI:   {eq_df['roi_pct'].max():.2f}%")
            logger.info(f"Equity Trough ROI: {eq_df['roi_pct'].min():.2f}%")

        # --- Ranking quality: NDCG@1/3, LogLoss & Brier (model vs market baseline) ---
        if eval_frames:
            ev = pd.concat(eval_frames, ignore_index=True)
            ev = ev.dropna(subset=['true_prob', 'true_market_prob', 'relevance', 'is_win'])
            ndcg1, ndcg3, ndcg1_mkt, ndcg3_mkt = [], [], [], []
            for _, g in ev.groupby('race_id'):
                if len(g) < 2:
                    continue
                rels = g['relevance'].values
                order = np.argsort(-g['true_prob'].values)
                order_mkt = np.argsort(-g['true_market_prob'].values)
                ndcg1.append(_ndcg_at_k(rels[order], 1))
                ndcg3.append(_ndcg_at_k(rels[order], 3))
                ndcg1_mkt.append(_ndcg_at_k(rels[order_mkt], 1))
                ndcg3_mkt.append(_ndcg_at_k(rels[order_mkt], 3))
            p = np.clip(ev['true_prob'].values, 1e-12, 1.0 - 1e-12)
            pm = np.clip(ev['true_market_prob'].values, 1e-12, 1.0 - 1e-12)
            y = ev['is_win'].astype(float).values
            logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            logloss_mkt = float(-np.mean(y * np.log(pm) + (1.0 - y) * np.log(1.0 - pm)))
            brier = float(np.mean((p - y) ** 2))
            brier_mkt = float(np.mean((pm - y) ** 2))

            logger.info("-------------------------------------------------------")
            logger.info(" RANKING QUALITY (out-of-sample, model vs market baseline) ")
            logger.info(f" NDCG@1:          {np.nanmean(ndcg1):.4f}  (market {np.nanmean(ndcg1_mkt):.4f})")
            logger.info(f" NDCG@3:          {np.nanmean(ndcg3):.4f}  (market {np.nanmean(ndcg3_mkt):.4f})")
            logger.info(f" Log-Loss:        {logloss:.5f}  (market {logloss_mkt:.5f})")
            logger.info(f" Brier Score:     {brier:.5f}  (market {brier_mkt:.5f})")

        logger.info("-------------------------------------------------------")
        logger.info(" ODDS BAND BREAKDOWN (ROI)")
        for label, bets, inv, pnl, roi_band in band_rows:
            logger.info(f"   {label:10s} bets={bets:5d} invested=${inv:9.2f} "
                        f"pnl=${pnl:+9.2f} roi={roi_band:+7.2f}%")
        if not band_rows:
            logger.info("   (no bets placed)")
        logger.info("=======================================================\n")
        
        # Save artifacts locally
        try:
            model = models[0]
            model.save_model("data/lgbm_model.txt")
            with open("data/calibrator.pkl", "wb") as f:
                pickle.dump(models[1], f)
            logger.info("Model saved to data/lgbm_model.txt and calibrator.pkl")
        except Exception as e:
            logger.error(f"Save model failed: {e}")

if __name__ == "__main__":
    pipeline = RacingPipeline()
    pipeline.run_walk_forward_backtest()

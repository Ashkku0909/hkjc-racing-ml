"""
Generate data/all_predictions.csv (UI & Discord bot data source)
================================================================
Refreshes the walk-forward out-of-sample prediction cache and exports a
full prediction table joined with the feature store.

Run after scraping + feature engineering:
    python modeling/generate_predictions.py
"""

import logging

import pandas as pd

from modeling.model_training import RacingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRED_CACHE = "data/walk_forward_preds.csv"
FEATURES = "data/model_features.csv"
OUTPUT = "data/all_predictions.csv"


def main():
    # 1. Refresh the walk-forward out-of-sample prediction cache
    logger.info("Refreshing walk-forward prediction cache...")
    RacingPipeline().run_walk_forward_predictions(PRED_CACHE)

    # 2. Join predictions with the full feature store
    preds = pd.read_csv(PRED_CACHE, parse_dates=['race_date'])
    feats = pd.read_csv(FEATURES, parse_dates=['race_date'])
    df = feats.merge(
        preds[['race_id', 'horse_name', 'true_prob', 'true_market_prob',
               'prob_edge', 'expected_value', 'decay_factor']],
        on=['race_id', 'horse_name'], how='left')

    # 3. Derived columns used by app.py / bot
    df['finish_rank'] = pd.to_numeric(
        df['finish_position'].astype(str).str.extract(r'^(\d+)')[0], errors='coerce')
    df['relevance'] = 1
    df['implied_prob'] = 1.0 / df['win_odds']
    df['market_overround'] = df.groupby('race_id')['implied_prob'].transform('sum')
    # pred_score: the raw LightGBM ranking score is not cached; use the
    # calibrated per-race win probability as the ranking proxy
    df['pred_score'] = df['true_prob']
    df['uncalibrated_prob'] = df['true_prob']
    df['calibrated_prob'] = df['true_prob']

    df = df.sort_values(['race_date', 'race_id']).reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    logger.info("Saved %d rows to %s (latest race date: %s)",
                len(df), OUTPUT, df['race_date'].max().date())
    logger.info("Columns: %d", len(df.columns))


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os

PREDICTIONS_FILE = "data/all_predictions.csv"
_df = None

def load_data():
    global _df
    if os.path.exists(PREDICTIONS_FILE):
        _df = pd.read_csv(PREDICTIONS_FILE)
        print(f"Loaded {len(_df)} rows from {PREDICTIONS_FILE}")
    else:
        print(f"Warning: {PREDICTIONS_FILE} not found. Run model_training.py first.")

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

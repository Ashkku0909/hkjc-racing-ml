import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import os
import pickle
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

class RacingPipeline:
    def __init__(self, features_csv="data/model_features.csv"):
        self.features_csv = features_csv

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

    def convert_to_probabilities(self, df: pd.DataFrame, models: tuple, features: list, scale_factor=6.0) -> pd.DataFrame:
        if len(df) == 0:
            return df
            
        model, calibrator = models
        raw_prob = model.predict(df[features])
        # Calibrate the raw probabilities using LogisticRegression for more reliable scale
        df['raw_prob'] = calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
        
        # Apply massive scale factor to only favor the absolute highest confidence predict
        df['raw_prob'] = df['raw_prob'] ** scale_factor
        
        def normalize_prob(probs):
            return probs / probs.sum()
            
        df['true_prob'] = df.groupby('race_id')['raw_prob'].transform(normalize_prob)
        df = self.apply_power_law(df)
        df['prob_edge'] = df['true_prob'] - df['true_market_prob']
        
        return df

    def simultaneous_kelly(self, group_df, fraction=0.01):
        """Simultaneous Kelly Algorithm for Mutually Exclusive Outcomes (One race)."""
        df = group_df.copy()
        
        df['ev'] = df['true_prob'] * df['win_odds']
        
        # Pure statistical value filtering based on walk-forward analysis
        # Strict rule: Bet ONLY the heaviest, most certain favorites where public edge is clear
        df = df[(df['win_odds'] <= 2.2) & (df['true_prob'] >= 0.50) & (df['ev'] > 1.15)]
        
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
            
        for idx in S:
            f_star = race_df.loc[idx, 'true_prob'] - (R_S / O_S) * (1.0 / race_df.loc[idx, 'win_odds'])
            fractions[idx] = max(f_star * fraction, 0.0)
            
        return fractions

    def run_walk_forward_backtest(self):
        df = self.load_data()
        
        features = [
            'distance', 'barrier_draw', 'weight_carried',
            'last_speed_figure', 'speed_figure_change', 'speed_figure_ewm',
            'horse_win_rate_50', 'horse_place_rate_50', 'horse_win_rate_ewm', 'horse_place_rate_ewm',
            'jockey_win_rate_50', 'trainer_win_rate_50', 'jockey_roi_50', 'trainer_roi_50',
            'weight_diff', 'horse_weight_diff', 'avg_early_pct_last_3', 'avg_late_speed_last_3',
            'course_type', 'track_condition', 'race_class', 'track', 'jockey', 'trainer',
            'days_since_last_race', 'rest_period', 'jockey_trainer_win_rate_50', 'barrier_win_rate_50',
            'jockey_trainer_roi_50', 'is_high_value_combo', 'run_style', 'num_front_runners', 'pace_advantage', 'pace_scenario',
            'race_class_diff', 'distance_diff', 'last_finish_pos', 'career_starts', 'is_unexposed_star'
        ]

        features = [f for f in features if f in df.columns]
        categorical_cols = ['course_type', 'track_condition', 'race_class', 'run_style', 'rest_period', 'pace_scenario', 'track', 'jockey', 'trainer']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                
        if 'horse_weight' in df.columns:
            features.append('horse_weight')
            df['horse_weight'] = df['horse_weight'].fillna(df['horse_weight'].median())

        logger.info("Starting Walk-Forward Backtesting optimizing ROI using Simultaneous Kelly...")
        
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
            
            # Predict and evaluate with a lower scale factor to avoid overconfidence
            test_df = self.convert_to_probabilities(test_df, models, features, scale_factor=1.5)
            
            monthly_invested = 0.0
            monthly_returned = 0.0
            daily_bets = 0
            
            for date, group in test_df.groupby('race_date'):
                if current_bankroll < 10.0: break
                
                # Sizing based on simultaneous kelly solver per race
                for race_id, race_group in group.groupby('race_id'):
                    # Use fractional Kelly of 1.0% for safer sizing
                    fractions = self.simultaneous_kelly(race_group, fraction=0.01)
                    
                    race_invested = 0.0
                    race_returned = 0.0
                    for idx, f in fractions.items():
                        if f > 0:
                            # Hard cap the fraction per bet at 1.0%
                            f = min(f, 0.01) 
                            bet_amt = current_bankroll * f
                            
                            if bet_amt >= 5.0: # Minimum bet threshold
                                race_invested += bet_amt
                                total_bets += 1
                                daily_bets += 1
                                if race_group.loc[idx, 'is_win'] == 1:
                                    race_returned += bet_amt * race_group.loc[idx, 'win_odds']
                    
                    # Update bankroll after each race so it recalculates sizing properly
                    current_bankroll = current_bankroll - race_invested + race_returned
                    monthly_invested += race_invested
                    monthly_returned += race_returned
            
            profit = monthly_returned - monthly_invested
            roi = (profit / monthly_invested * 100) if monthly_invested > 0 else 0
            
            logger.info(f"Month: {current_date.strftime('%Y-%m')} | Bets: {daily_bets} | Invested: ${monthly_invested:.2f} | Profit: ${profit:.2f} | ROI: {roi:.2f}%")
            
            monthly_results.append({
                'Month': current_date.strftime('%Y-%m'),
                'Invested': monthly_invested,
                'Profit': profit,
                'ROI': roi
            })
            
            total_invested += monthly_invested
            total_returned += monthly_returned
            current_date = month_end + pd.DateOffset(days=1)

        total_profit = total_returned - total_invested
        total_roi = (total_profit / total_invested * 100) if total_invested > 0 else 0
        
        logger.info("=======================================================")
        logger.info(" WALK-FORWARD BACKTEST SUMMARY (MULTI-BET KELLY & POWER LAW) ")
        logger.info("=======================================================")
        logger.info(f"Total Bets:        {total_bets}")
        logger.info(f"Final Bankroll:    ${current_bankroll:.2f}")
        logger.info(f"Total Invested:    ${total_invested:.2f}")
        logger.info(f"Total Profit:      ${total_profit:.2f}")
        logger.info(f"Overall ROI:       {total_roi:.2f}%")
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

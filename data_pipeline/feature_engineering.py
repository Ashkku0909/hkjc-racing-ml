import pandas as pd
import numpy as np
import logging
import os
import glob
import re

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Forgive-Flag Keyword Dictionaries (English + Chinese) ---
BLOCKED_KEYWORDS = [
    "held up", "denied clear run", "checked", "severely checked",
    "bumped heavily", "blocked", "受阻", "勒避", "受困"
]
WIDE_NO_COVER_KEYWORDS = [
    "raced wide without cover", "wide throughout", "without cover",
    "全程走外疊", "走外疊", "無遮擋"
]
BAD_START_KEYWORDS = [
    "slow to begin", "bounded in the air", "knuckled", "slow into stride",
    "出閘緩慢", "慢閘", "閘內跪低"
]

# --- Deterministic HKJC Apprentice Claim Imputation ---
# Official ladder (wins-to-date at race time):
#   0-19 wins  -> 10 lb claim
#   20-44 wins -> 7 lb
#   45-69 wins -> 5 lb
#   70-94 wins -> 3 lb
#   95+ wins   -> graduated (0)
# APPRENTICE_PRIOR_WINS holds each apprentice's career HK wins before the
# data window (2020-01-01), so the ladder is seeded correctly for riders
# whose claim milestone was reached before our CSV history begins.
APPRENTICE_PRIOR_WINS = {
    'C L Chau': 5,      # debut 2019/20 season
    'M F Poon': 45,     # claimed 5 lb entering 2020, graduated later
    'K H Chan': 24,     # A K Chan: 7 lb claim entering 2020
    'H T Mo': 40,       # 5 lb claim entering 2020
    'Y L Chung': 0,     # Angus Chung: debut 2022/23
    'E C W Wong': 0,    # Ellis Wong: debut 2022/23
    'P N Wong': 0,      # debut 2024/25
    'H N Wong': 28,     # 7 lb claim entering 2020
    'C Wong': 30,       # Victor Wong: 7 lb claim entering 2020
}


def claim_from_wins(wins: float) -> int:
    """HKJC apprentice claim ladder as a pure function of career wins."""
    if wins < 20:
        return 10
    if wins < 45:
        return 7
    if wins < 70:
        return 5
    if wins < 95:
        return 3
    return 0


# Typical handicap band width per class (used for implied-rating fallback).
CLASS_BAND_WIDTHS = {
    'class 1': 15, 'class 2': 20, 'class 3': 20,
    'class 4': 20, 'class 5': 15,
}
# Typical class ceiling when the scraped band is unavailable.
CLASS_FALLBACK_MAX = {
    'class 1': 105, 'class 2': 100, 'class 3': 80,
    'class 4': 60, 'class 5': 40,
}

def _text_contains_flags(text_series: pd.Series, keywords: list) -> pd.Series:
    """Vectorized keyword scan (case-insensitive) over a text series."""
    pattern = '|'.join(re.escape(k) for k in keywords)
    return text_series.str.contains(pattern, case=False, na=False, regex=True)

class FeatureEngineer:
    def __init__(self, csv_dir: str = "data/raw_csvs"):
        self.csv_dir = csv_dir

    def load_data(self) -> pd.DataFrame:
        """Extracts data from the raw CSV files."""
        logger.info(f"Loading data from CSVs in {self.csv_dir}...")
        
        all_files = glob.glob(os.path.join(self.csv_dir, "*.csv"))
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_dir}")
            
        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # --- Data Cleaning (The Quirks) ---
        logger.info("Cleaning data...")
        
        # 1. Drop Withdrawn (WV) horses and other non-numeric finishes
        df = df[df['finish_position'].apply(lambda x: str(x).split(' ')[0].isnumeric())]
        
        # 2. Ensure correct data types and fix floating point artifacts
        df['race_date'] = pd.to_datetime(df['race_date'])
        df['finishing_time'] = pd.to_numeric(df['finishing_time'], errors='coerce').round(2)
        df['weight_carried'] = pd.to_numeric(df['weight_carried'], errors='coerce')
        df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
        
        # Parse sectional times if available
        if 'sec1_time' in df.columns:
            df['sec1_time'] = pd.to_numeric(df['sec1_time'], errors='coerce')
        else:
            df['sec1_time'] = np.nan

        # NEW: Official ratings, gear, jockey allowance & incident report.
        # Old CSVs may lack these columns entirely -> fall back to defaults.
        for col in ['horse_rating', 'jockey_allowance', 'class_max_rating']:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'gear' not in df.columns:
            df['gear'] = ''
        df['gear'] = df['gear'].fillna('').astype(str)
        if 'incident_report' not in df.columns:
            df['incident_report'] = ''
        df['incident_report'] = df['incident_report'].fillna('').astype(str)
        
        # 3. Create a unique race_id
        # Creates a unique ID like "2020-01-01_Race1"
        if 'race_number' in df.columns:
            df['race_id'] = df['race_date'].dt.strftime('%Y-%m-%d') + "_Race" + df['race_number'].astype(str)
        else:
            # Fallback if race_number is missing (e.g. old scraped data)
            df['race_id'] = df['race_date'].dt.strftime('%Y-%m-%d') + "_" + df['track'] + "_" + df['distance'].astype(str) + "_" + df['race_class'].astype(str)
        
        # Create unique horse_id
        # Use the official HKJC horse_code if available, otherwise fallback to name
        if 'horse_code' in df.columns:
            # Fill missing codes with name just in case
            df['horse_code'] = df['horse_code'].fillna(df['horse_name'])
            df['horse_id'] = df['horse_code'].astype('category').cat.codes
        else:
            df['horse_id'] = df['horse_name'].astype('category').cat.codes
            
        # Ensure new columns exist
        if 'course_type' not in df.columns:
            df['course_type'] = 'Unknown'
        if 'track_condition' not in df.columns:
            df['track_condition'] = 'Unknown'
        if 'horse_weight' not in df.columns:
            df['horse_weight'] = df['weight_carried'] * 10 # Fallback if missing
        if 'running_position' not in df.columns:
            df['running_position'] = None
            
        # Fill missing values
        df['course_type'] = df['course_type'].fillna('Unknown')
        df['track_condition'] = df['track_condition'].fillna('Unknown')
        df['horse_weight'] = df['horse_weight'].fillna(df['weight_carried'] * 10)
        
        # Sort chronologically
        df = df.sort_values(by=['race_date', 'race_id']).reset_index(drop=True)
        
        # Create binary win column (handling '1' and '1 DH' but ignoring '10', '11', etc.)
        df['is_win'] = df['finish_position'].astype(str).str.match(r'^1(\s|$)').astype(int)
        
        return df

    def calculate_claim_and_implied_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic imputation of apprentice jockey claims and implied ratings.

        jockey_allowance (stored POSITIVE, e.g. 7 = 7 lb claim) is reconstructed
        in precedence order:
          1. Inline claim in the jockey cell, e.g. 'M F Poon(-5)'.
          2. HKJC claim ladder from the jockey's wins-to-date (prior wins offset
             + wins counted inside our own results, strictly .shift(1)).
          3. Any previously scraped (absolute) value.

        declared_weight  = weight_carried + jockey_allowance  (official handicap)
        effective_carried_weight = weight_carried (HKJC results are already net
        of apprentice claims).

        implied_rating maps carried weight inside [115, 135] lb linearly onto
        the race's class band [class_min, class_max] and is used ONLY where the
        scraped horse_rating is missing. It uses current-race static conditions
        only -> zero look-ahead leakage.
        """
        logger.info("Imputing jockey allowances (claim ladder) & implied ratings...")
        df = df.sort_values(['race_date', 'race_id']).reset_index(drop=True)

        # 1. Inline claim regex: 'P N Wong(-7)' / 'C L Chau (-2)'
        inline = df['jockey'].astype(str).str.extract(r'\([^\d]*?(\d{1,2})[^\d]*?\)')[0]
        inline = pd.to_numeric(inline, errors='coerce').fillna(0)

        # 2. Ladder claim from career wins-to-date (prior offset + data wins)
        wins_to_date = (
            df.groupby('jockey')['is_win']
              .transform(lambda x: x.shift(1).cumsum().fillna(0))
        )
        offset = df['jockey'].map(APPRENTICE_PRIOR_WINS)
        is_apprentice = offset.notna()
        total_wins = wins_to_date + offset.fillna(0)
        ladder_claim = total_wins.apply(claim_from_wins).where(is_apprentice, 0)

        # 3. Scraped allowance (may use negative convention, e.g. -5)
        scraped = pd.to_numeric(df['jockey_allowance'], errors='coerce').abs().fillna(0)

        claim = inline.where(inline > 0, ladder_claim)
        claim = claim.where(claim > 0, scraped)
        df['jockey_allowance'] = claim.astype(float)

        # 4. Weight bookkeeping (allowance stored positive -> add to get declared)
        df['declared_weight'] = pd.to_numeric(df['weight_carried'], errors='coerce') + df['jockey_allowance']
        df['effective_carried_weight'] = pd.to_numeric(df['weight_carried'], errors='coerce')

        # 5. Implied handicap rating for rows missing an official rating
        cls = df['race_class'].astype(str).str.lower()
        cmax = pd.to_numeric(df['class_max_rating'], errors='coerce')
        fallback_max = cls.map(CLASS_FALLBACK_MAX)
        cmax = cmax.where(cmax > 0, fallback_max)
        band_width = cls.map(CLASS_BAND_WIDTHS).fillna(20)
        cmin = cmax - band_width

        w = pd.to_numeric(df['weight_carried'], errors='coerce')
        frac = ((w - 115.0) / (135.0 - 115.0)).clip(0, 1)
        df['implied_rating'] = (cmin + frac * (cmax - cmin)).round(1)

        return df

    def calculate_speed_figures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Track Variant and Normalized Speed Figures without look-ahead bias."""
        logger.info("Calculating Speed Figures...")

        group_cols = ['track', 'distance', 'race_class']
        if 'course_type' in df.columns and 'track_condition' in df.columns:     
            group_cols.extend(['course_type', 'track_condition'])

        # Ensure sorted chronologically to avoid future leakage
        df = df.sort_values(by=['race_date', 'race_id'])

        # Calculate expanding mean and std so we only use past data
        # We need to group by track etc. and get expanding mean of finishing_time
        # Since pandas expanding.mean() is index-based, we sort by race_date first
        
        # Calculate moving average and std for the track variant up to the previous race.
        # shifting 1 ensures that the current race itself isn't in the expanding mean
        var_df = df.dropna(subset=['finishing_time'])
        
        df['track_variant_mean'] = var_df.groupby(group_cols)['finishing_time'].transform(lambda x: x.shift(1).expanding().mean())
        df['track_variant_std'] = var_df.groupby(group_cols)['finishing_time'].transform(lambda x: x.shift(1).expanding().std())

        # Fill NaNs where expanding history is too short with global medians just for stability
        df['track_variant_mean'] = df['track_variant_mean'].fillna(df['finishing_time'].median())
        df['track_variant_std'] = df['track_variant_std'].fillna(1.0) # default std

        epsilon = 1e-6
        df['speed_figure'] = (df['track_variant_mean'] - df['finishing_time']) / (df['track_variant_std'] + epsilon)

        # Winsorize to prevent explosion when the expanding variance is near zero
        # (early samples with 1-2 past runs could produce figures up to ~600k)
        df['speed_figure'] = np.clip(df['speed_figure'], -4.5, 4.5)

        # Shift the speed figure so it represents the horse's PREVIOUS race speed figure
        df['past_speed_figure'] = df.groupby('horse_id')['speed_figure'].shift(1)
        
        # Fill NaNs (first race for a horse) with 0 (average speed)
        df['past_speed_figure'] = df['past_speed_figure'].fillna(0)
        
        return df

    def calculate_rolling_win_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates past 50-race win percentages without data leakage."""
        logger.info("Calculating Rolling Win Rates (No Data Leakage)...")
        
        # CRITICAL: Ensure strict chronological sorting to prevent look-ahead bias
        df = df.sort_values(by=['race_date', 'race_id'])
        
        # Create is_place column
        try:
            # Handle forms like '1', '1 DH', '10'
            extracted_pos = df['finish_position'].astype(str).str.extract(r'^(\d+)')[0]
            df['is_place'] = (extracted_pos.fillna('99').astype(int) <= 3).astype(int)
        except Exception:
            df['is_place'] = df['is_win'] # fallback
            
        group_col = 'horse_code' if 'horse_code' in df.columns else 'horse_id'
        
        # Horse Rolling 50 Win & Place Rate
        df['horse_win_rate_50'] = (
            df.groupby(group_col)['is_win']
            .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).mean())
        )
        df['horse_place_rate_50'] = (
            df.groupby(group_col)['is_place']
            .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).mean())
        )

        # NEW: Horse Exponentially Weighted Win & Place Rate (Heavy weight on recent starts)
        df['horse_win_rate_ewm'] = (
            df.groupby(group_col)['is_win']
            .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        )
        df['horse_place_rate_ewm'] = (
            df.groupby(group_col)['is_place']
            .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        )
        
        # Jockey Rolling 50 Win Rate
        # .shift(1) ensures the current race's outcome is NOT included in the rolling calculation
        df['jockey_win_rate_50'] = (
            df.groupby('jockey')['is_win']
            .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).mean())
        )
        
        # Trainer Rolling 50 Win Rate
        df['trainer_win_rate_50'] = (
            df.groupby('trainer')['is_win']
            .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).mean())
        )
        
        # Calculate ROIs
        df['flat_return'] = np.where(df['is_win'] == 1, df['win_odds'], 0)
        
        # Jockey ROI
        j_returns = df.groupby('jockey')['flat_return'].transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).sum())
        j_bets = df.groupby('jockey')['is_win'].transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).count())
        df['jockey_roi_50'] = np.where(j_bets > 0, (j_returns - j_bets) / j_bets, 0)
        
        # Trainer ROI
        t_returns = df.groupby('trainer')['flat_return'].transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).sum())
        t_bets = df.groupby('trainer')['is_win'].transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).count())
        df['trainer_roi_50'] = np.where(t_bets > 0, (t_returns - t_bets) / t_bets, 0)
        
        # Fill NaNs for first-time runners/jockeys/trainers with 0
        df['horse_win_rate_50'] = df['horse_win_rate_50'].fillna(0)
        df['horse_place_rate_50'] = df['horse_place_rate_50'].fillna(0)
        df['horse_win_rate_ewm'] = df['horse_win_rate_ewm'].fillna(0)
        df['horse_place_rate_ewm'] = df['horse_place_rate_ewm'].fillna(0)
        df['jockey_win_rate_50'] = df['jockey_win_rate_50'].fillna(0)
        df['trainer_win_rate_50'] = df['trainer_win_rate_50'].fillna(0)
        df['jockey_roi_50'] = df['jockey_roi_50'].fillna(0)
        df['trainer_roi_50'] = df['trainer_roi_50'].fillna(0)
        
        return df

    def calculate_weight_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the change in weight carried and horse weight from the previous race."""
        logger.info("Calculating Race-to-Race Weight Differentials...")
        
        # Sort chronologically per horse
        df = df.sort_values(by=['horse_id', 'race_date', 'race_id'])
        
        # Calculate difference from previous race
        df['weight_diff'] = df.groupby('horse_id')['weight_carried'].diff()
        df['horse_weight_diff'] = df.groupby('horse_id')['horse_weight'].diff()
        
        # Fill NaNs (first race for a horse) with 0
        df['weight_diff'] = df['weight_diff'].fillna(0)
        df['horse_weight_diff'] = df['horse_weight_diff'].fillna(0)
        
        return df

    def calculate_running_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses running_position into early_position_pct and late_speed_rating."""
        logger.info("Calculating Running Position Features and Early Speed Standardizations...")

        def parse_positions(pos_str):
            if pd.isna(pos_str) or not isinstance(pos_str, str):
                return np.nan, np.nan

            parts = pos_str.strip().split()
            if len(parts) < 2:
                return np.nan, np.nan

            try:
                positions = [int(p) for p in parts if p.isdigit()]
                if len(positions) < 2:
                    return np.nan, np.nan

                early_pos = positions[0]
                final_pos = positions[-1]
                positions_passed = early_pos - final_pos
                return early_pos, positions_passed
            except ValueError:
                return np.nan, np.nan

        parsed = df['running_position'].apply(parse_positions)
        df['early_position_raw'] = [p[0] for p in parsed]
        df['positions_passed'] = [p[1] for p in parsed]

        field_size = df.groupby('race_id')['horse_id'].transform('count')

        # Early position percentage (1.0 = dead last, 0.0 = leading)
        df['early_position_pct'] = (df['early_position_raw'] - 1) / (field_size - 1).replace(0, 1)

        # Late speed rating: positions passed / field size
        df['late_speed_rating'] = df['positions_passed'] / field_size

        df = df.drop(columns=['early_position_raw', 'positions_passed'])

        df['early_position_pct'] = df['early_position_pct'].fillna(0.5)
        df['late_speed_rating'] = df['late_speed_rating'].fillna(0)
        
        # --- NEW: Standardize Early Speed (sec1_time) ---
        group_cols = ['track', 'distance', 'race_class']
        if 'course_type' in df.columns and 'track_condition' in df.columns:
            group_cols.extend(['course_type', 'track_condition'])
            
        # Ensure chronological sorting
        df = df.sort_values(by=['race_date', 'race_id'])
        
        # We use an expanding window (excluding the current race) to get historical sec1_time mean/std
        var_df = df.dropna(subset=['sec1_time'])
        df['sec1_mean'] = var_df.groupby(group_cols)['sec1_time'].transform(lambda x: x.shift(1).expanding().mean())
        df['sec1_std'] = var_df.groupby(group_cols)['sec1_time'].transform(lambda x: x.shift(1).expanding().std())
        
        # Fill missing values
        df['sec1_mean'] = df['sec1_mean'].fillna(df['sec1_time'].median())
        df['sec1_std'] = df['sec1_std'].fillna(df['sec1_time'].std())
        
        epsilon = 0.1 # Minimum standard deviation fallback to avoid explosions
        # Z-score for early speed. Negative means faster than average
        # Apply a max/min clip to prevent insane outliers where the track has 1 past run
        # Winsorized at +/-4.5 standard deviations (matching speed_figure winsorization)
        df['standardized_early_speed'] = ((df['sec1_time'] - df['sec1_mean']) / df['sec1_std'].clip(lower=epsilon)).clip(-4.5, 4.5)

        # For horses where sec1_time was completely missing, fill with 0 (average)
        df['standardized_early_speed'] = df['standardized_early_speed'].fillna(0)

        # Shift the standardized speed so it represents the horse's PREVIOUS standardized early speed
        df['past_standardized_early_speed'] = df.groupby('horse_id')['standardized_early_speed'].shift(1).fillna(0)

        return df

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced features like days since last race, jockey/trainer combo, and track bias."""
        logger.info("Calculating Advanced Features (Days Since Last Race, Jockey/Trainer Combo, Track Bias)...")
        
        # 1. Days Since Last Race
        # Sort chronologically per horse
        df = df.sort_values(by=['horse_id', 'race_date', 'race_id'])
        df['days_since_last_race'] = df.groupby('horse_id')['race_date'].diff().dt.days
        # Fill NaNs (first-time starters) with a large number, e.g., 365 days
        df['days_since_last_race'] = df['days_since_last_race'].fillna(365)
        
        # --- NEW: Categorical Rest Period (Fatigue vs Fitness) ---
        def categorize_rest(days):
            if days <= 7: return 'Short_Backup'
            elif days <= 14: return 'Quick_Return'
            elif days <= 30: return 'Optimal'
            elif days <= 60: return 'Freshened'
            elif days <= 120: return 'Spell'
            else: return 'Long_Layoff'
            
        df['rest_period'] = df['days_since_last_race'].apply(categorize_rest)

        # 2. Jockey/Trainer Combination Win Rate & ROI
        # Sort chronologically overall
        df = df.sort_values(by=['race_date', 'race_id'])
        df['jockey_trainer'] = df['jockey'] + "_" + df['trainer']
        
        # Win Rate
        df['jockey_trainer_win_rate_50'] = (
            df.groupby('jockey_trainer')['is_win']
            .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).mean())
        )
        df['jockey_trainer_win_rate_50'] = df['jockey_trainer_win_rate_50'].fillna(0)
        
        # --- NEW: Jockey/Trainer Combo ROI (Last 50 races) ---
        # Calculate the return for a flat $1 bet on this horse
        # If it wins, return is win_odds. If it loses, return is 0.
        df['flat_return'] = np.where(df['is_win'] == 1, df['win_odds'], 0)
        
        # Calculate rolling sum of returns and rolling count of bets
        rolling_returns = df.groupby('jockey_trainer')['flat_return'].transform(
            lambda x: x.shift(1).rolling(window=50, min_periods=1).sum()
        )
        rolling_bets = df.groupby('jockey_trainer')['is_win'].transform(
            lambda x: x.shift(1).rolling(window=50, min_periods=1).count()
        )
        
        # ROI = (Total Returns - Total Invested) / Total Invested
        # Total Invested is just the number of bets (since we assume $1 flat bets)
        df['jockey_trainer_roi_50'] = np.where(
            rolling_bets > 0,
            (rolling_returns - rolling_bets) / rolling_bets,
            0
        )
        df['jockey_trainer_roi_50'] = df['jockey_trainer_roi_50'].fillna(0)
        
        # --- NEW: Under-the-Radar Jockey/Trainer Value Flag ---
        # Find combos that have a low overall public win_rate but a very high ROI
        # Meaning the public constantly underestimates them
        df['is_high_value_combo'] = ((df['jockey_trainer_win_rate_50'] < 0.12) & (df['jockey_trainer_roi_50'] > 0.15)).astype(int)
        
        # 3. Track Bias (Barrier Win Rate at specific Track/Distance)
        if 'barrier_draw' in df.columns:
            df['track_dist_barrier'] = df['track'] + "_" + df['distance'].astype(str) + "_" + df['barrier_draw'].astype(str)
            df['barrier_win_rate_50'] = (
                df.groupby('track_dist_barrier')['is_win']
                .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).mean())
            )
            # Fill NaNs with average win rate (e.g., 1/12 = 0.083)
            df['barrier_win_rate_50'] = df['barrier_win_rate_50'].fillna(0.083)
            
        # --- NEW: Pace Mapping (Run Style & Pace Pressure) ---
        logger.info("Calculating Pace Mapping Features...")
        
# NOTE: Run style should be calculated in apply_lag_features using historical data to avoid leakage.
        
        return df

    def calculate_rating_and_gear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rating benchmark, gear change and effective carried weight features."""
        logger.info("Calculating Rating / Gear / Effective Weight Features...")

        group_col = 'horse_code' if 'horse_code' in df.columns else 'horse_id'
        df = df.sort_values(by=[group_col, 'race_date', 'race_id'])

        # Official rating where available; implied handicap rating otherwise
        # (deterministic fallback built from carried weight + class band).
        official = pd.to_numeric(df['horse_rating'], errors='coerce').fillna(0)
        implied = pd.to_numeric(df.get('implied_rating'), errors='coerce').fillna(0)
        rating_used = official.where(official > 0, implied)

        # 1. rating_diff_vs_class_max: horse rating minus top benchmark of the race class.
        #    Prefer the scraped class band; fall back to the max rating among runners.
        if 'class_max_rating' in df.columns:
            class_max = pd.to_numeric(df['class_max_rating'], errors='coerce')
            class_max = class_max.where(class_max > 0)
        else:
            class_max = pd.Series(np.nan, index=df.index)
        race_benchmark = class_max.fillna(rating_used.groupby(df['race_id']).transform('max'))
        df['rating_diff_vs_class_max'] = rating_used - race_benchmark

        # 2. rating_change: rating delta vs the horse's previous start (lag, not scraped form card)
        df['rating_change'] = rating_used.groupby(df[group_col]).diff().fillna(0)

        # 3. gear_change_flag: 1 if current gear differs from previous race's gear
        prev_gear = df.groupby(group_col)['gear'].shift(1)
        df['gear_change_flag'] = ((df['gear'] != prev_gear) & prev_gear.notna()).astype(int)

        # 4. first_time_blinkers: wears 'B' now AND never wore 'B' in any prior start
        wears_blinkers = df['gear'].str.contains('B', case=False, na=False).astype(int)
        prior_blinker_runs = (
            wears_blinkers.groupby(df[group_col])
            .transform(lambda x: x.shift(1).cumsum().fillna(0))
        )
        df['first_time_blinkers'] = ((wears_blinkers == 1) & (prior_blinker_runs == 0)).astype(int)

        # 5. effective_carried_weight: HKJC results are already net of apprentice
        #    claims, so the effective carried weight IS weight_carried. The separate
        #    declared_weight (weight + claim) is computed in the imputation step.
        df['effective_carried_weight'] = pd.to_numeric(df['weight_carried'], errors='coerce')

        return df

    def calculate_forgive_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Builds rule-based 'forgive' flags from the previous race's stewards' incident text.

        All flags are strictly shifted by one race (.shift(1)) so is_forgive_run is
        derived exclusively from past starts (no look-ahead).
        """
        logger.info("Calculating Forgive Flags from Racing Incident Reports...")

        group_col = 'horse_code' if 'horse_code' in df.columns else 'horse_id'
        df = df.sort_values(by=[group_col, 'race_date', 'race_id'])

        text = df['incident_report'].fillna('').astype(str)

        blocked = _text_contains_flags(text, BLOCKED_KEYWORDS)
        wide = _text_contains_flags(text, WIDE_NO_COVER_KEYWORDS)
        bad_start = _text_contains_flags(text, BAD_START_KEYWORDS)

        # Shift so the flag describes what happened in the horse's LAST race
        df['last_race_blocked'] = blocked.groupby(df[group_col]).transform(lambda x: x.shift(1).fillna(False)).astype(int)
        df['last_race_wide_no_cover'] = wide.groupby(df[group_col]).transform(lambda x: x.shift(1).fillna(False)).astype(int)
        df['last_race_bad_start'] = bad_start.groupby(df[group_col]).transform(lambda x: x.shift(1).fillna(False)).astype(int)

        df['is_forgive_run'] = (
            df[['last_race_blocked', 'last_race_wide_no_cover', 'last_race_bad_start']].sum(axis=1) > 0
        ).astype(int)

        return df

    # --- Trainer Intent / Trackwork Feature Engine ---
    TRACKWORK_FILE = "data/trackwork.csv"
    TRIALS_FILE = "data/trials.csv"
    TRAINER_SURVIVAL_TARGET = 16  # HKJC license benchmark wins per season

    def calculate_trainer_intent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trainer survival & seasonality motivation features (all strictly lag-shifted)."""
        logger.info("Calculating Trainer Intent & Seasonality Features...")
        df = df.sort_values(by=['race_date', 'race_id'])

        # HKJC season runs Sep 1 - Jul 31 (season id = calendar year of the season start)
        df['season'] = np.where(df['race_date'].dt.month >= 9,
                                df['race_date'].dt.year,
                                df['race_date'].dt.year - 1)

        # Cumulative PRIOR wins per trainer per season (strict .shift(1) -> no look-ahead)
        df['trainer_season_wins_to_date'] = (
            df.groupby(['trainer', 'season'])['is_win']
              .transform(lambda x: x.shift(1).cumsum())
              .fillna(0)
        )

        # Remaining wins to hit the official survival benchmark
        df['trainer_quota_gap'] = np.maximum(
            0, self.TRAINER_SURVIVAL_TARGET - df['trainer_season_wins_to_date'])

        # Urgency = max(0, Target - Wins) / (remaining weeks in season + 1)
        end_year = np.where(df['race_date'].dt.month >= 9,
                            df['race_date'].dt.year + 1,
                            df['race_date'].dt.year)
        season_end = pd.to_datetime({'year': end_year, 'month': 7, 'day': 31})
        weeks_left = ((season_end - df['race_date']).dt.days / 7.0).clip(lower=0)
        df['trainer_urgency_index'] = df['trainer_quota_gap'] / (weeks_left + 1)

        # Title contender: trainer among the day's top-3 by season wins, in May-Jul only
        day_rank = df.groupby('race_date')['trainer_season_wins_to_date'].rank(
            method='dense', ascending=False)
        df['trainer_is_title_contender'] = (
            (day_rank <= 3) & (df['race_date'].dt.month.isin([5, 6, 7]))
        ).astype(int)

        df = df.drop(columns=['season'])
        return df

    def _load_events(self, path: str, cols: list):
        """Loads an optional event CSV (trackwork / trials); returns None if absent."""
        if not os.path.exists(path):
            return None
        try:
            ev = pd.read_csv(path)
            for c in cols:
                if c not in ev.columns:
                    ev[c] = None
            return ev[cols].copy()
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
            return None

    def calculate_trackwork_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Jockey commitment & fitness features from trackwork / trial CSVs.

        All windows are computed strictly on events BEFORE the race day (no
        look-ahead). When the CSVs are missing (historical data), every feature
        defaults to a neutral 0.
        """
        logger.info("Calculating Trackwork & Trial Intent Features...")
        group_col = 'horse_code' if 'horse_code' in df.columns else 'horse_id'
        zero_cols = ['jockey_rode_trackwork_count_14d', 'is_jockey_exclusive_worker',
                     'fast_gallop_count_14d', 'swim_count_14d', 'trackwork_time_zscore',
                     'trial_won_before_race']
        for c in zero_cols:
            df[c] = 0

        tw = self._load_events(self.TRACKWORK_FILE,
                               ['horse_code', 'activity_date', 'work_type', 'rider', 'time_400'])
        trials = self._load_events(self.TRIALS_FILE,
                                   ['horse_code', 'trial_date', 'finish_pos', 'margin_behind_leader'])
        if tw is None and trials is None:
            logger.info("No trackwork/trials CSVs found - intent features default to 0.")
            return df

        work = df.sort_values(['race_date', 'race_id'])
        w_codes = work[group_col].astype(str)
        w_jocks = work['jockey'].astype(str).str.upper()
        w_dates = work['race_date']

        if tw is not None and len(tw):
            tw = tw.copy()
            tw['activity_date'] = pd.to_datetime(tw['activity_date'], errors='coerce')
            tw['time_400'] = pd.to_numeric(tw['time_400'], errors='coerce')
            tw['is_fast'] = tw['work_type'].fillna('').astype(str).str.contains(
                'gallop|fast', case=False, na=False).astype(int)
            tw['is_swim'] = tw['work_type'].fillna('').astype(str).str.contains(
                'swim', case=False, na=False).astype(int)
            global_mean = tw['time_400'].mean()
            global_std = tw['time_400'].std()
            if pd.isna(global_std) or global_std == 0:
                global_std = 1.0

            for code, ev in tw.groupby('horse_code'):
                ev = ev.sort_values('activity_date')
                ev_dates = ev['activity_date'].values
                rows = work.index[w_codes == str(code)]
                for ri in rows:
                    rd = w_dates.iloc[ri]
                    lo14 = pd.Timestamp(rd) - pd.Timedelta(days=14)
                    sel = (ev_dates >= lo14) & (ev_dates < pd.Timestamp(rd))
                    if not sel.any():
                        continue
                    sub = ev[sel]
                    work.at[ri, 'fast_gallop_count_14d'] = sub['is_fast'].sum()
                    work.at[ri, 'swim_count_14d'] = sub['is_swim'].sum()
                    work.at[ri, 'jockey_rode_trackwork_count_14d'] = (
                        sub['rider'].fillna('').astype(str).str.upper() == w_jocks.iloc[ri]
                    ).sum()
                    t400 = sub['time_400'].dropna()
                    if len(t400):
                        # Faster (lower) time -> higher positive z-score
                        work.at[ri, 'trackwork_time_zscore'] = (
                            (global_mean - t400.min()) / global_std).clip(-4.5, 4.5)
                    lo21 = pd.Timestamp(rd) - pd.Timedelta(days=21)
                    fast21 = ev[(ev_dates >= lo21) & (ev_dates < pd.Timestamp(rd))
                                & (ev['is_fast'].values == 1)]
                    if len(fast21):
                        frac = (fast21['rider'].fillna('').astype(str).str.upper()
                                == w_jocks.iloc[ri]).mean()
                        work.at[ri, 'is_jockey_exclusive_worker'] = int(frac >= 0.7)

        if trials is not None and len(trials):
            trials = trials.copy()
            trials['trial_date'] = pd.to_datetime(trials['trial_date'], errors='coerce')
            trials['finish_pos'] = pd.to_numeric(trials['finish_pos'], errors='coerce')
            trials['margin_behind_leader'] = pd.to_numeric(trials['margin_behind_leader'], errors='coerce')
            for code, ev in trials.groupby('horse_code'):
                ev = ev.sort_values('trial_date')
                ev_dates = ev['trial_date'].values
                rows = work.index[w_codes == str(code)]
                for ri in rows:
                    rd = w_dates.iloc[ri]
                    lo28 = pd.Timestamp(rd) - pd.Timedelta(days=28)
                    sel = (ev_dates >= lo28) & (ev_dates < pd.Timestamp(rd))
                    if not sel.any():
                        continue
                    sub = ev[sel]
                    work.at[ri, 'trial_won_before_race'] = int(
                        ((sub['finish_pos'] == 1)
                         | (sub['margin_behind_leader'].fillna(np.inf) <= 1.0)).any())

        # Align back onto the original frame (by index)
        df[zero_cols] = work[zero_cols]
        return df

    def apply_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shifts features to prevent data leakage (look-ahead bias)."""
        logger.info("Applying lag features to prevent data leakage...")
        
        # Use horse_code if available, otherwise horse_id
        group_col = 'horse_code' if 'horse_code' in df.columns else 'horse_id'
        
        # First, ensure the data is strictly sorted chronologically!
        df = df.sort_values(by=[group_col, 'race_date', 'race_id'])

        # Calculate Career Starts for the horse (Cumulative count of previous races)
        df['career_starts'] = df.groupby(group_col).cumcount()

        logger.info("Applying lag features: speed figure...")
        # 1. Shift the Speed Figure (What was the horse's speed figure in its LAST race?)
        df['last_speed_figure'] = df.groupby(group_col)['speed_figure'].shift(1)
        
        logger.info("Applying lag features: speed figure ewm...")
        # Exponentially Weighted Speed Figure (More weight to recent speed figures)
        df['speed_figure_ewm'] = (
            df.groupby(group_col)['speed_figure']
            .transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
        )

        logger.info("Applying lag features: diffs...")
        # Speed figure trend: Difference between last race and the race before that
        df['speed_figure_change'] = df['last_speed_figure'] - df.groupby(group_col)['speed_figure'].shift(2)
        df['speed_figure_change'] = df['speed_figure_change'].fillna(0)
        
        # 1b. Track class changes and distance changes
        # Ensure race_class and distance are numeric/convertible
        try:
            # Drop letters from race class if any, default to 4
            num_class = df['race_class'].astype(str).str.extract(r'(\d+)')[0].fillna('4').astype(float)
            df['race_class_diff'] = num_class - df.groupby(group_col)[num_class.name].shift(1)
            df['race_class_diff'] = df['race_class_diff'].fillna(0)
        except Exception:
            df['race_class_diff'] = 0
            
        df['distance_diff'] = df['distance'] - df.groupby(group_col)['distance'].shift(1)
        df['distance_diff'] = df['distance_diff'].fillna(0)
        
        # 1c. Last Finish Position
        try:
            num_pos = df['finish_position'].astype(str).str.extract(r'^(\d+)')[0].fillna('99').astype(float)
            df['last_finish_pos'] = df.groupby(group_col)[num_pos.name].shift(1)
            df['last_finish_pos'] = df['last_finish_pos'].fillna(99) # fallback for no history
        except Exception:
            df['last_finish_pos'] = 99

        # --- Rapid Improver / Unexposed Star Flag ---
        # Flag horses with few starts (< 10) who WON their last race
        df['is_unexposed_star'] = ((df['career_starts'] < 10) & (df['last_finish_pos'] == 1)).astype(int)

        logger.info("Applying lag features: pace rolling...")
        # 2. Shift the Pace Features (How does this horse NORMALLY run?)
        # Let's get the rolling average of their early position over their last 3 races
        df['avg_early_pct_last_3'] = df.groupby(group_col)['early_position_pct'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )

        df['avg_late_speed_last_3'] = df.groupby(group_col)['late_speed_rating'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        
        logger.info("Applying lag features: run style...")
        # Fill NaNs for first-time runners
        df['last_speed_figure'] = df['last_speed_figure'].fillna(0)
        df['speed_figure_ewm'] = df['speed_figure_ewm'].fillna(0)
        df['avg_early_pct_last_3'] = df['avg_early_pct_last_3'].fillna(0.5)
        df['avg_late_speed_last_3'] = df['avg_late_speed_last_3'].fillna(0)

        # 2b. Define Run Style based on HISTORICAL average early position percentage
        # 0.0 to 0.33 = Leader/Early Speed
        # 0.33 to 0.66 = Stalker
        # 0.66 to 1.0 = Closer
        def categorize_run_style(pct):
            if pd.isna(pct):
                return 'Unknown'
            elif pct <= 0.33:
                return 'Leader'
            elif pct <= 0.66:
                return 'Stalker'
            else:
                return 'Closer'

        df['run_style'] = df['avg_early_pct_last_3'].apply(categorize_run_style)

        logger.info("Applying lag features: pace advantage...")
        # --- NEW: Pace Pressure Index ---
        # Now that we have the historical average early position, we can calculate race shape
        # Sort by race to group horses together
        df = df.sort_values(by=['race_date', 'race_id'])
        
        # Count how many horses in the race have a historical early pct <= 0.33 (Leaders)
        df['is_historical_leader'] = (df['avg_early_pct_last_3'] <= 0.33).astype(int)
        
        # Sum the leaders per race to get the Pace Pressure Index (num_front_runners)
        df['num_front_runners'] = df.groupby('race_id')['is_historical_leader'].transform('sum')
        
        # Vectorized Pace Advantage Feature
        # If there are >= 3 front runners, it's a hot pace, advantage to Closers.
        # If <= 1 front runner, it's a slow pace, advantage to Leaders.
        df['pace_advantage'] = 0.0
        cond_closer_adv = (df['num_front_runners'] >= 3) & (df['run_style'] == 'Closer')
        cond_leader_adv = (df['num_front_runners'] <= 1) & (df['run_style'] == 'Leader')
        cond_leader_disadv = (df['num_front_runners'] >= 3) & (df['run_style'] == 'Leader')
        
        df.loc[cond_closer_adv, 'pace_advantage'] = 1.0
        df.loc[cond_leader_adv, 'pace_advantage'] = 1.0
        df.loc[cond_leader_disadv, 'pace_advantage'] = -1.0
        
        # --- NEW: Categorical Pace Scenario ---
        def categorize_pace(leaders):
            if leaders >= 3: return 'Fast' # Burnout scenario for outside draws
            elif leaders == 2: return 'Normal'
            else: return 'Slow' # Leaders get it easy
            
        df['pace_scenario'] = df['num_front_runners'].apply(categorize_pace)
        
        # Clean up temporary column
        df = df.drop(columns=['is_historical_leader'])
        
        logger.info("Applying lag features: early pressure index...")
        # --- NEW: Early Pressure Index (EPI) ---
        # 1. Take the top 3 fastest early horses (lowest past_standardized_early_speed)
        # 2. Add penalty if they draw wide (barriers 10-14)
        if 'past_standardized_early_speed' in df.columns:
            # Safely coerce to float just in case
            df['past_standardized_early_speed'] = pd.to_numeric(df['past_standardized_early_speed'], errors='coerce').fillna(0)
            df['barrier_draw_num'] = pd.to_numeric(df['barrier_draw'], errors='coerce').fillna(0)
            
            # Vectorized calculate
            penalty = np.where(df['barrier_draw_num'] >= 10, 0.5, 0)
            df['adjusted_early_speed'] = -df['past_standardized_early_speed'] + penalty
            
            # Vectorized top 3 mean computation
            # Sort descending by adjusted early speed, group by race, take top 3, then average
            top3_speeds = df[['race_id', 'adjusted_early_speed']].sort_values(
                ['race_id', 'adjusted_early_speed'], ascending=[True, False]
            ).groupby('race_id').head(3)
            
            epi_map = top3_speeds.groupby('race_id')['adjusted_early_speed'].mean().reset_index(name='early_pressure_index')
            df = df.merge(epi_map, on='race_id', how='left')
            df['early_pressure_index'] = df['early_pressure_index'].fillna(0)
            df = df.drop(columns=['barrier_draw_num'])
        else:
            df['early_pressure_index'] = 0

        return df

    def calculate_race_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intra-race comparative features (grouped by race_id).

        All inputs are either lagged form features (last_speed_figure,
        past_standardized_early_speed, run_style) or current-race static
        conditions (weight_carried) -> zero look-ahead leakage.
        """
        logger.info("Calculating race-context relative features...")

        for col in ['last_speed_figure', 'weight_carried', 'past_standardized_early_speed']:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce')

        g = df.groupby('race_id')

        # Speed relativity (higher speed_figure = faster)
        df['rel_speed_to_race_mean'] = df['last_speed_figure'] - g['last_speed_figure'].transform('mean')
        df['rel_speed_to_race_max'] = df['last_speed_figure'] - g['last_speed_figure'].transform('max')
        df['speed_rank_in_field'] = g['last_speed_figure'].rank(ascending=False, method='dense')

        # Weight advantage (讓磅優勢)
        df['weight_rel_to_top'] = df['weight_carried'] - g['weight_carried'].transform('max')
        df['weight_rel_to_mean'] = df['weight_carried'] - g['weight_carried'].transform('mean')

        # Pace density: share of declared front-runners in the field
        leaders = (df['run_style'] == 'Leader').astype(int)
        field_size = g['horse_id'].transform('count')
        df['race_front_runner_density'] = leaders.groupby(df['race_id']).transform('sum') / field_size

        # Early speed vs the fastest early horse in the field
        df['early_speed_vs_field_fastest'] = (
            df['past_standardized_early_speed'] - g['past_standardized_early_speed'].transform('max')
        )

        return df

    def save_features(self, df: pd.DataFrame):
        """Saves the engineered features to a new CSV file."""
        logger.info("Saving features to CSV file 'data/model_features.csv'...")
        
        # Select only the necessary columns for the feature store
        # We don't have result_id anymore, so we use race_id and horse_id as the composite key
        feature_cols = [
            'race_id', 'horse_id', 'horse_name', 'race_date', 'track', 'distance', 'race_class', 'jockey', 'trainer',
            'finish_position', 'win_odds', 'is_win',
            'speed_figure', 'past_speed_figure', 'last_speed_figure', 'speed_figure_change', 'speed_figure_ewm',
            'horse_win_rate_50', 'horse_place_rate_50', 'horse_win_rate_ewm', 'horse_place_rate_ewm',
            'jockey_win_rate_50', 'trainer_win_rate_50', 'jockey_roi_50', 'trainer_roi_50',
            'weight_diff', 'horse_weight_diff', 'early_position_pct', 'late_speed_rating',
            'avg_early_pct_last_3', 'avg_late_speed_last_3',
            'past_standardized_early_speed', 'early_pressure_index', 
            'days_since_last_race', 'jockey_trainer_win_rate_50', 'jockey_trainer_roi_50',
            'run_style', 'num_front_runners', 'pace_advantage', 'pace_scenario', 'race_class_diff', 'distance_diff', 'last_finish_pos',
            'career_starts', 'is_unexposed_star', 'rest_period', 'is_high_value_combo'
        ]
        
        # Add new columns if they exist
        if 'race_number' in df.columns:
            feature_cols.insert(4, 'race_number')
        if 'horse_code' in df.columns:
            feature_cols.insert(3, 'horse_code')
        if 'course_type' in df.columns:
            feature_cols.insert(6, 'course_type')
        if 'track_condition' in df.columns:
            feature_cols.insert(7, 'track_condition')
        if 'horse_weight' in df.columns:
            feature_cols.append('horse_weight')
        if 'running_position' in df.columns:
            pass # Already included in feature_cols
        if 'finishing_time' in df.columns:
            feature_cols.append('finishing_time')
        if 'barrier_draw' in df.columns:
            feature_cols.append('barrier_draw')
            feature_cols.append('barrier_win_rate_50')
        if 'weight_carried' in df.columns:
            feature_cols.append('weight_carried')
            
        # Add sectional times if they exist
        for i in range(1, 7):
            col = f'sec{i}_time'
            if col in df.columns:
                feature_cols.append(col)

        # NEW: Ratings, gear, allowance, incident text & derived features
        for col in ['horse_rating', 'rating_change', 'jockey_allowance', 'class_max_rating',
                    'gear', 'incident_report', 'rating_diff_vs_class_max', 'gear_change_flag',
                    'first_time_blinkers', 'effective_carried_weight',
                    'last_race_blocked', 'last_race_wide_no_cover', 'last_race_bad_start',
                    'is_forgive_run']:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)

        # NEW: Trainer intent & trackwork features
        for col in ['trainer_season_wins_to_date', 'trainer_quota_gap', 'trainer_urgency_index',
                    'trainer_is_title_contender', 'jockey_rode_trackwork_count_14d',
                    'is_jockey_exclusive_worker', 'fast_gallop_count_14d', 'swim_count_14d',
                    'trackwork_time_zscore', 'trial_won_before_race']:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)

        # NEW: Claim imputation, implied rating & race-context relative features
        for col in ['declared_weight', 'implied_rating',
                    'rel_speed_to_race_mean', 'rel_speed_to_race_max', 'speed_rank_in_field',
                    'weight_rel_to_top', 'weight_rel_to_mean',
                    'race_front_runner_density', 'early_speed_vs_field_fastest']:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)
            
        features_df = df[feature_cols].copy()
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        features_df.to_csv('data/model_features.csv', index=False)
        
        logger.info("Feature engineering complete and saved successfully to data/model_features.csv.")

    def run_pipeline(self):
        """Executes the full feature engineering pipeline."""
        df = self.load_data()
        df = self.calculate_claim_and_implied_rating(df)
        df = self.calculate_speed_figures(df)
        df = self.calculate_rolling_win_rates(df)
        df = self.calculate_weight_differentials(df)
        df = self.calculate_running_position_features(df)
        df = self.calculate_advanced_features(df)
        df = self.calculate_rating_and_gear_features(df)
        df = self.calculate_forgive_flags(df)
        df = self.calculate_trainer_intent_features(df)
        df = self.calculate_trackwork_features(df)
        df = self.apply_lag_features(df)
        df = self.calculate_race_context_features(df)
        self.save_features(df)

if __name__ == "__main__":
    # No DB URL needed anymore, just point to the CSV directory
    engineer = FeatureEngineer(csv_dir="data/raw_csvs")
    engineer.run_pipeline()

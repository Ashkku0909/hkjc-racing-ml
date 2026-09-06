"""
Live Stacker - walk-forward secondary meta-learner over the Bayesian fusion.

Aligns persisted odds snapshots (data/odds_snapshots/*.csv) against official
dividends (data/historical_dividends.csv) and evaluates whether a stacked
LightGBM meta-learner beats the pure time-varying Bayesian fusion.

Feature set per runner (real data only):
  - offline LightGBM score (true_prob / pred_score)
  - final-5-minute log-odds slope
  - late 90s odds velocity (win_velocity_90s)
  - gate draw (barrier_draw), venue, field size
Target: won the race (WIN dividend combo == horse number).

Walk-forward: train on the first 70% of race dates, evaluate the last 30%.
Reports log loss for BOTH the stacker and the Bayesian-fusion baseline and
persists the stacker to data/live_stacker.txt only when it wins.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SNAP_DIR = os.path.join("data", "odds_snapshots")
DIVIDENDS_PATH = "data/historical_dividends.csv"
PREDICTIONS_PATH = "data/all_predictions.csv"
FEATURES_PATH = "data/model_features.csv"
STACKER_OUT = "data/live_stacker.txt"


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_all_snapshots() -> pd.DataFrame:
    """Concatenates every persisted snapshot CSV into one frame."""
    if not os.path.isdir(SNAP_DIR):
        return pd.DataFrame()
    frames = []
    for fn in sorted(os.listdir(SNAP_DIR)):
        if fn.endswith(".csv"):
            try:
                frames.append(pd.read_csv(os.path.join(SNAP_DIR, fn)))
            except Exception as e:
                print(f"skip {fn}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")
    df["place_odds"] = pd.to_numeric(df["place_odds"], errors="coerce")
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce")
    return df


def final_and_t5_frames(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per race: the FINAL jump frame (is_final_jump_price, else last epoch)
    joined with the nearest T-5m frame, yielding the 5-minute odds slope."""
    rows = []
    for race_id, g in snaps.groupby("race_id"):
        g = g.dropna(subset=["epoch", "horse_number"])
        if not len(g):
            continue
        if "is_final_jump_price" in g.columns and g["is_final_jump_price"].any():
            final_epoch = g.loc[g["is_final_jump_price"], "epoch"].max()
        else:
            final_epoch = g["epoch"].max()
        final = g[g["epoch"] == final_epoch]
        # nearest T-5m frame strictly before the final frame
        prior = g[g["epoch"] <= final_epoch - 180.0]
        if not len(prior):
            prior = g[g["epoch"] < final_epoch]
        if not len(prior):
            continue
        t5_epoch = prior.loc[(prior["epoch"] - (final_epoch - 300.0)).abs().idxmin(), "epoch"]
        t5 = g[g["epoch"] == t5_epoch]
        t5_map = {int(r["horse_number"]): r for _, r in t5.iterrows()}
        for _, r in final.iterrows():
            hn = r["horse_number"]
            t5r = t5_map.get(int(hn)) if pd.notna(hn) else None
            ow_f, ow_t = _fnum(r.get("win_odds")), None
            if t5r is not None:
                ow_t = _fnum(t5r.get("win_odds"))
            slope = None
            if ow_f and ow_t and ow_f > 0 and ow_t > 0:
                slope = float(np.log(ow_f) - np.log(ow_t))
            rows.append({
                "race_id": race_id,
                "epoch": final_epoch,
                "horse_number": hn,
                "horse_name": r.get("horse_name"),
                "final_win_odds": ow_f,
                "final_place_odds": _fnum(r.get("place_odds")),
                "odds_slope_5m": slope,
                "win_velocity_90s": _fnum(r.get("win_velocity_90s")),
            })
    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    """Join final-jump frames with dividends, predictions and features."""
    if not os.path.exists(DIVIDENDS_PATH):
        raise FileNotFoundError(f"{DIVIDENDS_PATH} missing - run the scraper first")
    div = pd.read_csv(DIVIDENDS_PATH)
    win = div[div["pool"] == "WIN"].copy()
    win["winner_no"] = pd.to_numeric(win["combo_codes"], errors="coerce")
    win_map = win.dropna(subset=["winner_no"]).set_index("race_id")["winner_no"].to_dict()

    snaps = load_all_snapshots()
    if not len(snaps):
        raise ValueError("no odds snapshots persisted yet - run the live terminal first")
    base = final_and_t5_frames(snaps)
    base["is_win"] = base.apply(
        lambda r: int(r["horse_number"]) == int(win_map[r["race_id"]])
        if r["race_id"] in win_map and pd.notna(r["horse_number"]) else 0, axis=1)
    base["field_size"] = base.groupby("race_id")["horse_number"].transform("size")

    pred = pd.read_csv(PREDICTIONS_PATH, usecols=lambda c: c in
                       {"race_id", "horse_name", "true_prob", "pred_score"})
    pred["_key"] = pred["race_id"].astype(str) + "|" + pred["horse_name"].astype(str).str.upper()
    base["_key"] = base["race_id"].astype(str) + "|" + base["horse_name"].astype(str).str.upper()
    base = base.merge(pred[["_key", "true_prob", "pred_score"]], on="_key", how="left")

    feat = pd.read_csv(FEATURES_PATH, usecols=lambda c: c in
                       {"race_id", "horse_name", "barrier_draw", "run_style"})
    feat["_key"] = feat["race_id"].astype(str) + "|" + feat["horse_name"].astype(str).str.upper()
    feat = feat.drop_duplicates("_key").drop(columns=["race_id", "horse_name"])
    base = base.merge(feat, on="_key", how="left")
    base = base.drop(columns=["_key"])
    base["barrier_draw"] = pd.to_numeric(base["barrier_draw"], errors="coerce")
    base["venue"] = base["race_id"].str.contains("ST").astype(int)
    base["race_date"] = base["race_id"].str.extract(r"^(\d{4}-\d{2}-\d{2})_")[0]
    base = base.dropna(subset=["race_date", "final_win_odds", "is_win"])
    return base


def fusion_baseline(df: pd.DataFrame) -> np.ndarray:
    """Bayesian-fusion proxy at peak sensitivity (w=0.85) with the 90s flow delta.

    logit(P) = 0.15*logit(p_model) + 0.85*logit(p_mkt) + delta_flow, softmax.
    """
    from web_live import bayesian_fusion

    out = np.zeros(len(df))
    for race_id, g in df.groupby("race_id"):
        pm = pd.to_numeric(g["true_prob"], errors="coerce").to_numpy(dtype=float)
        impl = 1.0 / pd.to_numeric(g["final_win_odds"], errors="coerce").to_numpy(dtype=float)
        pk = impl / (impl.sum() + 1e-12)
        d = pd.to_numeric(g["win_velocity_90s"], errors="coerce").to_numpy(dtype=float)
        d = np.where(np.isfinite(d) & (d <= -0.15), 0.10, 0.0)
        out[g.index] = bayesian_fusion(pm, pk, 0.85, d)
    return out


def main() -> None:
    from sklearn.metrics import log_loss
    try:
        df = build_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"[live-stacker] dataset unavailable: {e}")
        return
    print(f"[live-stacker] {len(df)} runners across {df['race_id'].nunique()} races")

    df = df.sort_values("race_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    if len(train) < 50 or len(test) < 50:
        print("[live-stacker] not enough history yet for a walk-forward split - skip")
        return

    feats = ["true_prob", "pred_score", "odds_slope_5m", "win_velocity_90s",
             "barrier_draw", "field_size", "venue"]
    Xtr = train[feats].fillna(-999.0)
    Xte = test[feats].fillna(-999.0)
    ytr = train["is_win"].to_numpy()
    yte = test["is_win"].to_numpy()

    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               num_leaves=15, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    p_stack = model.predict_proba(Xte)[:, 1]
    p_fuse = fusion_baseline(test)
    p_fuse = np.where(np.isfinite(p_fuse), p_fuse, 1e-6)

    ll_stack = log_loss(yte, np.clip(p_stack, 1e-6, 1 - 1e-6))
    ll_fuse = log_loss(yte, np.clip(p_fuse, 1e-6, 1 - 1e-6))
    print(f"[live-stacker] test log-loss  stacker={ll_stack:.4f}  bayes-fusion={ll_fuse:.4f}")
    if ll_stack < ll_fuse:
        model.booster_.save_model(STACKER_OUT)
        print(f"[live-stacker] stacker WINS -> saved {STACKER_OUT}")
    else:
        print("[live-stacker] Bayesian fusion wins - keep the pure engine (no model saved)")


if __name__ == "__main__":
    main()

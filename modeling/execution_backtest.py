"""Zero-Lookahead Execution Backtest (REAL persisted snapshots only)
====================================================================
Dual-track decision/settlement audit of live tote odds:

  Decision leg : strictly the T-3m persisted snapshot odds  -> EV / Kelly /
                 smart-money trigger (simulates the live bet condition).
  Settlement leg: strictly the T-0 official FINAL win dividend from the
                 results store (real payout) -> realized PnL.

Also reports the Odds Slippage Ratio  ΔO = O_final / O_T-3m  (quantifies how
much late money erodes the decision-time edge) and runs a PURE smart-money
signal test (S >= threshold AND odds band, NO pre-race model features) versus
the ~ -18.4% HKJC win-pool takeout baseline.

REAL-DATA ONLY: this module reads data/odds_snapshots/*.csv (written by
scraping/live_scraper.persist_odds_snapshot on real race days) and settles on
data/raw_csvs/*.csv final dividends. It refuses to fabricate snapshots: with
an empty store it prints coverage and exits cleanly.

CLI:
    python -m modeling.execution_backtest coverage
    python -m modeling.execution_backtest slippage [--date 2026-09-09] [--venue ST]
    python -m modeling.execution_backtest smart    [--S 75] [--lo 4.5] [--hi 8.0]
    python -m modeling.execution_backtest execution [--lo 4.5] [--hi 8.0] [--ev-ratio 1.22]
"""
import argparse
import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from modeling.model_training import EDGE_DECAY_C, EDGE_DECAY_GAMMA
from bot.analyzer_service import calculate_smart_money_metrics
from scraping.live_scraper import load_odds_snapshots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_CSV_DIR = "data/raw_csvs"
PREDICTIONS_FILE = "data/all_predictions.csv"
STAKE = 10.0                     # flat ticket per qualifying horse (HK$)
TAKEOUT_BASELINE = -18.4         # HKJC win-pool effective takeout (%) from real overround 1.226
RNG_SEED = 20260905

# Decision snapshot selection: T-3m target and acceptance windows (seconds)
T3M_TARGET = 180.0
T3M_WINDOW = (60.0, 330.0)
T15M_WINDOW = (600.0, 1200.0)


def _decay(odds: pd.Series) -> pd.Series:
    """Favorite-longshot decay (C/O)^gamma on a real odds series."""
    o = odds.replace(0, np.nan)
    return (EDGE_DECAY_C / o) ** EDGE_DECAY_GAMMA


def _bootstrap_roi_ci(profits: np.ndarray, stakes: np.ndarray,
                      b: int = 10000) -> Tuple[float, float, float, float]:
    """Percentile bootstrap CI over the REAL profit vector (iid resample)."""
    n = len(profits)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(b, n))
    roi = profits[idx].sum(1) / stakes[idx].sum(1) * 100.0
    lo, hi = np.percentile(roi, [2.5, 97.5])
    return float(roi.mean()), float(lo), float(hi), float(np.mean(roi <= 0.0))


def load_final_dividends() -> pd.DataFrame:
    """Official final win odds + finish rank from the REAL results store,
    restricted to race days present in the snapshot store."""
    snaps = load_odds_snapshots()
    if len(snaps) == 0:
        return pd.DataFrame()
    days = sorted({str(r).split('_Race')[0] for r in snaps['race_id'].astype(str)})
    frames = []
    for day in days:
        path = os.path.join(RAW_CSV_DIR, f"{day}.csv")
        if not os.path.exists(path):
            continue
        try:
            r = pd.read_csv(path)
        except Exception as e:
            logger.warning("Cannot read %s: %s", path, e)
            continue
        keep = [c for c in ['race_number', 'horse_number', 'horse_name',
                            'win_odds', 'finish_position'] if c in r.columns]
        if not keep:
            continue
        r = r[keep].copy()
        r['race_id'] = day + "_Race" + r['race_number'].astype(int).astype(str)
        r['horse_number'] = pd.to_numeric(r['horse_number'], errors='coerce')
        r['final_odds'] = pd.to_numeric(r['win_odds'], errors='coerce')
        r['finish_rank'] = pd.to_numeric(
            r['finish_position'].astype(str).str.extract(r'^(\d+)')[0], errors='coerce')
        frames.append(r[['race_id', 'horse_number', 'horse_name', 'final_odds',
                         'finish_rank']])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _horse_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One snapshot -> horse-level frame (unique per horse_number)."""
    f = rows.drop_duplicates(subset='horse_number', keep='first').copy()
    f['horse_name'] = f['horse_name'].fillna('')
    f['win_odds'] = pd.to_numeric(f['win_odds'], errors='coerce')
    f['place_odds'] = pd.to_numeric(f['place_odds'], errors='coerce')
    return f


def _select_snapshots(race_snap: pd.DataFrame, target: float,
                      window: Tuple[float, float]) -> Optional[pd.DataFrame]:
    """Picks the snapshot closest to `target` seconds before post within window."""
    lab = race_snap[race_snap['time_to_post'].notna()].copy()
    if len(lab) == 0:
        return None
    t = pd.to_numeric(lab['time_to_post'], errors='coerce')
    in_win = t.between(window[0], window[1])
    if not in_win.any():
        return None
    sel = lab[in_win].iloc[(t[in_win] - target).abs().argmin()]
    return _horse_frame(race_snap[race_snap['epoch'] == sel['epoch']]) \
        if (race_snap['epoch'] == sel['epoch']).any() else _horse_frame(sel.to_frame().T)


def build_decision_panel(final: pd.DataFrame) -> pd.DataFrame:
    """Pairs T-15m baseline / T-3m decision snapshots with final dividends.

    Returns per-horse rows: race_id, horse_number, horse_name, odds_base,
    odds_dec, final_odds, finish_rank, smart_money_score.
    """
    snaps = load_odds_snapshots()
    if len(snaps) == 0 or len(final) == 0:
        return pd.DataFrame()
    snaps['race_id'] = snaps['race_id'].astype(str)
    final['race_id'] = final['race_id'].astype(str)

    rows = []
    for race_id, g in snaps.groupby('race_id'):
        base = _select_snapshots(g, 900.0, T15M_WINDOW)   # ~T-15m
        dec = _select_snapshots(g, T3M_TARGET, T3M_WINDOW)  # ~T-3m
        if base is None or dec is None or len(base) == 0 or len(dec) == 0:
            continue
        # Smart money score strictly from baseline -> decision snapshots
        scored = calculate_smart_money_metrics(dec, base)
        fin = final[final['race_id'] == race_id]
        if len(fin) == 0:
            continue
        merged = scored.merge(
            fin[['horse_number', 'final_odds', 'finish_rank']],
            on='horse_number', how='inner')
        merged['race_id'] = race_id
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    for c in ['win_odds', 'final_odds', 'finish_rank', 'smart_money_score']:
        panel[c] = pd.to_numeric(panel[c], errors='coerce')
    panel = panel.dropna(subset=['win_odds', 'final_odds', 'finish_rank',
                                 'smart_money_score'])
    return panel


def report_coverage() -> None:
    snaps = load_odds_snapshots()
    if len(snaps) == 0:
        print("SNAPSHOT STORE: empty (data/odds_snapshots/).")
        print("Real T-15m/T-3m/T-0 snapshots are collected automatically by the ")
        print("Discord pre-race daemon on HKJC race days (scrape_live_odds + ")
        print("persist_odds_snapshot). No snapshots exist yet for any past date -")
        print("zero-synthetic rule: nothing to backtest until real data accrues.")
        return
    races = snaps['race_id'].nunique()
    days = snaps['race_id'].str[:10].nunique()
    print(f"SNAPSHOT STORE: {len(snaps):,} rows | {races:,} races | {days} days")
    lbl = snaps[snaps['time_to_post'].notna()]
    if len(lbl):
        t = pd.to_numeric(lbl['time_to_post'], errors='coerce')
        print(f"labelled polls: {len(lbl):,} "
              f"(T-15m zone: {(t.between(600,1200)).sum():,} | "
              f"T-3m zone: {t.between(60,330).sum():,} | "
              f"T-0 zone: {(t <= 12).sum():,})")


def report_slippage(panel: pd.DataFrame) -> None:
    if len(panel) == 0:
        print("No decision/final pairs available yet (no real snapshots).")
        return
    panel['slippage'] = panel['final_odds'] / panel['win_odds']
    s = panel['slippage']
    print("\n=== ODDS SLIPPAGE  ΔO = O_final / O_T-3m ===")
    print(f"horses paired: {len(panel):,} | races: {panel['race_id'].nunique():,}")
    print(f"mean={s.mean():.4f}  p25={s.quantile(.25):.4f}  median={s.median():.4f}  "
          f"p75={s.quantile(.75):.4f}")
    print(f"fraction where final < T-3m (steam/erosion): {(s < 1).mean() * 100:.1f}%")
    ero = (panel['final_odds'] - panel['win_odds']) / panel['win_odds']
    print(f"avg odds move: {ero.mean() * 100:+.2f}%  (negative = late money compresses odds)")


def report_smart_money_signal(panel: pd.DataFrame, s_thresh: float,
                              odds_lo: float, odds_hi: float) -> None:
    if len(panel) == 0:
        print("No decision/final pairs available yet (no real snapshots).")
        return
    sel = panel[(panel['smart_money_score'] >= s_thresh)
                & panel['win_odds'].between(odds_lo, odds_hi)].copy()
    if len(sel) == 0:
        print(f"\n=== PURE SMART-MONEY SIGNAL (S>={s_thresh:.0f}, odds "
              f"[{odds_lo:.1f},{odds_hi:.1f}]) ===")
        print("No qualifying real signals yet.")
        return
    stake = np.full(len(sel), STAKE)
    profits = np.where(sel['finish_rank'] == 1,
                       stake * (sel['final_odds'] - 1.0), -stake)
    roi = profits.sum() / stake.sum() * 100
    mean, lo, hi, p0 = _bootstrap_roi_ci(profits, stake)
    print(f"\n=== PURE SMART-MONEY SIGNAL (S>={s_thresh:.0f}, odds "
          f"[{odds_lo:.1f},{odds_hi:.1f}]) ===")
    print(f"signals: {len(sel)} (races {sel['race_id'].nunique()}) | "
          f"wins: {int((sel['finish_rank'] == 1).sum())} "
          f"({(sel['finish_rank'] == 1).mean() * 100:.1f}%)")
    print(f"realized ROI (final dividend): {roi:+.2f}%   "
          f"[vs win-pool takeout baseline {TAKEOUT_BASELINE:+.1f}%]")
    print(f"bootstrap 95% CI: [{lo:+.2f}%, {hi:+.2f}%]  P(ROI<=0)={p0:.3f}")


def report_execution_backtest(panel: pd.DataFrame, odds_lo: float, odds_hi: float,
                              ev_ratio: float) -> None:
    if len(panel) == 0:
        print("No decision/final pairs available yet (no real snapshots).")
        return
    preds = pd.read_csv(PREDICTIONS_FILE, usecols=['race_id', 'horse_name', 'true_prob'])
    preds['race_id'] = preds['race_id'].astype(str)
    m = panel.merge(preds, on=['race_id', 'horse_name'], how='left')
    m = m.dropna(subset=['true_prob'])
    if len(m) == 0:
        print("No model predictions matched snapshot runners (real data only).")
        return

    # Decision at T-3m: decay-adjusted EV uses ONLY the T-3m snapshot odds
    dec = pd.to_numeric(m['win_odds'], errors='coerce')
    ev = m['true_prob'] * dec * _decay(dec)
    qual = m[(dec.between(odds_lo, odds_hi)) & (ev >= ev_ratio)].copy()
    if len(qual) == 0:
        print(f"\n=== EXECUTION (model, odds [{odds_lo:.1f},{odds_hi:.1f}], "
              f"EV>={ev_ratio:.2f}) ===\nNo qualifying decisions yet.")
        return

    stake = np.full(len(qual), STAKE)
    won = (qual['finish_rank'] == 1).to_numpy()
    naive = np.where(won, stake * (qual['win_odds'] - 1.0), -stake)
    realized = np.where(won, stake * (qual['final_odds'] - 1.0), -stake)
    slip = (qual['final_odds'] / qual['win_odds']).to_numpy()

    print(f"\n=== EXECUTION DUAL-TRACK (model, odds [{odds_lo:.1f},{odds_hi:.1f}], "
          f"EV>={ev_ratio:.2f}) ===")
    print(f"decisions: {len(qual)} (races {qual['race_id'].nunique()}) | "
          f"wins: {int(won.sum())} ({won.mean() * 100:.1f}%)")
    print(f"decide-at-T-3m ROI (hypothetical T-3m payout): "
          f"{naive.sum() / stake.sum() * 100:+.2f}%")
    print(f"REALIZED ROI (official final dividend):        "
          f"{realized.sum() / stake.sum() * 100:+.2f}%")
    print(f"edge erosion by slippage: "
          f"{(naive.sum() - realized.sum()) / stake.sum() * 100:+.2f}pp "
          f"(mean ΔO={slip.mean():.4f})")
    mroi, lo, hi, p0 = _bootstrap_roi_ci(realized, stake)
    print(f"realized bootstrap 95% CI: [{lo:+.2f}%, {hi:+.2f}%]  "
          f"P(ROI<=0)={p0:.3f}  [takeout baseline {TAKEOUT_BASELINE:+.1f}%]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-lookahead execution backtest (real snapshots)")
    ap.add_argument('cmd', nargs='?', default='coverage',
                    choices=['coverage', 'slippage', 'smart', 'execution'])
    ap.add_argument('--date', default=None, help='filter snapshots by date (YYYY-MM-DD)')
    ap.add_argument('--venue', default=None, help='filter by venue (ST/HV)')
    ap.add_argument('--S', type=float, default=75.0, help='smart-money score threshold')
    ap.add_argument('--lo', type=float, default=4.5)
    ap.add_argument('--hi', type=float, default=8.0)
    ap.add_argument('--ev-ratio', type=float, default=1.22)
    args = ap.parse_args()

    snaps = load_odds_snapshots(date_str=args.date, venue=args.venue)
    if args.cmd == 'coverage':
        report_coverage()
        return

    final = load_final_dividends()
    panel = build_decision_panel(final)
    logger.info("Decision/final panel: %d rows", len(panel))

    if args.cmd == 'slippage':
        report_slippage(panel)
    elif args.cmd == 'smart':
        report_smart_money_signal(panel, args.S, args.lo, args.hi)
    elif args.cmd == 'execution':
        report_execution_backtest(panel, args.lo, args.hi, args.ev_ratio)


if __name__ == "__main__":
    main()

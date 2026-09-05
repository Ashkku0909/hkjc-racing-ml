import argparse
import logging

import numpy as np
import pandas as pd

from modeling.exotics_pricing import (
    build_core_satellite_bets,
    load_dividend_store,
)
from modeling.model_training import (
    DEFAULT_TEMPERATURE,
    EDGE_DECAY_C,
    EDGE_DECAY_GAMMA,
    RacingPipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STAKE = 10.0          # flat ticket size (HK$)
INITIAL_BANKROLL = 10000.0


def run_walk_forward_backtest(mode='plackett_luce', legacy_features=False):
    logger.info(
        "Starting Walk-Forward Backtest (mode=%s, legacy_features=%s) with Simultaneous "
        "Fractional Kelly and Softmax temperature scaling (tau=%.2f) + longshot decay "
        "(C=%.1f, gamma=%.2f).",
        mode, legacy_features, DEFAULT_TEMPERATURE, EDGE_DECAY_C, EDGE_DECAY_GAMMA,
    )
    # Pass the shared defaults so backtesting and training stay consistent
    pipeline = RacingPipeline(
        features_csv="data/model_features.csv",
        temperature=DEFAULT_TEMPERATURE,
        edge_decay_c=EDGE_DECAY_C,
        edge_decay_gamma=EDGE_DECAY_GAMMA,
        mode=mode,
        legacy_features=legacy_features,
    )
    pipeline.run_walk_forward_backtest()


def run_exotics_backtest(dividends_path="data/historical_dividends.csv",
                         preds_path="data/walk_forward_preds.csv",
                         equity_path="data/exotics_equity_curve.csv"):
    """Real-dividend exotics backtest (Task D).

    Settles every ticket against ACTUAL scraped dividends (data/dividends.csv).
    Coverage is limited to race days whose dividends have been backfilled.
    """
    store, place_map, qpl_map, _ = load_dividend_store(dividends_path)
    if store is None:
        logger.error("No real dividend data available - run scraping/backfill_dividends.py first.")
        return

    covered_races = set(store['race_id'].astype(str))
    logger.info("Dividend store: %d rows covering %d races",
                len(store), len(covered_races))

    preds = pd.read_csv(preds_path, parse_dates=['race_date'])
    mf = pd.read_csv('data/model_features.csv')[
        ['race_id', 'horse_name', 'horse_code', 'finish_position',
         'trainer_urgency_index', 'is_forgive_run', 'trial_won_before_race']]
    mf['finish_rank'] = pd.to_numeric(
        mf['finish_position'].astype(str).str.extract(r'^(\d+)')[0], errors='coerce')
    mf = mf.drop_duplicates(subset=['race_id', 'horse_name'])
    preds = preds.merge(mf, on=['race_id', 'horse_name'], how='left')
    preds = preds[preds['horse_code'].notna() & preds['finish_rank'].notna()]
    preds['race_id'] = preds['race_id'].astype(str)

    data = preds[preds['race_id'].isin(covered_races)].sort_values(['race_date', 'race_id'])
    logger.info("Out-of-sample races with real dividend coverage: %d",
                data['race_id'].nunique())
    if len(data) == 0:
        logger.error("No overlap between predictions and dividend coverage.")
        return

    bankroll = INITIAL_BANKROLL
    equity = []
    records = []
    wins = 0

    for race_id, g in data.groupby('race_id'):
        g = g[g['win_odds'].notna() & g['true_prob'].notna()].reset_index(drop=True)
        if len(g) < 2:
            continue
        race_date = g['race_date'].iloc[0]
        picks = build_core_satellite_bets(g)
        rank = dict(zip(g['horse_code'], g['finish_rank']))

        # --- QPL tickets: banker x satellite pairs (tiered) ---
        for b_code, s_code, tier in picks['qpl_pairs']:
            hit = (rank.get(b_code, 99) <= 3) and (rank.get(s_code, 99) <= 3)
            ret = STAKE * qpl_map.get((race_id, frozenset({b_code, s_code})), 0.0) if hit else 0.0
            bankroll += ret - STAKE
            wins += int(hit)
            records.append({'date': race_date, 'race_id': race_id, 'pool': 'QPL',
                            'tier': tier,
                            'combo': f"{b_code}/{s_code}", 'stake': STAKE,
                            'hit': int(hit), 'return': ret, 'pnl': ret - STAKE})

        # --- PLA tickets: place qualifiers ---
        for code in picks['pla_bets']:
            hit = rank.get(code, 99) <= 3
            ret = STAKE * place_map.get((race_id, code), 0.0) if hit else 0.0
            bankroll += ret - STAKE
            wins += int(hit)
            records.append({'date': race_date, 'race_id': race_id, 'pool': 'PLA',
                            'combo': str(code), 'stake': STAKE,
                            'hit': int(hit), 'return': ret, 'pnl': ret - STAKE})

        equity.append({'date': race_date, 'bankroll': bankroll})

    if not records:
        logger.error("No exotics tickets generated - strategy produced no qualifiers.")
        return

    bets = pd.DataFrame(records)
    bets['date'] = pd.to_datetime(bets['date'])
    eq = pd.DataFrame(equity)
    eq['date'] = pd.to_datetime(eq['date'])

    total_staked = bets['stake'].sum()
    total_return = bets['return'].sum()
    net_pnl = bets['pnl'].sum()
    roi = net_pnl / total_staked * 100 if total_staked else 0.0
    hit_rate = bets['hit'].mean() * 100

    eq['pnl'] = eq['bankroll'] - INITIAL_BANKROLL
    eq['peak_bankroll'] = eq['bankroll'].cummax()
    # Drawdown relative to the peak bankroll (flat staking can push bankroll negative)
    eq['drawdown'] = (eq['bankroll'] - eq['peak_bankroll']) / eq['peak_bankroll'].replace(0, np.nan)
    eq['drawdown'] = eq['drawdown'].replace([np.inf, -np.inf], np.nan).fillna(0)
    max_dd = eq['drawdown'].min() * 100

    daily = eq.groupby('date')['bankroll'].last()
    rets = daily.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if len(rets) > 1 and rets.std() > 0 else 0.0

    top_hits = bets[bets['hit'] == 1].nlargest(5, 'return')[
        ['date', 'race_id', 'pool', 'combo', 'return']]

    logger.info("=======================================================")
    logger.info(" EXOTICS BACKTEST (REAL SETTLED DIVIDENDS ONLY) ")
    logger.info("=======================================================")
    logger.info(f"Coverage:          {data['race_id'].nunique()} races with real dividends")
    logger.info(f"Total Bets:        {len(bets)} (QPL={int((bets['pool']=='QPL').sum())}, "
                f"PLA={int((bets['pool']=='PLA').sum())})")
    logger.info(f"Hit Rate:          {hit_rate:.2f}%")
    logger.info(f"Total Staked:      ${total_staked:.2f}")
    logger.info(f"Total Return:      ${total_return:.2f}")
    logger.info(f"Net PnL:           ${net_pnl:+.2f}")
    logger.info(f"ROI:               {roi:+.2f}%")
    logger.info(f"Max Drawdown:      {max_dd:.2f}%")
    logger.info(f"Sharpe (daily ann.): {sharpe:.2f}")
    # per-pool and per-tier breakdown
    for pool in ['QPL', 'PLA']:
        sub = bets[bets['pool'] == pool]
        if len(sub):
            inv = sub['stake'].sum()
            ret = sub['return'].sum()
            logger.info(f"  [{pool:3s}] bets={len(sub):4d} hit={sub['hit'].mean()*100:5.1f}% "
                        f"staked=${inv:7.0f} pnl=${ret-inv:+8.2f} roi={(ret-inv)/inv*100:+7.2f}%")
    if 'tier' in bets.columns:
        for tier in [1, 2]:
            sub = bets[(bets['pool'] == 'QPL') & (bets['tier'] == tier)]
            if len(sub):
                inv = sub['stake'].sum()
                ret = sub['return'].sum()
                logger.info(f"  [QPL T{tier}] bets={len(sub):4d} hit={sub['hit'].mean()*100:5.1f}% "
                            f"pnl=${ret-inv:+8.2f} roi={(ret-inv)/inv*100:+7.2f}%")
    logger.info("-------------------------------------------------------")
    logger.info(" TOP 5 REAL DIVIDEND PAYOFFS CAPTURED ")
    for _, r in top_hits.iterrows():
        logger.info(f"   {r['date'].date()} {r['race_id']:16s} {r['pool']:4s} "
                    f"{r['combo']:22s} -> ${r['return']:.2f} (div ${r['return']/STAKE*10:.1f})")
    logger.info("=======================================================")

    eq.to_csv(equity_path, index=False)
    bets.to_csv('data/exotics_bets_log.csv', index=False)
    logger.info("Equity curve saved to %s; bet log saved to data/exotics_bets_log.csv", equity_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--exotics', action='store_true',
                    help='run the real-dividend exotics backtest instead of the win-model backtest')
    ap.add_argument('--mode', choices=['plackett_luce', 'lambdarank', 'binary'],
                    default='plackett_luce',
                    help='win-model family: plackett_luce (default), lambdarank, or binary')
    ap.add_argument('--legacy', action='store_true',
                    help='drop the Task A/B feature block (legacy production feature set)')
    args = ap.parse_args()
    if args.exotics:
        run_exotics_backtest()
    else:
        run_walk_forward_backtest(mode=args.mode, legacy_features=args.legacy)

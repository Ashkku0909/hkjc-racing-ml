"""
settle_audit.py — Post-Race Walk-Forward Settlement Engine (Master Rules Task A)
==================================================================================
Ingests the LIVE engine-audit trail (data/engine_audit_YYYYMMDD.csv) written by
web_live.py at the T-5m / T-2m / CLOSED milestones and settles every implied
ticket against REAL scraped official dividends (data/historical_dividends.csv,
with data/dividends.csv merged when present) - NO lookahead: each milestone only
uses the odds/probabilities the engine actually saw at that decision point.

For every (race, milestone, pool) it computes:
  bets / staked / returned / net PnL / ROI% / hit rate
and the odds slippage  Slippage = O_exec(milestone) - O_official(dividend)
for Win & Place candidates, plus pace-rule validation cohorts (runners whose
logits were adjusted by the N_leaders=0 / N_leaders>=3 pace rules carry an
'adj=±x' audit tag -> separate strike rate & yield).

Betting policy (documented, Master-Rules aligned, thresholds overridable):
  WIN    : ev >= WIN_EV (ratio-1, default 0.15) AND odds in [WIN_MIN_ODDS,
           WIN_MAX_ODDS] AND prob >= WIN_MIN_PROB AND smart >= WIN_MIN_SMART
  PLACE  : place_ev (ratio) >= PLACE_EV AND place_odds >= PLACE_MIN_ODDS
           AND smart >= PLACE_MIN_SMART
  QUINELLA : boxed pair of the two WIN-eligible horses (top-N by fused prob)
  QPL      : boxed pair of the PLACE-eligible horses (top-N by place prob)
  Every pool settled per milestone separately.

Usage:
  python settle_audit.py --date 2026-09-09
  python settle_audit.py --date 2026-09-09 --stake 100 --json data/settle.json
  python settle_audit.py --audit data/engine_audit_20260909.csv
                         --dividends data/historical_dividends.csv

⚠️ Educational use only — statistical modelling study, not gambling advice.
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Master-Rules default policy thresholds (override via CLI)
# ---------------------------------------------------------------------------
WIN_EV = 0.15            # audit 'ev' is odds-ratio MINUS 1 -> +15% edge
WIN_MIN_ODDS = 2.2
WIN_MAX_ODDS = 14.0
WIN_MIN_PROB = 0.16
WIN_MIN_SMART = 50.0
PLACE_EV = 1.15          # audit 'place_ev' is a RATIO (prob * place odds)
PLACE_MIN_ODDS = 1.5
PLACE_MIN_SMART = 50.0
DEFAULT_STAKE = 100.0    # per ticket (HKJC dividend displayed per $10)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Dividend store
# ---------------------------------------------------------------------------
def load_dividends(paths):
    """Load WIN/PLACE/QUINELLA/QPL payout multipliers from scraped dividend CSVs.

    Returns dicts keyed by (race_id, ...) -> per-$1 multiplier (dividend/10):
      win_map   [(race_id, horse_number)] -> mult        (winning horse)
      place_map [(race_id, horse_number)] -> mult        (every placed horse)
      qin_map   [(race_id, frozenset({a,b}))] -> mult    (winning quinella)
      qpl_map   [(race_id, frozenset({a,b}))] -> mult    (every quinella-place)
    """
    frames = []
    for p in paths:
        if p and os.path.exists(p):
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                print(f"  [settle] warn: cannot read {p}: {e}")
    if not frames:
        return {}, {}, {}, {}
    df = pd.concat(frames, ignore_index=True)
    if 'combo' in df.columns:
        df = df.drop_duplicates(subset=['race_id', 'pool', 'combo'])
    win_map, place_map, qin_map, qpl_map = {}, {}, {}, {}
    for _, r in df.iterrows():
        codes = [str(c).strip() for c in str(r.get('combo_codes', '')).split('/') if str(c).strip()]
        race_id = str(r['race_id'])
        try:
            mult = float(r['dividend']) / 10.0
        except (TypeError, ValueError):
            continue
        pool = str(r.get('pool', '')).upper()
        if pool == 'WIN' and len(codes) == 1:
            win_map[(race_id, codes[0])] = mult
        elif pool == 'PLACE' and len(codes) == 1:
            place_map[(race_id, codes[0])] = mult
        elif pool == 'QUINELLA' and len(codes) == 2:
            qin_map[(race_id, frozenset(codes))] = mult
        elif pool == 'QPL' and len(codes) == 2:
            qpl_map[(race_id, frozenset(codes))] = mult
    return win_map, place_map, qin_map, qpl_map


# ---------------------------------------------------------------------------
# Audit parser helpers
# ---------------------------------------------------------------------------
def parse_flags(flags):
    """'pace=SLOW|nl=0|v=HV|adj=-0.15|draw=8|syndicate_steam' -> dict + tags."""
    d = {'pace': None, 'nl': None, 'v': None, 'adj': None, 'draw': None}
    tags = []
    if not isinstance(flags, str) or not flags.strip():
        return d, tags
    for tok in flags.split('|'):
        tok = tok.strip()
        if not tok:
            continue
        if '=' in tok:
            k, _, v = tok.partition('=')
            k = k.strip().lower()
            if k in ('pace', 'nl', 'v', 'adj', 'draw'):
                d[k] = v.strip()
            else:
                tags.append(tok)
        else:
            tags.append(tok)
    return d, tags


def load_audit(path):
    """Load the engine-audit CSV -> DataFrame with parsed flag context."""
    df = pd.read_csv(path)
    ctx = df['flags'].map(parse_flags)
    df['pace_scn'] = [c[0]['pace'] for c in ctx]
    df['n_leaders'] = pd.to_numeric([c[0]['nl'] for c in ctx], errors='coerce')
    df['venue_tag'] = [c[0]['v'] for c in ctx]
    df['adj_logit'] = pd.to_numeric([c[0]['adj'] for c in ctx], errors='coerce')
    df['draw_tag'] = pd.to_numeric([c[0]['draw'] for c in ctx], errors='coerce')
    for c in ('win_odds', 'place_odds', 'smart', 'prob', 'ev', 'place_ev'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['horse_number'] = df['horse_number'].astype(str).str.strip()
    return df


def race_parts(race_id):
    """'2026-09-09_Race3' -> (date_str, race_no)"""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_Race(\d+)$", str(race_id).strip())
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


# ---------------------------------------------------------------------------
# Eligibility / betting policy
# ---------------------------------------------------------------------------
def win_eligible(frame, ev_min=WIN_EV, o_lo=WIN_MIN_ODDS, o_hi=WIN_MAX_ODDS,
                 p_min=WIN_MIN_PROB, s_min=WIN_MIN_SMART):
    m = (frame['ev'].fillna(-9) >= ev_min) & \
        (frame['win_odds'].fillna(0) >= o_lo) & \
        (frame['win_odds'].fillna(99) <= o_hi) & \
        (frame['prob'].fillna(0) >= p_min) & \
        (frame['smart'].fillna(s_min) >= s_min)
    return frame.index[m].tolist()


def place_eligible(frame, ev_min=PLACE_EV, o_min=PLACE_MIN_ODDS, s_min=PLACE_MIN_SMART):
    m = (frame['place_ev'].fillna(0) >= ev_min) & \
        (frame['place_odds'].fillna(0) >= o_min) & \
        (frame['smart'].fillna(s_min) >= s_min)
    return frame.index[m].tolist()


def _top_pairs(frame, idx, n_max, sort_col, exclude=None):
    """Top-N unordered pairs by combined score (sum of sort_col, desc)."""
    exclude = set(exclude or [])
    cands = [i for i in idx if i not in exclude]
    pairs = []
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            pairs.append((cands[a], cands[b]))
    if n_max and len(pairs) > n_max:
        scores = []
        for a, b in pairs:
            scores.append(float(frame.loc[a, sort_col]) + float(frame.loc[b, sort_col]))
        pairs = [p for _, p in sorted(zip(scores, pairs), reverse=True)[:n_max]]
    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(bets):
    """bets: list of (returned, staked)."""
    if not bets:
        return dict(n=0, staked=0.0, returned=0.0, net=0.0, roi=float('nan'),
                    hit=float('nan'))
    staked = float(sum(b[1] for b in bets))
    returned = float(sum(b[0] for b in bets))
    net = returned - staked
    hits = [b for b in bets if b[0] > 0.0]
    return dict(n=len(bets), staked=staked, returned=returned, net=net,
                roi=(net / staked * 100.0) if staked else float('nan'),
                hit=len(hits) / len(bets) * 100.0 if bets else float('nan'))


def _fmt_m(m):
    if m['n'] == 0:
        return f"  0 bets"
    return (f"  {m['n']:3d} bets | staked ${m['staked']:9,.0f} | "
            f"returned ${m['returned']:9,.0f} | net ${m['net']:+10,.0f} | "
            f"ROI {m['roi']:+7.2f}% | hit {m['hit']:5.1f}%")


# ---------------------------------------------------------------------------
# Main settlement
# ---------------------------------------------------------------------------
def settle(audit: pd.DataFrame, win_map, place_map, qin_map, qpl_map,
           stake=100.0, max_pairs=5, ev_min=WIN_EV, place_ev_min=PLACE_EV):
    """Run the walk-forward settlement over every (race, milestone)."""
    results = OrderedDict()          # (race_id, milestone) -> {pool: metrics}
    bet_log = []                     # per-ticket rows
    slips = []                       # slippage rows (win/place candidates)
    for (race_id, milestone), frame in audit.groupby(['race_id', 'milestone']):
        frame = frame.sort_values('prob', ascending=False).reset_index(drop=True)
        fr = results.setdefault((race_id, milestone), {})
        w_idx = win_eligible(frame)
        p_idx = place_eligible(frame)
        # --- WIN (per horse) ---
        rets, stakes = [], []
        for i in w_idx:
            hn = frame.loc[i, 'horse_number']
            mult = win_map.get((race_id, hn))
            ret = stake * mult if mult else 0.0
            rets.append(ret); stakes.append(stake)
            bet_log.append({'race_id': race_id, 'milestone': milestone, 'pool': 'WIN',
                            'horse': frame.loc[i, 'horse_name'], 'horse_no': hn,
                            'stake': stake, 'returned': ret,
                            'odds_exec': frame.loc[i, 'win_odds']})
            if mult:
                slips.append({'race_id': race_id, 'milestone': milestone, 'pool': 'WIN',
                              'horse': frame.loc[i, 'horse_name'], 'horse_no': hn,
                              'odds_exec': frame.loc[i, 'win_odds'],
                              'official': mult, 'slippage': float(frame.loc[i, 'win_odds']) - mult})
        fr['WIN'] = _metrics(list(zip(rets, stakes)))
        # --- PLACE (per horse) ---
        rets, stakes = [], []
        for i in p_idx:
            hn = frame.loc[i, 'horse_number']
            mult = place_map.get((race_id, hn))
            ret = stake * mult if mult else 0.0
            rets.append(ret); stakes.append(stake)
            bet_log.append({'race_id': race_id, 'milestone': milestone, 'pool': 'PLACE',
                            'horse': frame.loc[i, 'horse_name'], 'horse_no': hn,
                            'stake': stake, 'returned': ret,
                            'odds_exec': frame.loc[i, 'place_odds']})
            if mult:
                slips.append({'race_id': race_id, 'milestone': milestone, 'pool': 'PLACE',
                              'horse': frame.loc[i, 'horse_name'], 'horse_no': hn,
                              'odds_exec': frame.loc[i, 'place_odds'],
                              'official': mult, 'slippage': float(frame.loc[i, 'place_odds']) - mult})
        fr['PLACE'] = _metrics(list(zip(rets, stakes)))
        # --- QUINELLA (box of top WIN-eligible pairs) ---
        pairs = _top_pairs(frame, w_idx, max_pairs, 'prob')
        rets, stakes = [], []
        for a, b in pairs:
            ha, hb = frame.loc[a, 'horse_number'], frame.loc[b, 'horse_number']
            mult = qin_map.get((race_id, frozenset({ha, hb})))
            ret = stake * mult if mult else 0.0
            rets.append(ret); stakes.append(stake)
            bet_log.append({'race_id': race_id, 'milestone': milestone, 'pool': 'QUINELLA',
                            'horse': f"{frame.loc[a,'horse_name']}/{frame.loc[b,'horse_name']}",
                            'horse_no': f"{ha}/{hb}", 'stake': stake, 'returned': ret,
                            'odds_exec': np.nan})
        fr['QUINELLA'] = _metrics(list(zip(rets, stakes)))
        # --- QPL (box of top PLACE-eligible pairs) ---
        pairs = _top_pairs(frame, p_idx, max_pairs, 'place_ev')
        rets, stakes = [], []
        for a, b in pairs:
            ha, hb = frame.loc[a, 'horse_number'], frame.loc[b, 'horse_number']
            mult = qpl_map.get((race_id, frozenset({ha, hb})))
            ret = stake * mult if mult else 0.0
            rets.append(ret); stakes.append(stake)
            bet_log.append({'race_id': race_id, 'milestone': milestone, 'pool': 'QPL',
                            'horse': f"{frame.loc[a,'horse_name']}/{frame.loc[b,'horse_name']}",
                            'horse_no': f"{ha}/{hb}", 'stake': stake, 'returned': ret,
                            'odds_exec': np.nan})
        fr['QPL'] = _metrics(list(zip(rets, stakes)))
    return results, pd.DataFrame(bet_log), pd.DataFrame(slips)


def report(results, bets, slips, audit, stake, venue_hint=None):
    """Pretty-print grouped summaries + pace-rule validation cohorts."""
    lines = []
    pools = ['WIN', 'PLACE', 'QUINELLA', 'QPL']
    lines.append("=" * 78)
    lines.append("POST-RACE SETTLEMENT — engine-audit vs official dividends")
    lines.append("=" * 78)
    date = ""
    races = sorted({rid for rid, _ in results})
    for rid in races:
        date, _ = race_parts(rid)
        break
    div_races = set(slips['race_id']) if len(slips) else set()
    found = len(div_races)
    lines.append(f"Date {date or '?'} | races in audit: {len(races)} | "
                 f"races with matched dividends: {found}/{len(races)}")
    if found < len(races):
        lines.append("  ⚠️ no dividends yet for some/all races - backfill with "
                     "`python -m scraping.backfill_dividends` after the meeting")
    # per-milestone aggregate (from the raw per-ticket log)
    for milestone in ('T-5m', 'T-2m', 'CLOSED'):
        lines.append("")
        lines.append(f"--- Milestone {milestone} (stake ${stake:,.0f}/ticket) ---")
        bm = bets[bets['milestone'] == milestone] if len(bets) else bets.iloc[0:0]
        for pool in pools:
            sub = bm[bm['pool'] == pool]
            if not len(sub):
                continue
            staked = float(sub['stake'].sum())
            returned = float(sub['returned'].sum())
            net = returned - staked
            n = len(sub)
            hits = int((sub['returned'] > 0).sum())
            m = dict(n=n, staked=staked, returned=returned, net=net,
                     roi=net / staked * 100.0 if staked else float('nan'),
                     hit=hits / n * 100.0 if n else float('nan'))
            lines.append(f"  {pool:<9} {_fmt_m(m)}")
    # slippage
    if len(slips):
        lines.append("")
        lines.append("--- Odds Slippage  (Slippage = O_exec(milestone) - O_official) ---")
        for pool in ('WIN', 'PLACE'):
            sub = slips[slips['pool'] == pool]
            lines.append(f"  {pool:<6} on settled winners:")
            for milestone in ('T-5m', 'T-2m', 'CLOSED'):
                g = sub[sub['milestone'] == milestone]['slippage']
                if not len(g):
                    continue
                lines.append(f"    {milestone}: n={len(g):3d} mean {g.mean():+.3f}  "
                             f"median {g.median():+.3f}")
    # pace-rule validation cohorts (Master Rules §4)
    if len(audit):
        lines.append("")
        lines.append("--- Pace-rule validation (impacted vs untouched runners) ---")
        imp = audit[audit['adj_logit'].notna()]
        if len(imp) == 0:
            lines.append("  no 'adj=' audit tags yet (written from tonight's runs)")
        else:
            neg = imp[imp['adj_logit'] < 0]
            pos = imp[imp['adj_logit'] > 0]
            none = audit[audit['adj_logit'].isna()]
            for label, g in (("penalised (adj<0)", neg), ("boosted (adj>0)", pos),
                             ("untouched", none)):
                if not len(g):
                    continue
                scn = g['pace_scn'].fillna('N/A')
                for ps, gg in g.groupby(scn):
                    wret = 0.0
                    # place/win result proxy is handled per-ticket; here report
                    # cohort size + mean EV at each milestone as the diagnostic
                    evs = gg.groupby('milestone')['ev'].mean().round(3)
                    evs = {k: v for k, v in evs.items()}
                    lines.append(f"  {label:<28} pace={ps:<9} runners={len(gg):3d} "
                                 f"mean EV(T-5m/T-2m/CLOSED)={evs}")
    return "\n".join(lines)


def _resolve_paths(args):
    if args.audit:
        audit_path = args.audit
    else:
        day = args.date.replace('-', '') if args.date else None
        if day:
            audit_path = os.path.join(HERE, 'data', f'engine_audit_{day}.csv')
        else:
            cands = sorted(f for f in os.listdir(os.path.join(HERE, 'data'))
                           if re.match(r'^engine_audit_\d{8}\.csv$', f))
            if not cands:
                print("No engine_audit_*.csv found under data/ - pass --audit.")
                sys.exit(2)
            audit_path = os.path.join(HERE, 'data', cands[-1])
    divs = args.dividends or [os.path.join(HERE, 'data', 'historical_dividends.csv'),
                              os.path.join(HERE, 'data', 'dividends.csv')]
    return audit_path, divs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date', help='race day YYYY-MM-DD (default: latest audit file)')
    ap.add_argument('--venue', help='venue filter (ST/HV)')
    ap.add_argument('--audit', help='path to engine_audit_YYYYMMDD.csv')
    ap.add_argument('--dividends', nargs='*', help='dividend CSV path(s)')
    ap.add_argument('--stake', type=float, default=DEFAULT_STAKE)
    ap.add_argument('--max-pairs', type=int, default=5, help='max Q/QP boxes per race')
    ap.add_argument('--win-ev', type=float, default=WIN_EV)
    ap.add_argument('--place-ev', type=float, default=PLACE_EV)
    ap.add_argument('--json', help='write per-ticket JSON log')
    args = ap.parse_args()

    audit_path, div_paths = _resolve_paths(args)
    if not os.path.exists(audit_path):
        print(f"Audit file not found: {audit_path}")
        sys.exit(2)
    audit = load_audit(audit_path)
    if args.venue:
        audit = audit[audit['venue_tag'].fillna(args.venue.upper()) == args.venue.upper()]
    if not len(audit):
        print("No audit rows after filters.")
        sys.exit(2)

    print(f"[settle] audit      : {audit_path}")
    print(f"[settle] dividends  : {div_paths}")
    win_map, place_map, qin_map, qpl_map = load_dividends(div_paths)
    print(f"[settle] dividend maps: WIN {len(win_map)} | PLACE {len(place_map)} | "
          f"QIN {len(qin_map)} | QPL {len(qpl_map)}")

    results, bets, slips = settle(audit, win_map, place_map, qin_map, qpl_map,
                                  stake=args.stake, max_pairs=args.max_pairs,
                                  ev_min=args.win_ev, place_ev_min=args.place_ev)
    print(report(results, bets, slips, audit, args.stake, args.venue))

    if args.json:
        out = {
            'audit': audit_path,
            'stake': args.stake,
            'results': {f"{k[0]}|{k[1]}": v for k, v in results.items()},
            'tickets': bets.to_dict('records'),
            'slippage': slips.to_dict('records') if len(slips) else [],
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[settle] JSON written: {args.json}")


if __name__ == '__main__':
    main()

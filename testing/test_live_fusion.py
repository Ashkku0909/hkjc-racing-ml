"""
Playback tests: dynamic post-time state machine + Bayesian fusion + dual-mode
betting guardrails (no live scraping - pure functions only).

Simulates a race whose nominal post is NOW but which actually jumps ~2 minutes
late (gate loading): the market keeps ticking past nominal post, so the engine
must stay in LOADING_DELAY with turbo polling (0.8s) and the Bayesian weight
pinned at 0.85 until the pool ACTUALLY freezes.
"""
import os
import sys
import time
import itertools
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_live import (  # noqa: E402
    execution_state, fusion_weight, t_tilde_seconds, effective_poll_ttl,
    bayesian_fusion, flag_row, _henery_rank, CLOSED_FROZEN_SEC,
)
from modeling.exotics_pricing import henery_gamma_for_field  # noqa: E402

HKT = timezone(timedelta(hours=8))


def _frame(epoch, n=6, odds=None):
    """One snapshot frame: epoch (unix s) -> win odds per horse 1..n."""
    if odds is None:
        odds = [5.0 + i * 1.5 for i in range(n)]
    return pd.DataFrame({
        "epoch": [epoch] * n,
        "horse_number": list(range(1, n + 1)),
        "win_odds": odds,
    })


def _ticking_sub(n=6, last_move_ago=10.0):
    """Market still ticking: consecutive frames differ, last move was recent."""
    now = time.time()
    frames = []
    for k in range(12):
        shift = 0.2 * (11 - k)          # odds drift a touch each frame
        frames.append(_frame(now - k * 30.0, n=n,
                             odds=[5.0 + i * 1.5 + shift for i in range(n)]))
    return pd.concat(frames, ignore_index=True)


def _frozen_sub(n=6, last_move_ago=90.0):
    """Frozen pool: last move >= frozen window ago."""
    now = time.time()
    frames = []
    for k in range(12):
        shift = 0.2 * (11 - k) if (k * 30.0) >= last_move_ago else 0.0
        frames.append(_frame(now - k * 30.0, n=n,
                             odds=[5.0 + i * 1.5 + shift for i in range(n)]))
    return pd.concat(frames, ignore_index=True)


def test_state_machine_delayed_race():
    now = datetime.now(HKT)
    post = now + timedelta(minutes=10)
    assert execution_state(post, now=now) == "PRE_POST"
    assert execution_state(post, now=post - timedelta(seconds=60)) == "TURBO_APPROACH"

    # 2-minute gate delay: past nominal, still ticking -> NOT closed
    post_past = now - timedelta(seconds=120)
    sub = _ticking_sub()
    assert execution_state(post_past, sub, now=now) == "LOADING_DELAY"
    assert execution_state(post_past, None, now=now) == "LOADING_DELAY"

    # pool freezes 90s ago (>= CLOSED_FROZEN_SEC) -> OFFICIAL_CLOSED
    sub2 = _frozen_sub(last_move_ago=CLOSED_FROZEN_SEC + 30.0)
    assert execution_state(post_past, sub2, now=now) == "OFFICIAL_CLOSED"


def test_fusion_weight_pinned_until_freeze():
    post = datetime.now(HKT) + timedelta(hours=2)
    # far from post -> model-dominated asymptote (w_min = 0.25)
    assert abs(fusion_weight(post, "PRE_POST") - 0.25) < 1e-9
    # peak sensitivity pinned in the final window / gate delay
    assert abs(fusion_weight(post, "TURBO_APPROACH") - 0.85) < 1e-9
    assert abs(fusion_weight(post, "LOADING_DELAY") - 0.85) < 1e-9
    # smooth logistic ramp between the asymptotes (kappa=0.02, T0=360s)
    post_near = datetime.now(HKT) + timedelta(seconds=200)
    w200 = fusion_weight(post_near)
    assert 0.80 < w200 < 0.84            # approaching peak, market dominates
    post_mid = datetime.now(HKT) + timedelta(seconds=900)
    assert 0.24 < fusion_weight(post_mid) < 0.45   # smooth mid-ramp, below late-step


def test_t_tilde_stretch():
    now = datetime.now(HKT)
    assert abs(t_tilde_seconds(now + timedelta(seconds=600)) - 600.0) < 1e-9
    assert abs(t_tilde_seconds(now + timedelta(seconds=10)) - 20.0) < 1e-9
    assert abs(t_tilde_seconds(now - timedelta(seconds=30)) - 20.0) < 1e-9


def test_turbo_ttl_throttle():
    now = datetime.now(HKT)
    assert abs(effective_poll_ttl(now + timedelta(seconds=60), 1.0) - 0.8) < 1e-9
    assert abs(effective_poll_ttl(now - timedelta(seconds=60), 1.0) - 0.8) < 1e-9
    assert abs(effective_poll_ttl(now + timedelta(minutes=10), 1.0) - 1.0) < 1e-9


def test_bayesian_fusion_softmax_and_flow():
    p_model = np.array([0.5, 0.3, 0.2])
    p_mkt = np.array([0.2, 0.3, 0.5])
    fused = bayesian_fusion(p_model, p_mkt, 0.85, np.array([0.0, 0.0, 0.0]))
    assert abs(fused.sum() - 1.0) < 1e-6
    # steamer boost lifts its share
    boosted = bayesian_fusion(p_model, p_mkt, 0.85, np.array([0.0, 0.35, 0.0]))
    assert boosted[1] > fused[1]
    # NaN model prob falls back to market only
    fused_nan = bayesian_fusion(np.array([np.nan, 0.5, 0.5]),
                                np.array([0.2, 0.4, 0.4]), 0.85)
    assert abs(fused_nan.sum() - 1.0) < 1e-6


def test_dual_mode_bet_filter():
    # solid mid-price value -> straight WIN
    assert flag_row({"prob": 0.30, "win_odds": 6.0, "ev": 0.30,
                     "smart_money_score": 60}) == "🎯 PRIME W"
    # longshot with EV 1.3 -> NEVER straight WIN, routed exotics
    assert flag_row({"prob": 0.10, "win_odds": 20.0, "ev": 0.30,
                     "smart_money_score": 60}) == "🎯 EXOTIC"
    assert flag_row({"prob": 0.40, "win_odds": 20.0, "ev": 0.40,
                     "smart_money_score": 60}) == "🎯 EXOTIC"
    # low-probability trap -> no signal even with EV
    assert flag_row({"prob": 0.05, "win_odds": 6.0, "ev": 0.50,
                     "smart_money_score": 60}) == "—"
    # drifter
    assert flag_row({"prob": 0.30, "win_odds": 6.0, "ev": 0.10,
                     "smart_money_score": 35}) == "⚠️ DRIFT"


def test_henery_gamma_bounds():
    assert abs(henery_gamma_for_field(4) - 0.75) < 1e-9
    assert abs(henery_gamma_for_field(14) - 0.88) < 1e-9
    g = [henery_gamma_for_field(n) for n in range(4, 15)]
    assert all(0.75 - 1e-9 <= x <= 0.88 + 1e-9 for x in g)
    assert all(g[i] <= g[i + 1] + 1e-9 for i in range(len(g) - 1))


def _henery_ref(p, r, gamma):
    """Brute-force recursive Henery chain reference (exact, O(n^r)).

    res[k] = sum over ordered (a1..a_{r-1}) distinct, none == k of
        p[a1] * prod_{t=2..r-1} g[a_t] / (S - sum_{u<t} g[a_u])
    (the same chain the original web_live loop enumerated).
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < r:
        return np.full(n, np.nan)
    if r == 1:
        return p.copy()
    g = np.clip(p, 1e-12, 1.0) ** gamma
    S = float(g.sum())
    res = np.zeros(n)
    for k in range(n):
        others = [i for i in range(n) if i != k]
        total = 0.0
        for prefix in itertools.permutations(others, r - 1):
            # chain: a1 (win prob p[a1]) -> a2 -> ... -> a_{r-1} -> k last,
            # each conditional draw removes its g from the denominator.
            denom = S - g[prefix[0]]
            prob = p[prefix[0]]
            for a in prefix[1:]:
                if denom <= 1e-12:
                    prob = 0.0
                    break
                prob *= g[a] / denom
                denom -= g[a]
            if denom > 1e-12:
                prob *= g[k] / denom
            else:
                prob = 0.0
            total += prob
        res[k] = total
    return res


def test_henery_r3_r4_vectorized_parity():
    """Vectorized r=3/r=4 must match the recursive Plackett-Luce chain to ~1e-12
    (Master Rules Task D: zero parity divergence)."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for n in (4, 5, 7, 10, 14):
        for gamma in (0.75, 0.81, 0.88):
            for _ in range(6):
                p = rng.dirichlet(np.ones(n))
                for r in (1, 2, 3, 4):
                    fast = _henery_rank(p, r, gamma)
                    ref = _henery_ref(p, r, gamma)
                    assert np.allclose(fast, ref, atol=1e-9, rtol=1e-9), \
                        f"parity break n={n} r={r} gamma={gamma}"
                    worst = max(worst, float(np.nanmax(np.abs(fast - ref))))
    assert worst < 1e-9, worst
    # every exact rank marginal sums to 1 whenever the field can fill the rank
    p = rng.dirichlet(np.ones(12))
    for r in (1, 2, 3, 4):
        assert abs(_henery_rank(p, r).sum() - 1.0) < 1e-9, r


def test_henery_extreme_field_edge():
    """n == r boundaries (trio on a 3-runner field / quartet on 4) must work and
    short fields must not leak mass via degenerate denominators."""
    for n, r in ((3, 3), (4, 4), (4, 3), (5, 4)):
        p = np.array([0.5 / i if i else 0.0 for i in range(1, n + 1)])
        p = p / p.sum()
        res = _henery_rank(p, r, 0.81)
        assert np.isfinite(res).all()
        assert abs(res.sum() - 1.0) < 1e-9, (n, r)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()

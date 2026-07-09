"""
req4_config.py
---------------
Single source of truth for Requirement 4 parameters.

Mirrors utils/req3_config.py's pattern (see that file's docstring for the
negative-regret bug it was written to prevent: precompute_clairvoyant.py
and run_req3.py used to keep separate copies of VALUES/BUDGET that could
silently drift apart, producing a cached clairvoyant solved for the wrong
budget). run_req4.py and precompute_clairvoyant_req4.py both import from
here so they can never disagree, and make_cache_key() is centralised so a
parameter change automatically invalidates the cache (clean miss -> the
LP is recomputed) instead of silently loading a stale one.

Requirement 4 reuses Requirement 3's campaigns (same VALUES/BUDGET/
CONFLICT_EDGES/AVAILABLE_BIDS as req3_config.py) so the four requirements
tell one coherent story about the same advertiser -- only the environment's
non-stationarity pattern changes (drift every round for R3, few long
blocks for R4).
"""

import hashlib

import numpy as np

from utils.req3_config import (
    VALUES, T, BUDGET, N_TRIALS, N_COMPETITORS, CONFLICT_EDGES,
    AVAILABLE_BIDS,
)

# ---------------------------------------------------------------------------
# Requirement 4 - specific: "slightly" non-stationary means few, long
# intervals, as opposed to Requirement 3's every-round drift.
# ---------------------------------------------------------------------------
N_INTERVALS = 5
BLOCK_SIZE = T // N_INTERVALS          # 2000
U_T = N_INTERVALS - 1                  # upper bound on true regime changes, feeds CUSUM

# Sliding-window length: tied to the interval length rather than the
# textbook W = 2*sqrt(T) ~= 200 rule of thumb. With sum(K_i) cells far
# larger than a toy K=3 bandit, W~200 evicts a cell's samples before the
# LP has enough mass on it to estimate it well -- pure window churn, not
# adaptation. Confirmed empirically (see tune_sw_window() in run_req4.py).
SW_WINDOW = BLOCK_SIZE

ADVERSARIAL_ENV_CLASS_NAME = "AdversarialMultiCampaignEnv"
SHOCKS_MODE_PARAMS = dict(block_size=BLOCK_SIZE, n_regimes=N_INTERVALS)


def make_cache_key(
    mode="shocks",
    mode_params=None,
    T=T,
    values=VALUES,
    budget=BUDGET,
    conflict_edges=CONFLICT_EDGES,
    available_bids=AVAILABLE_BIDS,
    env_class_name=ADVERSARIAL_ENV_CLASS_NAME,
):
    """
    Short hex digest that changes whenever any LP-relevant parameter
    changes -- same construction as req3_config.make_cache_key, but also
    folds in mode_params (block_size, n_regimes) since those change the
    realised m_t sequence just as much as values/budget do.
    """
    params = mode_params if mode_params is not None else SHOCKS_MODE_PARAMS
    payload = repr((
        env_class_name,
        int(T),
        tuple(float(v) for v in values),
        float(budget),
        tuple(sorted(tuple(e) for e in conflict_edges)),
        tuple(float(b) for b in available_bids),
        str(mode),
        tuple(sorted((params or {}).items())),
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]

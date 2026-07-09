"""
req4_config.py
--------------
Requirement 4 parameters, extending req3_config.py.

Reuses the same VALUES, BUDGET, T, CONFLICT_EDGES, AVAILABLE_BIDS, and
N_COMPETITORS from req3_config so Requirements 2–4 operate on the same
problem instance. Only the non-stationarity pattern changes: Requirement 4
uses mode='shocks' with few, long stationary blocks (block_size=2000 → 5
blocks over T=10000) rather than the per-round drift of Requirement 3.

run_req4.py and precompute_clairvoyant_req4.py both import from here.
make_cache_key() folds in mode_params (block_size, n_regimes) so any
parameter change automatically invalidates cached dynamic clairvoyants.
"""

import hashlib

import numpy as np

from utils.req3_config import (
    VALUES, T, BUDGET, N_TRIALS, N_COMPETITORS, CONFLICT_EDGES,
    AVAILABLE_BIDS,
)

# Requirement 4: piecewise-stationary with few, long blocks.
N_INTERVALS = 5
BLOCK_SIZE = T // N_INTERVALS          # 2000
U_T = N_INTERVALS - 1                  # upper bound on regime changes (feeds CUSUM)

# Sliding-window length tied to the block length rather than the textbook
# W = 2*sqrt(T) ≈ 200: with sum(K_i) cells much larger than a toy K=3 bandit,
# W≈200 evicts a cell's samples before the LP accumulates enough mass,
# causing pure window churn rather than adaptation.
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
    """Short hex digest of all LP-relevant parameters including mode_params."""
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

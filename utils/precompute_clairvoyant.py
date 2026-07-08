"""
precompute_clairvoyant.py
--------------------------
One-shot script: precompute the dynamic clairvoyant (Requirements 3 and 4)
for every trial seed and cache it to disk, so run_req3.py / run_req4.py
don't re-solve an expensive LP (~440k variables for T=10000, N=4, K=11)
inside every trial of every re-run.

Usage
-----
    python -m utils.precompute_clairvoyant

Loops over every (mode, mode_params) pair in MODE_CONFIGS, producing one
cache file per config (keyed by a hash of every LP-relevant parameter, so
configs that happen to share `mode` but differ in generation parameters,
e.g. R3's fast drift vs. R4's slow shocks, never collide). Prints the
cache key to paste into CLAIRVOYANT_CACHE_KEY at the top of run_req3.py /
run_req4.py.
"""

import hashlib
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from utils.environments import NonStationaryMultiCampaignEnv
from utils.experiments import compute_clairvoyant_dynamic_multi, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

T = 10_000
VALUES = [0.8, 0.6, 0.9, 0.7]
BUDGET = 1_600.0
CONFLICT_EDGES = [(0, 1), (2, 3)]
AVAILABLE_BIDS = np.linspace(0, 1, 11)
N_TRIALS = 20
SEEDS = list(range(N_TRIALS))

# One entry per non-stationary regime used across the project.
MODE_CONFIGS = [
    ("drift", dict()),                                 # Requirement 3: highly non-stationary
    ("shocks", dict(block_size=2000, n_regimes=5)),     # Requirement 4: slightly non-stationary
]


def make_cache_key(mode, mode_params):
    payload = repr((
        int(T), tuple(float(v) for v in VALUES), float(BUDGET),
        tuple(sorted(tuple(e) for e in CONFLICT_EDGES)),
        tuple(float(b) for b in AVAILABLE_BIDS),
        str(mode), tuple(sorted((mode_params or {}).items())),
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def precompute_for_mode(mode, mode_params):
    key = make_cache_key(mode, mode_params)
    cache_path = DATA_DIR / f"clairvoyant_dyn_{key}.pkl"
    logger.info("=" * 60)
    logger.info("mode=%s params=%s -> %s", mode, mode_params, cache_path)

    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logger.info("Loaded existing cache with %d entries", len(cache))

    missing = [s for s in SEEDS if s not in cache]
    if not missing:
        logger.info("Nothing to do.")
        return key

    for k, seed in enumerate(missing):
        env = NonStationaryMultiCampaignEnv(
            values=VALUES, budget=BUDGET, T=T, available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode=mode, **mode_params,
        )
        t0 = time.time()
        opt_total, opt_per_round = compute_clairvoyant_dynamic_multi(
            m_seq=env.m, values=env.values, bid_sets=env.bid_sets,
            budget=env.budget, conflict_edges=env.conflict_edges,
        )
        cache[seed] = {"opt_total": opt_total, "opt_per_round": opt_per_round}
        logger.info("[%2d/%d] seed=%d opt_per_round=%.4f (%.1fs)",
                    k + 1, len(missing), seed, opt_per_round, time.time() - t0)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return key


def main():
    keys = {}
    for mode, params in MODE_CONFIGS:
        keys[(mode, tuple(sorted(params.items())))] = precompute_for_mode(mode, params)
    logger.info("=" * 60)
    for (mode, params), key in keys.items():
        logger.info("%-8s %-40s CLAIRVOYANT_CACHE_KEY = %r", mode, dict(params), key)
    return keys


if __name__ == "__main__":
    main()

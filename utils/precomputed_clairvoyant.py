"""
precompute_clairvoyant.py
-------------------------
One-shot script: pre-compute the dynamic clairvoyant (Requirement 3
adversarial baseline) for every trial seed and pickle the results.

Why this script exists
----------------------
For T=10000, N=4, K=11, |E|=2 the dynamic-clairvoyant LP has ~440k
variables and ~70k constraints.  Even with a sparse CSR encoding and
HiGHS, each LP takes 5-30 seconds.  Computing it inside the trial loop
means n_trials LPs per re-run — a strong incentive to cache.

Run this once (e.g. overnight).  `run_primal_dual_trials` will then
read the pickle and skip the LP entirely.  This script loops over every
mode in `MODES` so you get a separate cache file for each adversarial
regime, and `run_req3.py` can pick the right one based on its `ENV_MODE`.

Usage
-----
    python -m utils.precompute_clairvoyant

The cache key is a hash of the relevant problem parameters:
  (T, N, values, budget, conflict_edges, available_bids, n_competitors, mode)
plus the seed.  Changing any of these invalidates the cache for that key.

Output
------
data/picklefiles/clairvoyant_dyn_{key}.pkl  — dict {seed: opt_per_round}
                                              one file per mode
"""

import hashlib
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from environments import AdversarialMultiCampaignEnv
from experiments  import compute_clairvoyant_dynamic_multi, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration -- mirror run_req3.py here so the cache is consistent
# ---------------------------------------------------------------------------

T               = 10_000
VALUES          = [0.8, 0.6, 0.9, 0.7]
BUDGET          = 1_600.0
N_COMPETITORS   = [3, 3, 3, 3]
CONFLICT_EDGES  = [(0, 1), (2, 3)]
AVAILABLE_BIDS  = np.linspace(0, 1, 11)
N_TRIALS        = 20
SEEDS           = list(range(N_TRIALS))

# Non-stationarity patterns to precompute.  Each produces a separate cache
# file with its own hash key.  Trim the list to just one mode if you only
# care about one regime.
MODES           = ["drift", "shocks"]


# ---------------------------------------------------------------------------
# Cache key — a short stable hash of the problem parameters
# ---------------------------------------------------------------------------

def make_cache_key(T, values, budget, conflict_edges, available_bids,
                   n_competitors, env_class_name, mode):
    """Short hex digest that changes whenever any LP-relevant parameter changes."""
    payload = repr((
        env_class_name,
        int(T),
        tuple(float(v) for v in values),
        float(budget),
        tuple(sorted(tuple(e) for e in conflict_edges)),
        tuple(float(b) for b in available_bids),
        tuple(int(n) for n in n_competitors),
        str(mode),
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-mode precomputation
# ---------------------------------------------------------------------------

def precompute_for_mode(mode):
    """Compute (or extend) the cache for a single adversarial mode."""
    env_class = AdversarialMultiCampaignEnv
    key = make_cache_key(
        T=T, values=VALUES, budget=BUDGET, conflict_edges=CONFLICT_EDGES,
        available_bids=AVAILABLE_BIDS, n_competitors=N_COMPETITORS,
        env_class_name=env_class.__name__, mode=mode,
    )
    cache_path = DATA_DIR / f"clairvoyant_dyn_{key}.pkl"

    logger.info("=" * 60)
    logger.info("Mode: %s", mode)
    logger.info("Cache path: %s", cache_path)
    logger.info("Cache key parameters: T=%d N=%d B=%.1f edges=%s env=%s mode=%s",
                T, len(VALUES), BUDGET, CONFLICT_EDGES, env_class.__name__, mode)

    # Load existing cache, if any
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logger.info("Loaded existing cache with %d entries", len(cache))

    # Compute missing seeds
    missing = [s for s in SEEDS if s not in cache]
    if not missing:
        logger.info("Cache already complete for SEEDS=%s — nothing to do.", SEEDS)
        return key, cache

    logger.info("Need to compute %d / %d seeds: %s", len(missing), len(SEEDS), missing)
    t_start = time.time()

    for k, seed in enumerate(missing):
        env = env_class(
            values=VALUES, budget=BUDGET, T=T,
            available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES,
            seed=seed,
            mode=mode,
        )
        t0 = time.time()
        opt_total, opt_per_round = compute_clairvoyant_dynamic_multi(
            m_seq          = env.m,
            values         = env.values,
            bid_sets       = env.bid_sets,
            budget         = env.budget,
            conflict_edges = env.conflict_edges,
        )
        elapsed = time.time() - t0

        cache[seed] = {
            "opt_total":     opt_total,
            "opt_per_round": opt_per_round,
        }
        logger.info("[%2d/%d] mode=%s seed=%d  opt_total=%.3f  per_round=%.4f  (%.1fs)",
                    k + 1, len(missing), mode, seed,
                    opt_total, opt_per_round, elapsed)

        # Persist after every seed: an interrupt does not lose work
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    logger.info("Mode %s done in %.1fs.  Cache: %s (%d entries)",
                mode, time.time() - t_start, cache_path, len(cache))
    return key, cache


# ---------------------------------------------------------------------------
# Main: loop over all modes
# ---------------------------------------------------------------------------

def main():
    t_start_total = time.time()
    keys_by_mode = {}

    for mode in MODES:
        key, _ = precompute_for_mode(mode)
        keys_by_mode[mode] = key

    logger.info("=" * 60)
    logger.info("All modes done in %.1fs total.", time.time() - t_start_total)
    logger.info("")
    logger.info("Cache keys to paste in run_req3.py:")
    for mode, key in keys_by_mode.items():
        logger.info("    %-7s → CLAIRVOYANT_CACHE_KEY = %r", mode, key)
    logger.info("=" * 60)
    return keys_by_mode


if __name__ == "__main__":
    main()
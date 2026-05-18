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
read the pickle and skip the LP entirely.

Usage
-----
    python -m utils.precompute_clairvoyant

The cache key is a hash of the relevant problem parameters:
  (T, N, values, budget, conflict_edges, available_bids, n_competitors)
plus the seed.  Changing any of these invalidates the cache for that key.

Output
------
data/picklefiles/clairvoyant_dyn_{key}.pkl  — dict {seed: opt_per_round}
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

T               = 20_000
VALUES          = [0.8, 0.6, 0.9, 0.7]
BUDGET          = 1_600.0
N_COMPETITORS   = [3, 3, 3, 3]
CONFLICT_EDGES  = [(0, 1), (2, 3)]
AVAILABLE_BIDS  = np.linspace(0, 1, 11)
N_TRIALS        = 20
SEEDS           = list(range(N_TRIALS))


# ---------------------------------------------------------------------------
# Cache key — a short stable hash of the problem parameters
# ---------------------------------------------------------------------------

def make_cache_key(T, values, budget, conflict_edges, available_bids,
                   n_competitors, env_class_name):
    """Short hex digest that changes whenever any LP-relevant parameter changes."""
    payload = repr((
        env_class_name,
        int(T),
        tuple(float(v) for v in values),
        float(budget),
        tuple(sorted(tuple(e) for e in conflict_edges)),
        tuple(float(b) for b in available_bids),
        tuple(int(n) for n in n_competitors),
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_class = AdversarialMultiCampaignEnv
    key = make_cache_key(
        T=T, values=VALUES, budget=BUDGET, conflict_edges=CONFLICT_EDGES,
        available_bids=AVAILABLE_BIDS, n_competitors=N_COMPETITORS,
        env_class_name=env_class.__name__,
    )
    cache_path = DATA_DIR / f"clairvoyant_dyn_{key}.pkl"
    logger.info("Cache path: %s", cache_path)
    logger.info("Cache key parameters: T=%d N=%d B=%.1f edges=%s env=%s",
                T, len(VALUES), BUDGET, CONFLICT_EDGES, env_class.__name__)

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
        return cache

    logger.info("Need to compute %d / %d seeds: %s", len(missing), len(SEEDS), missing)
    t_start = time.time()

    for k, seed in enumerate(missing):
        env = env_class(
            values=VALUES, budget=BUDGET, T=T,
            available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES,
            seed=seed,
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
        logger.info("[%2d/%d] seed=%d  opt_total=%.3f  per_round=%.4f  (%.1fs)",
                    k + 1, len(missing), seed, opt_total, opt_per_round, elapsed)

        # Persist after every seed: an interrupt does not lose work
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    logger.info("Done. Total time: %.1fs", time.time() - t_start)
    logger.info("Cache: %s (%d entries)", cache_path, len(cache))
    return cache


if __name__ == "__main__":
    main()
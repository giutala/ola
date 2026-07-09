"""
precompute_clairvoyant.py
-------------------------
One-shot script: pre-compute and cache the dynamic/prophet clairvoyant for
Requirement 3 (adversarial baseline) across all trial seeds and modes.

The dynamic-clairvoyant LP has ~440k variables for T=10000, N=4, K=11, so
solving it inside the trial loop would take 5–30 s per trial. This script
runs the LPs once and pickles the results; run_primal_dual_trials loads the
cache and skips the LP entirely on subsequent runs.

All problem parameters are imported from req3_config.py (shared with
run_req3.py) so the two scripts can never use inconsistent parameters.
The cache key is a hash of all LP-relevant parameters; any change
automatically invalidates the cache (clean fallback to on-the-fly LP).

Usage
-----
    Run from the utils/ directory (uses bare imports):

        cd utils/
        python precomputed_clairvoyant.py

Output
------
data/picklefiles/clairvoyant_dyn_{key}.pkl — dict {seed: {opt_total, opt_per_round}}
"""

import logging
import pickle
import time

from environments import AdversarialMultiCampaignEnv
from experiments  import compute_clairvoyant_dynamic_multi, DATA_DIR
from req3_config  import (
    VALUES, T, BUDGET, N_COMPETITORS, CONFLICT_EDGES, AVAILABLE_BIDS,
    N_TRIALS, make_cache_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration -- imported from req3_config.py so it is always in sync
# with run_req3.py.  Only the modes to precompute are local to this script.
# ---------------------------------------------------------------------------

SEEDS = list(range(N_TRIALS))

# Non-stationarity patterns to precompute.  Each produces a separate cache
# file with its own hash key.  Trim the list to just one mode if you only
# care about one regime.
MODES = ["drift", "shocks"]


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
"""
precompute_clairvoyant_req4.py
------------------------------
One-shot script: pre-compute and cache the dynamic/prophet clairvoyant for
Requirement 4 (mode='shocks') across all trial seeds, so run_req4.py does
not need to solve the full LP (~440k variables for T=10000, N=4, K=11) on
every re-run.

All problem parameters come from req4_config.py (which imports the shared
base from req3_config.py), so this script and run_req4.py are always in
sync. The cache key is a hash of all LP-relevant parameters; a parameter
change automatically invalidates the cache.

Usage
-----
    python -m utils.precompute_clairvoyant_req4
"""

import logging
import pickle
import time

from utils.environments import AdversarialMultiCampaignEnv
from utils.experiments import compute_clairvoyant_dynamic_multi, DATA_DIR
from utils.req4_config import (
    VALUES, T, BUDGET, CONFLICT_EDGES, AVAILABLE_BIDS, N_TRIALS,
    SHOCKS_MODE_PARAMS, make_cache_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = list(range(N_TRIALS))


def main():
    key = make_cache_key(mode="shocks", mode_params=SHOCKS_MODE_PARAMS)
    cache_path = DATA_DIR / f"clairvoyant_dyn_{key}.pkl"
    logger.info("=" * 60)
    logger.info("Requirement 4 dynamic clairvoyant | mode=shocks params=%s -> %s",
                SHOCKS_MODE_PARAMS, cache_path)

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
        env = AdversarialMultiCampaignEnv(
            values=VALUES, budget=BUDGET, T=T, available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode="shocks",
            **SHOCKS_MODE_PARAMS,
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

    logger.info("=" * 60)
    logger.info("CLAIRVOYANT_CACHE_KEY = %r  (run_req4.py reads this via req4_config.make_cache_key,"
                " no manual paste needed)", key)
    return key


if __name__ == "__main__":
    main()

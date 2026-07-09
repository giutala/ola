"""
req3_config.py
---------------
Single source of truth for Requirement 3 parameters.

Both run_req3.py and precompute_clairvoyant.py import from here instead of
keeping their own copies. This is what caused the negative-regret bug:
precompute_clairvoyant.py had its own VALUES/BUDGET constants and went
stale after run_req3.py's were tuned, so the cached dynamic clairvoyant
was solved for a 3x tighter budget than the one actually used at run time
-- an artificially low baseline that made the agent's regret go negative.

make_cache_key() is also centralised here so the hash is always computed
the same way on both sides: same parameters -> same key, always. If a
parameter changes, the key changes automatically and a stale cache simply
misses (clean fallback to the on-the-fly LP) instead of being loaded
silently.
"""

import hashlib

import numpy as np

VALUES         = [0.8, 0.8, 0.9, 0.9]
T              = 10_000
BUDGET         = 1600.0
N_TRIALS       = 20
N_COMPETITORS  = [3, 3, 3, 3]
CONFLICT_EDGES = [(0, 1), (2, 3)]
AVAILABLE_BIDS = np.linspace(0, 1, 11)

ADVERSARIAL_ENV_CLASS_NAME = "AdversarialMultiCampaignEnv"


def make_cache_key(
    mode,
    T=T,
    values=VALUES,
    budget=BUDGET,
    conflict_edges=CONFLICT_EDGES,
    available_bids=AVAILABLE_BIDS,
    n_competitors=N_COMPETITORS,
    env_class_name=ADVERSARIAL_ENV_CLASS_NAME,
):
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

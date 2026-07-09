"""
req3_config.py
--------------
Single source of truth for all multi-campaign problem parameters.

All four requirements share the same T, BUDGET, N, VALUES, CONFLICT_EDGES,
AVAILABLE_BIDS, and N_COMPETITORS so that regret curves remain directly
comparable across requirements. run_req2.py, run_req3.py, run_req4.py, and
the precompute scripts all import from here.

make_cache_key() produces a short hex digest of the LP-relevant parameters
so that a parameter change automatically invalidates cached dynamic clairvoyants
instead of silently loading a stale result.
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
    mode: str,
    T: int = T,
    values=VALUES,
    budget: float = BUDGET,
    conflict_edges=CONFLICT_EDGES,
    available_bids=AVAILABLE_BIDS,
    n_competitors=N_COMPETITORS,
    env_class_name: str = ADVERSARIAL_ENV_CLASS_NAME,
) -> str:
    """Short 12-character hex digest of all LP-relevant parameters."""
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

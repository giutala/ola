"""
run_req4.py
-----------
Requirement 4: slightly non-stationary environment, multiple campaigns.

DEFINITIVE version: built on Requirement 2's CombinatorialUCBAgent AFTER
the empirical-cost fix (see Req1_Empirical_Cost_Justification.pdf and
agents.py's module docstring) -- both agents below inherit that fix
directly through subclassing, so neither reproduces the early-budget-
exhaustion pathology the pre-fix agent had.

Environment: NonStationaryMultiCampaignEnv(mode='shocks'), with FEW, LONG
blocks (block_size=2000 -> 5 intervals over T=10000) -- "slightly" non-
stationary, contrasted with Requirement 3's 'drift' mode which moves every
round. Same underlying mechanism as Requirement 3's environment class,
just a different parameterisation.

Three bidding strategies on the SAME environment (project slide 19):
  1. SlidingWindowCombinatorialUCBAgent   (Requirement 2 + Practical/10 SW-UCB)
  2. CUSUMCombinatorialUCBAgent            (Requirement 2 + Practical/10 CUSUM-UCB)
  3. PrimalDualMultiCampaignAgent          (Requirement 3, reused unchanged)

Baselines: the legacy dynamic realised clairvoyant is still reported for
continuity with Requirement 3, but the main diagnostic plot for Requirement 4
uses the piecewise expected clairvoyant. That benchmark knows the stationary
blocks and their distributions, while not observing the realised competing
bids round by round.

Call from the notebook: run_req4()
"""

import logging

import numpy as np

from utils.agents import (
    SlidingWindowCombinatorialUCBAgent,
    CUSUMCombinatorialUCBAgent,
    PrimalDualMultiCampaignAgent,
)
from utils.environments import NonStationaryMultiCampaignEnv
from utils.experiments import (
    load_clairvoyant_cache, plot_regret, plot_budget, plot_resets_histogram,
    run_nonstationary_trials,
)

logger = logging.getLogger(__name__)

VALUES = [0.8, 0.6, 0.9, 0.7]
T = 10_000
BUDGET = 1_600.0
N_TRIALS = 20
CONFLICT_EDGES = [(0, 1), (2, 3)]
AVAILABLE_BIDS = np.linspace(0, 1, 11)

N_INTERVALS = 5
BLOCK_SIZE = T // N_INTERVALS   # 2000 -- "slightly" vs Req 3's every-round drift
U_T = N_INTERVALS - 1           # upper bound on regime changes, feeds CUSUM

# Set from precompute_clairvoyant.py's printed key for
# mode='shocks', block_size=2000, n_regimes=5.
CLAIRVOYANT_CACHE_KEY = "d29cd9b737b5"

# Sliding-window length: tuned to the interval length, NOT the textbook
# W=2*sqrt(T) rule of thumb. With sum(K_i)=32 cells (much larger than
# Practical/10's toy K=3 example), W=2*sqrt(T)~200 forgets under-sampled
# cells before they are well estimated and keeps re-exploring them even
# absent any real regime change -- pure window churn. W ~= one interval
# length performed best in side-by-side tests (see README).
SW_WINDOW = BLOCK_SIZE


def run_req4():
    logger.info("=" * 60)
    logger.info("Requirement 4 - Slightly Non-Stationary, Multiple Campaigns")
    logger.info("=" * 60)
    logger.info("Parameters | N=%d T=%d B=%.1f rho=%.4f n_intervals=%d block_size=%d",
                len(VALUES), T, BUDGET, BUDGET / T, N_INTERVALS, BLOCK_SIZE)

    _env_ref = NonStationaryMultiCampaignEnv(
        values=VALUES, budget=BUDGET, T=T, available_bids=AVAILABLE_BIDS,
        conflict_edges=CONFLICT_EDGES, seed=0, mode="shocks",
        block_size=BLOCK_SIZE, n_regimes=N_INTERVALS,
    )
    N, Ks, bid_sets = _env_ref.N, _env_ref.Ks, _env_ref.bid_sets

    def env_factory(seed):
        return NonStationaryMultiCampaignEnv(
            values=VALUES, budget=BUDGET, T=T, available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode="shocks",
            block_size=BLOCK_SIZE, n_regimes=N_INTERVALS,
        )

    def make_sw_agent():
        return SlidingWindowCombinatorialUCBAgent(
            N=N, Ks=Ks, T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES, W=SW_WINDOW,
        )

    def make_cusum_agent():
        return CUSUMCombinatorialUCBAgent(
            N=N, Ks=Ks, T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES, U_T=U_T,
        )

    def make_pd_agent():
        return PrimalDualMultiCampaignAgent(
            N=N, Ks=Ks, bid_sets=bid_sets, T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES,
        )

    cache = load_clairvoyant_cache(CLAIRVOYANT_CACHE_KEY) if CLAIRVOYANT_CACHE_KEY else {}

    logger.info("-" * 60); logger.info("Sliding-Window Combinatorial-UCB")
    res_sw = run_nonstationary_trials(env_factory, make_sw_agent, N_TRIALS,
                                       name="req4_sw_cucb", clairvoyant_cache=cache or None)

    logger.info("-" * 60); logger.info("CUSUM Combinatorial-UCB")
    res_cusum = run_nonstationary_trials(env_factory, make_cusum_agent, N_TRIALS,
                                          name="req4_cusum_cucb", clairvoyant_cache=cache or None)

    logger.info("-" * 60); logger.info("Primal-Dual (Requirement 3, reused unchanged)")
    res_pd = run_nonstationary_trials(env_factory, make_pd_agent, N_TRIALS,
                                       name="req4_primal_dual", clairvoyant_cache=cache or None)

    results = {
        "Sliding-Window Combinatorial-UCB": res_sw,
        "CUSUM Combinatorial-UCB": res_cusum,
        "Primal-Dual (Req 3)": res_pd,
    }

    plot_regret(results=results, title="Req 4 - Cumulative Pseudo-Regret: Slightly Non-Stationary",
                filename="r4/req4_regret.png", add_reference=False)
    piecewise_results = {
        label: {
            **res,
            "mean_regret": res["mean_regret_piecewise"],
            "std_regret": res["std_regret_piecewise"],
        }
        for label, res in results.items()
        if "mean_regret_piecewise" in res
    }
    if piecewise_results:
        plot_regret(results=piecewise_results,
                    title="Req 4 - Cumulative Regret vs Piecewise Expected Clairvoyant",
                    filename="r4/req4_regret_piecewise_expected.png",
                    add_reference=False)
    plot_budget(results=results, budget=BUDGET, title="Req 4 - Cumulative Cost",
                filename="r4/req4_budget.png")

    if "resets_per_trial" in res_cusum:
        plot_resets_histogram(res_cusum["resets_per_trial"],
                               title="Req 4 - How often did the CUSUM detector fire?",
                               filename="r4/req4_cusum_resets.png")

    logger.info("=" * 60)
    logger.info("Final pseudo-regret (mean over %d trials):", N_TRIALS)
    logger.info("  Sliding-Window CUCB : %.2f", res_sw["mean_regret"][-1])
    logger.info("  CUSUM CUCB          : %.2f", res_cusum["mean_regret"][-1])
    logger.info("  Primal-Dual (Req 3) : %.2f", res_pd["mean_regret"][-1])
    if "mean_regret_piecewise" in res_sw:
        logger.info("Final regret vs piecewise expected clairvoyant:")
        logger.info("  Sliding-Window CUCB : %.2f", res_sw["mean_regret_piecewise"][-1])
        logger.info("  CUSUM CUCB          : %.2f", res_cusum["mean_regret_piecewise"][-1])
        logger.info("  Primal-Dual (Req 3) : %.2f", res_pd["mean_regret_piecewise"][-1])
    if "mean_resets" in res_cusum:
        logger.info("Mean CUSUM resets per trial: %.1f", res_cusum["mean_resets"])
    logger.info("Final cumulative cost:")
    logger.info("  Sliding-Window CUCB : %.2f / %.0f", res_sw["mean_cumcost"][-1], BUDGET)
    logger.info("  CUSUM CUCB          : %.2f / %.0f", res_cusum["mean_cumcost"][-1], BUDGET)
    logger.info("  Primal-Dual (Req 3) : %.2f / %.0f", res_pd["mean_cumcost"][-1], BUDGET)
    logger.info("=" * 60)
    logger.info("Requirement 4 complete.")

    return {"sw": res_sw, "cusum": res_cusum, "pd": res_pd}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run_req4()
